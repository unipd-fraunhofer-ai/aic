from __future__ import annotations

import os
import re
import math
import random
from typing import TYPE_CHECKING

import torch
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Matches the regex form Isaac Lab uses to instantiate per-env prim paths.
_ENV_REGEX_RE = re.compile(r"env_(?:\.\*|\[\^/\]\*)")


def _sample_axis(pose_range: dict, snap_step: dict, axis: str) -> float:
    """Sample a random offset for an axis. If snap_step has a value for this axis,
    snap to the nearest multiple of that step within the range."""
    lo, hi = pose_range.get(axis, (0.0, 0.0))
    step = snap_step.get(axis, 0.0)
    if step > 0 and (hi - lo) > 0:
        n_lo = math.ceil(lo / step)
        n_hi = math.floor(hi / step)
        n = random.randint(n_lo, n_hi)
        return n * step
    return torch.empty(1).uniform_(lo, hi).item()

def _write_usd_xform_pose(
    stage,
    prim_path_template: str,
    env_ids: torch.Tensor,
    env_origins: torch.Tensor,
    world_pos: torch.Tensor,
    world_rot: torch.Tensor,
) -> None:
    """Mirror a per-env rigid body pose onto its USD Xform.

    The prim translate is authored relative to its env root, so the world
    position is converted to env-local coordinates before writing.
    """
    ids = env_ids.tolist()
    local_pos = (world_pos - env_origins).tolist()
    rot = world_rot.tolist()

    for i, env_id in enumerate(ids):
        prim_path = _ENV_REGEX_RE.sub(f"env_{env_id}", prim_path_template)
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue

        xf = UsdGeom.Xformable(prim)
        tx, ty, tz = local_pos[i]
        qw, qx, qy, qz = rot[i]

        for op in xf.GetOrderedXformOps():
            name = op.GetOpName()
            if "translate" in name:
                if op.GetTypeName() == Sdf.ValueTypeNames.Float3:
                    op.Set(Gf.Vec3f(tx, ty, tz))
                else:
                    op.Set(Gf.Vec3d(tx, ty, tz))
            elif "orient" in name:
                if op.GetTypeName() == Sdf.ValueTypeNames.Quatf:
                    op.Set(Gf.Quatf(qw, qx, qy, qz))
                else:
                    op.Set(Gf.Quatd(qw, qx, qy, qz))


class reset_board_and_robot(ManagerTermBase ):
    """Reset and randomize the task board + parts and place the robot above it using IK."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._cached_orientations: dict[str, torch.Tensor] = {}

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        # Board and parts configuration
        board_scene_name: str = "task_board",
        board_default_pos: tuple = (0.2837, 0.229, 0.0),
        board_range: dict = {"x": (0.0, 0.0), "y": (0.0, 0.0)},
        parts: list[dict] = (),
        sync_usd_xforms: bool = True,
        # Robot configuration
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="sfp_tip_link"),
        arm_joint_names: list[str] = ["shoulder.*", "elbow.*", "wrist.*"],
        ee_pose_command_name: str = "sfp_port_pose_command",
        ee_offset_range: dict = {
            "x": (-0.02, 0.02), 
            "y": (-0.02, 0.02), 
            "z": (0.08, 0.11), 
            "roll": (-10.0, 10.0),
            "pitch": (-10.0, 10.0),
            "yaw": (-1.0, 1.0),
        },
    ):
        device = env.device
        n = len(env_ids)
        env_origins = env.scene.env_origins[env_ids]
        stage = omni.usd.get_context().get_stage() if sync_usd_xforms else None

        # 1. Reset board and parts (logic from aic 'randomize_board_and_parts' function)
        # - Board pose
        board_asset = env.scene[board_scene_name]
        all_names = [board_scene_name] + [p["scene_name"] for p in parts]
        if not self._cached_orientations:
            for name in all_names:
                asset = env.scene[name]
                self._cached_orientations[name] = asset.data.root_state_w[:, 3:7].clone()

        board_rot = self._cached_orientations[board_scene_name][env_ids]
        board_pos = torch.tensor([board_default_pos], device=device).expand(n, -1).clone()
        bx_off = torch.empty(n, device=device).uniform_(*board_range.get("x", (0.0, 0.0)))
        by_off = torch.empty(n, device=device).uniform_(*board_range.get("y", (0.0, 0.0)))
        board_pos[:, 0] += bx_off
        board_pos[:, 1] += by_off

        board_world_pos = board_pos + env_origins
        board_pose = torch.cat([board_world_pos, board_rot], dim=-1)
        board_asset.write_root_pose_to_sim(board_pose, env_ids=env_ids)
        board_asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)

        if sync_usd_xforms:
            _write_usd_xform_pose(
                stage,
                board_asset.cfg.prim_path,
                env_ids,
                env_origins,
                board_world_pos,
                board_rot,
            )

        # - Part poses, anchored to the board
        for part_cfg in parts:
            pname = part_cfg["scene_name"]
            part_asset = env.scene[pname]
            part_rot = self._cached_orientations[pname][env_ids]
            ox, oy, oz = part_cfg["offset"]
            pr = part_cfg.get("pose_range", {})
            snap = part_cfg.get("snap_step", {})

            part_pos = board_world_pos.clone()
            for idx in range(n):
                dx = _sample_axis(pr, snap, "x")
                dy = _sample_axis(pr, snap, "y")
                part_pos[idx, 0] += ox + dx
                part_pos[idx, 1] += oy + dy
                part_pos[idx, 2] = board_world_pos[idx, 2] + oz

            part_pose = torch.cat([part_pos, part_rot], dim=-1)
            part_asset.write_root_pose_to_sim(part_pose, env_ids=env_ids)
            part_asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)

            if sync_usd_xforms:
                _write_usd_xform_pose(
                    stage,
                    part_asset.cfg.prim_path,
                    env_ids,
                    env_origins,
                    part_pos,
                    part_rot,
                )

        # 2. Reset Robot above the board using iterative IK
        robot_cfg.resolve(env.scene)
        robot = env.scene[robot_cfg.name]
        ee_body_idx = robot_cfg.body_ids[0]

        arm_joint_ids, arm_joint_names_resolved = robot.find_joints(arm_joint_names)
        arm_joint_ids = torch.as_tensor(arm_joint_ids, device=device, dtype=torch.long)
        num_arm_joints = len(arm_joint_ids)

        if num_arm_joints == 0:
            raise RuntimeError(f"No arm joints found from patterns: {arm_joint_names}")

        # Get target pose from command manager
        command_term = env.command_manager.get_term(ee_pose_command_name)
        # Update command for the newly reset envs
        command_term._update_command()
        base_target_pos = command_term.poses_w[env_ids, 0:3]
        base_target_quat = command_term.poses_w[env_ids, 3:7]

        # Randomize EE offset around the target position
        pos_offset = torch.zeros((n, 3), device=device)
        pos_offset[:, 0] = torch.empty(n, device=device).uniform_(*ee_offset_range.get("x", (0.0, 0.0)))
        pos_offset[:, 1] = torch.empty(n, device=device).uniform_(*ee_offset_range.get("y", (0.0, 0.0)))
        pos_offset[:, 2] = torch.empty(n, device=device).uniform_(*ee_offset_range.get("z", (0.0, 0.0)))

        roll = torch.empty(n, device=device).uniform_(
            math.radians(ee_offset_range.get("roll", (0.0, 0.0))[0]),
            math.radians(ee_offset_range.get("roll", (0.0, 0.0))[1]),
        )
        pitch = torch.empty(n, device=device).uniform_(
            math.radians(ee_offset_range.get("pitch", (0.0, 0.0))[0]),
            math.radians(ee_offset_range.get("pitch", (0.0, 0.0))[1]),
        )
        yaw = torch.empty(n, device=device).uniform_(
            math.radians(ee_offset_range.get("yaw", (0.0, 0.0))[0]),
            math.radians(ee_offset_range.get("yaw", (0.0, 0.0))[1]),
        )
        rpy_offset_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

        target_pos = base_target_pos + math_utils.quat_apply(base_target_quat, pos_offset)
        target_quat = math_utils.quat_mul(base_target_quat, rpy_offset_quat)
        
        lambda_val = 0.1
        joint_pos_arm_des = robot.data.default_joint_pos[env_ids][:, arm_joint_ids].clone()
        joint_vel_arm_des = torch.zeros_like(joint_pos_arm_des)
        for i in range(20):
            # Write current guess to sim            
            robot.write_joint_state_to_sim(
                joint_pos_arm_des,
                joint_vel_arm_des,
                joint_ids=arm_joint_ids,
                env_ids=env_ids,
            )

            env.sim.forward()
            robot.update(0.0)

            # Compute pose error in world frame
            ee_pose_w = robot.data.body_link_pose_w[env_ids, ee_body_idx]
            ee_pos_w = ee_pose_w[:, :3]
            ee_quat_w = ee_pose_w[:, 3:7]            
            pos_error = target_pos - ee_pos_w
            
            # Orientation error in world frame
            # ensure shortest path
            dot = (target_quat * ee_quat_w).sum(dim=-1, keepdim=True)
            q_target_alt = torch.where(dot >= 0, target_quat, -target_quat)
            q_error = math_utils.quat_mul(q_target_alt, math_utils.quat_inv(ee_quat_w))
            axis_angle_error = math_utils.axis_angle_from_quat(q_error)
            
            delta_pose = torch.cat([pos_error, axis_angle_error], dim=-1)
            
            # Check convergence
            pos_err_norm = torch.norm(pos_error, dim=-1)
            rot_err_norm = torch.norm(axis_angle_error, dim=-1)
            
            if torch.all(pos_err_norm <= 1e-3) and torch.all(rot_err_norm <= 1e-3):
                break

            full_jacobian = robot.root_physx_view.get_jacobians()[env_ids, ee_body_idx - 1, :, :]
            jacobian = full_jacobian[:, :, arm_joint_ids]
            
            # DLS Solve: delta_q = J^T (J J^T + lambda^2 I)^-1 delta_x
            jacobian_T = torch.transpose(jacobian, 1, 2)
            lambda_matrix = (lambda_val**2) * torch.eye(6, device=device)
            delta_q = (jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)).squeeze(-1)
            
            # Update joint positions guess
            joint_pos_arm_des = joint_pos_arm_des + delta_q

        # Final write to sim
        robot.write_joint_state_to_sim(joint_pos_arm_des, torch.zeros_like(joint_pos_arm_des), joint_ids=arm_joint_ids, env_ids=env_ids)
        # Also set targets for the next step
        robot.set_joint_position_target(joint_pos_arm_des, joint_ids=arm_joint_ids, env_ids=env_ids)
        robot.set_joint_velocity_target(torch.zeros_like(joint_pos_arm_des), joint_ids=arm_joint_ids, env_ids=env_ids)


class reset_to_near_completion(ManagerTermBase):
    """Reset the robot and NIC card to a pre-collected 'near completion' state."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.data = None
        self.success_indices = None
        self.command_term = None
            
    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        reset_data_filename: str,
        robot_scene_name: str = "robot",
        nic_card_scene_name: str = "nic_card",
        partially_inserted_prob: float = 0.0,
    ):
        # Lazy load the data
        if self.data is None:
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                data_path = os.path.join(base_dir, "..", "data", reset_data_filename)
                self.data = torch.load(data_path, map_location=env.device)
                print(f"[reset_to_near_completion] Loaded {self.data['num_samples']} states from {data_path}")
                
                if "success" in self.data:
                    self.success_indices = torch.where(self.data["success"] > 0.5)[0]
                    print(f"  - Found {len(self.success_indices)} successful states out of {self.data['num_samples']}")
                else:
                    self.success_indices = torch.arange(self.data['num_samples'], device=env.device)
            except Exception as e:
                raise RuntimeError(f"Failed to load data from {data_path}: {e}")

        device = env.device
        n = len(env_ids)
        env_origins = env.scene.env_origins[env_ids]

        use_success = torch.rand(n, device=device) < partially_inserted_prob
        sample_ids = torch.empty(n, dtype=torch.long, device=device)

        # Handle environments resetting to success states
        if use_success.any():
            sub_ids = torch.randint(0, len(self.success_indices), (use_success.sum(),), device=device)
            sample_ids[use_success] = self.success_indices[sub_ids]

        # Handle environments resetting to any state
        if (~use_success).any():
            sample_ids[~use_success] = torch.randint(0, self.data["num_samples"], ((~use_success).sum(),), device=device)
        
        # 1. Reset Robot
        robot = env.scene[robot_scene_name]
        
        # Root state (position + orientation + velocities)
        # We assume the saved root_state is relative to env_origin (positions only)
        robot_root_state = self.data["robot_root_state"][sample_ids].clone()
        robot_root_state[:, :3] += env_origins
        robot.write_root_pose_to_sim(robot_root_state[:, :7], env_ids=env_ids)
        robot.write_root_velocity_to_sim(robot_root_state[:, 7:13], env_ids=env_ids)
        
        # Joint state (positions + velocities)
        joint_pos = self.data["robot_joint_pos"][sample_ids]
        joint_vel = self.data["robot_joint_vel"][sample_ids]
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        # Set targets for the next step (important for position/impedance control)
        robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)

        # 2. Reset NIC Card
        nic_card = env.scene[nic_card_scene_name]
        nic_root_state = self.data["nic_card_root_state"][sample_ids].clone()
        nic_root_state[:, :3] += env_origins
        nic_card.write_root_pose_to_sim(nic_root_state[:, :7], env_ids=env_ids)
        nic_card.write_root_velocity_to_sim(nic_root_state[:, 7:13], env_ids=env_ids)

        # 3. Enforce consistent command target
        if self.command_term is None:
            self.command_term = env.command_manager.get_term(self.cfg.params.get("command_name", "sfp_port_pose_command"))
            self.command_term.pending_targets_idx = torch.full((env.num_envs,), -1, dtype=torch.long, device=device)        
        self.command_term.pending_targets_idx[env_ids] = self.data["target_idx"][sample_ids].to(device)
