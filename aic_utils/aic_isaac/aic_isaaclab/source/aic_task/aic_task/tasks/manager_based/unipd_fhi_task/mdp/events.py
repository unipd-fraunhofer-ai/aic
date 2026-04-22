from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


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


class reset_board_and_robot(ManagerTermBase ):
    """Reset and randomize the task board + parts and place the robot above it using IK."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._cached_orientations: dict[str, torch.Tensor] = {}

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="wrist_3_link"),
        board_scene_name: str = "task_board",
        board_default_pos: tuple = (0.2837, 0.229, 0.0),
        board_range: dict = {"x": (0.0, 0.0), "y": (0.0, 0.0)},
        parts: list[dict] = (),
        ee_offset_range: dict = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.15, 0.25)},
        ee_tilt_range: float = 10.0,
    ):
        device = env.device
        n = len(env_ids)
        env_origins = env.scene.env_origins[env_ids]

        # 1. Reset board and parts (logic from aic 'randomize_board_and_parts' function)
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

        # 2. Reset Robot above the board (same logic as reset in gear assembly task)
        robot_cfg.resolve(env.scene)
        robot = env.scene[robot_cfg.name]
        ee_body_idx = robot_cfg.body_ids[0]

        # Sample target EE pose in world frame
        target_pos = board_world_pos.clone()
        off_x = torch.empty(n, device=device).uniform_(*ee_offset_range.get("x", (-0.1, 0.1)))
        off_y = torch.empty(n, device=device).uniform_(*ee_offset_range.get("y", (-0.1, 0.1)))
        off_z = torch.empty(n, device=device).uniform_(*ee_offset_range.get("z", (0.15, 0.25)))
        target_pos[:, 0] += off_x
        target_pos[:, 1] += off_y
        target_pos[:, 2] += off_z

        # Target orientation: pointing down (Z axis parallel to world Z but opposite)
        # Nominal down: rotate 180 deg around X axis
        nominal_quat = math_utils.quat_from_euler_xyz(
            torch.tensor(torch.pi, device=device),
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device)
        ).repeat(n, 1)

        # Random tilt
        if ee_tilt_range > 0:
            tilt_angle = torch.empty(n, device=device).uniform_(0, math.radians(ee_tilt_range))
            # Random horizontal axis (in X-Y plane)
            tilt_axis_angle = torch.empty(n, device=device).uniform_(0, 2 * torch.pi)
            tilt_axis = torch.stack([
                torch.cos(tilt_axis_angle),
                torch.sin(tilt_axis_angle),
                torch.zeros(n, device=device)
            ], dim=-1)
            tilt_quat = math_utils.quat_from_angle_axis(tilt_angle, tilt_axis)
            target_quat = math_utils.quat_mul(tilt_quat, nominal_quat)
        else:
            target_quat = nominal_quat
        
        lambda_val = 0.1
        joint_pos_des = robot.data.default_joint_pos[env_ids].clone()
        for i in range(20):
            # Write current guess to sim
            robot.write_joint_state_to_sim(joint_pos_des, torch.zeros_like(joint_pos_des), env_ids=env_ids)
            
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

            jacobian = robot.root_physx_view.get_jacobians()[env_ids, ee_body_idx - 1, :, :]
            
            # DLS Solve: delta_q = J^T (J J^T + lambda^2 I)^-1 delta_x
            jacobian_T = torch.transpose(jacobian, 1, 2)
            lambda_matrix = (lambda_val**2) * torch.eye(6, device=device)
            delta_q = (jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)).squeeze(-1)
            
            # Update joint positions guess
            joint_pos_des = joint_pos_des + delta_q

        # Final write to sim
        robot.write_joint_state_to_sim(joint_pos_des, torch.zeros_like(joint_pos_des), env_ids=env_ids)
        # Also set targets for the next step
        robot.set_joint_position_target(joint_pos_des, env_ids=env_ids)
        robot.set_joint_velocity_target(torch.zeros_like(joint_pos_des), env_ids=env_ids)

