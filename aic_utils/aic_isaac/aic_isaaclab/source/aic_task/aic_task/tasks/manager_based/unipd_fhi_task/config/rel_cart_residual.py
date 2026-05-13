from __future__ import annotations

import torch
import numpy as np
from collections.abc import Sequence

import isaaclab.utils.math as math_utils
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.envs.mdp import JointEffortActionCfg
from isaaclab.utils import configclass

from ..aic_task_base_env_cfg import AICTaskBaseEnvCfg
from ..aic_task_base_env import AICTaskBaseEnv
from .. import mdp

from .insertion_heuristic_logic import InsertionHeuristicLogic

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    # Define 12-dim action space for the policy: [pos(3), ori(3), stiffness(6)]
    # Use two dummy joint effort action with scale 0 to define the shape.
    arm_action: ActionTerm = JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        scale=0.0, 
    )
    stiffness_action: ActionTerm = JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        scale=0.0,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP: robot state, ee pose, pose command."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy: joint state, ee pose, pose command."""

        # Minimal target port position and orientation (x, y, yaw = 3 dims)
        port_target = ObsTerm(
            func=mdp.target_port_base_heuristic,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # End effector position, orientation, linear velocity, angular velocity (12 dims)
        ee_pos = ObsTerm(
            func=mdp.ee_pos_base,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_tcp")},
        )
        ee_rpy = ObsTerm(
            func=mdp.ee_rpy_base,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_tcp")},
        )
        ee_lin_vel = ObsTerm(
            func=mdp.ee_lin_vel_base,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_tcp")},
        )
        ee_ang_vel = ObsTerm(
            func=mdp.ee_ang_vel_base,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_tcp")},
        )

        # Body forces (wrench) at the end-effector (force xyz + torque xyz = 6 dims)
        body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"])
            },
        )

        # Last action (6 dims)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True # Total obs dim = 3 + 12 + 6 + 6 = 27

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    reset_scene = EventTerm(
         func=mdp.reset_board_and_robot,
            mode="reset",
            params={
                "board_scene_name": "task_board",
                "board_default_pos": (0.15, -0.2, 1.14),
                "board_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                "parts": [
                    {
                        "scene_name": "nic_card",
                        "offset": (-0.03235, 0.02329, 0.0743),
                        "pose_range": {"y": (0.0, 0.12)},
                        "snap_step": {"y": 0.04},
                    },
                ],
                "ee_pose_command_name": "sfp_port_pose_command",
                "ee_offset_range": {
                    "x": (0.0, 0.0), 
                    "y": (0.0, 0.0), 
                    "z": (0.07, 0.07), 
                    "roll": (15.0, 15.0),
                    "pitch": (15.0, 15.0),
                    "yaw": (0.0, 0.0),
                },
            },
    )


##
# Env definition
##


@configclass
class RelCartesianOSPResidualEnvCfg(AICTaskBaseEnvCfg):
    """Configuration for Residual RL Insertion environment."""

    # Residual scaling
    residual_pos_scale: float = 0.01  # ±1 cm
    residual_ori_scale: float = 0.05  # ~3 degrees
    residual_stiffness_scale: float = 50.0 # ±50 N/m
    osc_impedance_mode: str = "variable_kp"

    # Port noise
    pos_std: float = 0.003
    rot_std: float = float(np.deg2rad(7))

    # MDP settings
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()


class RelCartesianOSPResidualEnv(AICTaskBaseEnv):
    """RL env that combines a heuristic base command with policy residual corrections."""

    cfg: RelCartesianOSPResidualEnvCfg

    def __init__(self, cfg: RelCartesianOSPResidualEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        
        # Calculate tip to gripper transform
        self._pos_tg, self._quat_tg = self._get_T_tip_gripper()

        self._heuristic = InsertionHeuristicLogic(
            self.num_envs, 
            self.device, 
            self._pos_tg, 
            self._quat_tg
        )
        
        # Get body id for sfp_tip_link for force reading
        self._tip_body_id, _ = self.scene["robot"].find_bodies("sfp_tip_link")

        # Initial inputs for heuristic
        self._target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 4)
        self._h_tip_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._h_tip_force_z = torch.zeros(self.num_envs, device=self.device)
        self._update_heuristic_inputs()
        self._set_heuristic_targets()

    def _get_T_tip_gripper(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates the relative transform from tip to gripper."""
        # Ensure data is updated
        self.scene.update(dt=0.0)
        
        robot = self._robot
        tip_sensor = self.scene.sensors["sfp_tip_sensor"]
        
        gripper_pos = robot.data.body_pos_w[:, self._ee_body_idx, :]
        gripper_quat = robot.data.body_quat_w[:, self._ee_body_idx, :]
        
        tip_pos = tip_sensor.data.target_pos_w[:, 0]
        tip_quat = tip_sensor.data.target_quat_w[:, 0]

        pos_rel, quat_rel = math_utils.subtract_frame_transforms(tip_pos, tip_quat, gripper_pos, gripper_quat)
        return pos_rel.to(dtype=torch.float32), quat_rel.to(dtype=torch.float32)

    def _update_heuristic_inputs(self, env_ids: Sequence[int] | None = None) -> None:
        """Queries sensors and updates internal buffers for heuristic inputs."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        # 1. Tip position (world frame)
        tip_sensor = self.scene.sensors["sfp_tip_sensor"]
        self._h_tip_pos[env_ids] = tip_sensor.data.target_pos_w[env_ids, 0].to(dtype=torch.float32)
        
        # 2. Tip force (from raw articulation data)
        tip_wrench = mdp.body_incoming_wrench(self, SceneEntityCfg("robot", body_ids=self._tip_body_id)).abs()
        self._h_tip_force_z[env_ids] = tip_wrench[env_ids, 2]

    def _set_heuristic_targets(self, env_ids: Sequence[int] | None = None) -> None:
        """Sets the targets for the heuristic."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # 3. Port data (world frame)
        port_sensor = self.scene.sensors["sfp_port_sensor"]
        target_idx = self.command_manager.get_term("sfp_port_pose_command").targets_idx

        port_pos = port_sensor.data.target_pos_w[env_ids, target_idx[env_ids]].to(dtype=torch.float32)
        port_quat = port_sensor.data.target_quat_w[env_ids, target_idx[env_ids]].to(dtype=torch.float32)

        port_pos, port_quat = self._add_noise_to_port(port_pos, port_quat)
        
        self._target_pos_w[env_ids] = port_pos
        self._target_quat_w[env_ids] = port_quat
        self._heuristic.set_target(port_pos, port_quat, env_ids=env_ids)
    
    def _add_noise_to_port(self, pos: torch.Tensor, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Adds Gaussian/Uniform noise to port pose as in emp_cheat.py."""
        num_envs = pos.shape[0]
        
        # Translation noise: positive uniform in X and Y
        noise_x = torch.rand(num_envs, device=self.device) * self.cfg.pos_std
        noise_y = torch.rand(num_envs, device=self.device) * self.cfg.pos_std
        
        pos[:, 0] += noise_x
        pos[:, 1] += noise_y
        
        # Rotation noise: uniform yaw noise
        yaw_noise = (torch.rand(num_envs, device=self.device) * 2.0 - 1.0) * self.cfg.rot_std
        q_noise = math_utils.quat_from_euler_xyz(
            torch.zeros(num_envs, device=self.device),
            torch.zeros(num_envs, device=self.device),
            yaw_noise
        )
        
        # In Isaac Lab, quat is (w, x, y, z)
        new_quat = math_utils.quat_mul(q_noise, quat)
        
        return pos, new_quat

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        """Reset heuristic state along with the environment."""
        super()._reset_idx(env_ids)
        self._heuristic.reset_idx(env_ids)
        self._current_stiffness[env_ids] = torch.tensor(self.cfg.osc_stiffness, device=self.device)
        
        # Update inputs for the newly reset envs
        self._update_heuristic_inputs(env_ids)
        self._set_heuristic_targets(env_ids)

    def _update_target(self, action: torch.Tensor) -> None:
        """Modified update that combines heuristic base with policy residuals."""
        # Use stored heuristic inputs
        tip_pos = self._h_tip_pos
        tip_force_z = self._h_tip_force_z

        # 2. Query heuristic for base command
        heuristic_out = self._heuristic.compute(tip_pos, tip_force_z)
        base_pos = heuristic_out["target_pos"]
        base_quat = heuristic_out["target_quat"]
        base_stiffness = heuristic_out["stiffness"]

        # 3. Apply policy residuals
        res_pos_delta = action[:, 0:3] * self.cfg.residual_pos_scale
        res_ori_delta = action[:, 3:6] * self.cfg.residual_ori_scale
        res_stiff_delta = action[:, 6:12] * self.cfg.residual_stiffness_scale

        res_pos_w = math_utils.quat_apply(base_quat, res_pos_delta)
        self._target_pos_w = base_pos + res_pos_w

        dq = math_utils.quat_from_euler_xyz(res_ori_delta[:, 0], res_ori_delta[:, 1], res_ori_delta[:, 2])
        self._target_quat_w = math_utils.quat_mul(dq, base_quat)
        self._target_quat_w = self._target_quat_w / self._target_quat_w.norm(dim=-1, keepdim=True)

        self._current_stiffness = torch.clamp(base_stiffness + res_stiff_delta, min=1.0, max=2000.0)

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, extras = super().step(action)
        
        # Update heuristic inputs for the next step
        self._update_heuristic_inputs()
        
        extras["heuristic_state"] = self._heuristic.states
        return obs, reward, terminated, truncated, extras
