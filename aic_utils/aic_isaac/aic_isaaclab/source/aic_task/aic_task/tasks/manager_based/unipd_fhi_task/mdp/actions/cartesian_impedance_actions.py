from __future__ import annotations
from dataclasses import MISSING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class CartesianImpedanceTargetAction(ActionTerm):
    cfg: CartesianImpedanceTargetActionCfg

    def __init__(self, cfg: CartesianImpedanceTargetActionCfg, env):
        super().__init__(cfg, env)

        self.cfg = cfg
        self.robot = self._asset

        self._joint_ids, _ = self.robot.find_joints(cfg.joint_names)
        body_ids, _ = self.robot.find_bodies(cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected exactly one body for {cfg.body_name}, got {body_ids}")
        self._body_id = body_ids[0]

        if self.robot.is_fixed_base:
            self._jacobi_body_idx = self._body_id - 1
        else:
            self._jacobi_body_idx = self._body_id

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        self._target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_quat = torch.zeros(self.num_envs, 4, device=self.device)

        self._seg_start_pos = torch.zeros_like(self._target_pos)
        self._seg_start_quat = torch.zeros_like(self._target_quat)
        self._seg_goal_pos = torch.zeros_like(self._target_pos)
        self._seg_goal_quat = torch.zeros_like(self._target_quat)

        self._seg_tick = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._seg_total = torch.full(
            (self.num_envs,), int(self._env.cfg.decimation), dtype=torch.long, device=self.device
        )

        self._xdot_des = torch.zeros(self.num_envs, 6, device=self.device)

        self._Kp = self._build_cartesian_gain_matrix(
            cfg.translational_stiffness, cfg.rotational_stiffness
        )
        self._Kd = self._build_cartesian_gain_matrix(
            cfg.translational_damping, cfg.rotational_damping
        )

        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        ee_pos_b, ee_quat_b = self._get_ee_pose_base()

        self._target_pos[env_ids] = ee_pos_b[env_ids]
        self._target_quat[env_ids] = ee_quat_b[env_ids]

        self._seg_start_pos[env_ids] = ee_pos_b[env_ids]
        self._seg_start_quat[env_ids] = ee_quat_b[env_ids]
        self._seg_goal_pos[env_ids] = ee_pos_b[env_ids]
        self._seg_goal_quat[env_ids] = ee_quat_b[env_ids]

        self._seg_tick[env_ids] = 0
        self._xdot_des[env_ids] = 0.0
        self._initialized[env_ids] = True

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions

        # lazy init so targets start from the current pose
        if not bool(torch.all(self._initialized)):
            not_init = torch.nonzero(~self._initialized, as_tuple=False).squeeze(-1)
            if not_init.numel() > 0:
                self.reset(not_init)

        ee_pos_b, ee_quat_b = self._get_ee_pose_base()

        dpos = actions[:, 0:3] * self.cfg.position_scale
        # drot = actions[:, 3:6] * self.cfg.orientation_scale

        # dquat = self._axis_angle_to_quat(drot)

        dquat = actions[:, 3:7]

        if self.cfg.use_relative_mode:
            if self.cfg.command_frame == "base_link":
                goal_pos = ee_pos_b + dpos
                goal_quat = math_utils.quat_mul(dquat, ee_quat_b)
            elif self.cfg.command_frame == "gripper/tcp":
                goal_pos = ee_pos_b + math_utils.quat_apply(ee_quat_b, dpos)
                goal_quat = math_utils.quat_mul(ee_quat_b, dquat)
            else:
                raise ValueError(f"Unsupported command_frame: {self.cfg.command_frame}")
        else:
            goal_pos = dpos
            goal_quat = dquat

        if self.cfg.clamp_workspace:
            goal_pos = self._clamp_workspace(goal_pos)
            
        goal_quat = math_utils.normalize(goal_quat)

        # interpolate from current active target, not from current measured pose
        self._seg_start_pos[:] = self._target_pos
        self._seg_start_quat[:] = self._target_quat
        self._seg_goal_pos[:] = goal_pos
        self._seg_goal_quat[:] = goal_quat
        self._seg_tick[:] = 0

        if self.cfg.use_interpolated_twist:
            self._xdot_des[:] = self._compute_segment_twist(
                self._seg_start_pos,
                self._seg_start_quat,
                self._seg_goal_pos,
                self._seg_goal_quat,
            )
        else:
            self._xdot_des.zero_()

        print(f"goal_pos: {goal_pos}")
        print(f"goal_quat: {goal_quat}")
        print(f"raw_actions: {actions}")

    def apply_actions(self) -> None:
        if not bool(torch.all(self._initialized)):
            not_init = torch.nonzero(~self._initialized, as_tuple=False).squeeze(-1)
            if not_init.numel() > 0:
                self.reset(not_init)

        alpha = ((self._seg_tick.float() + 1.0) / self._seg_total.float()).clamp(0.0, 1.0)
        alpha_unsq = alpha.unsqueeze(-1)

        self._target_pos[:] = (1.0 - alpha_unsq) * self._seg_start_pos + alpha_unsq * self._seg_goal_pos
        self._target_quat[:] = self._quat_slerp_batch(
            self._seg_start_quat, self._seg_goal_quat, alpha
        )
        self._target_quat[:] = math_utils.normalize(self._target_quat)

        ee_pos_b, ee_quat_b = self._get_ee_pose_base()
        joint_vel = self.robot.data.joint_vel[:, self._joint_ids]

        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._joint_ids
        ]
        ee_twist = torch.bmm(jacobian, joint_vel.unsqueeze(-1)).squeeze(-1)

        pos_error = self._target_pos - ee_pos_b

        quat_error = math_utils.quat_mul(self._target_quat, math_utils.quat_inv(ee_quat_b))
        quat_error = self._standardize_quat(quat_error)
        rot_error = math_utils.axis_angle_from_quat(quat_error)

        pose_error = torch.cat([pos_error, rot_error], dim=-1)
        vel_error = self._xdot_des - ee_twist

        task_wrench = (
            torch.bmm(self._Kp, pose_error.unsqueeze(-1)).squeeze(-1)
            + torch.bmm(self._Kd, vel_error.unsqueeze(-1)).squeeze(-1)
        )

        joint_torques = torch.bmm(
            jacobian.transpose(-1, -2), task_wrench.unsqueeze(-1)
        ).squeeze(-1)

        print(f"joint_torques {self._seg_tick}: {joint_torques}")

        if self.cfg.torque_limit is not None:
            joint_torques = torch.clamp(
                joint_torques, -self.cfg.torque_limit, self.cfg.torque_limit
            )

        self.robot.set_joint_effort_target(joint_torques, joint_ids=self._joint_ids)

        self._seg_tick[:] = torch.minimum(self._seg_tick + 1, self._seg_total)

    def _get_ee_pose_base(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pose_w = self.robot.data.body_pose_w[:, self._body_id]
        root_pose_w = self.robot.data.root_pose_w

        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )
        ee_quat_b = math_utils.normalize(ee_quat_b)
        return ee_pos_b, ee_quat_b

    def _clamp_workspace(self, pos_b: torch.Tensor) -> torch.Tensor:
        lower = torch.tensor(self.cfg.pos_bounds[0], device=self.device, dtype=pos_b.dtype)
        upper = torch.tensor(self.cfg.pos_bounds[1], device=self.device, dtype=pos_b.dtype)
        return torch.max(torch.min(pos_b, upper), lower)

    def _build_cartesian_gain_matrix(
        self,
        translational: tuple[float, float, float],
        rotational: tuple[float, float, float],
    ) -> torch.Tensor:
        diag = torch.tensor(
            [*translational, *rotational],
            dtype=torch.float32,
            device=self.device,
        )
        return torch.diag_embed(diag.unsqueeze(0).repeat(self.num_envs, 1))

    def _axis_angle_to_quat(self, axis_angle: torch.Tensor) -> torch.Tensor:
        angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
        safe_axis = axis_angle / angle.clamp(min=1e-8)
        quat = math_utils.quat_from_angle_axis(angle.squeeze(-1), safe_axis)
        small = angle.squeeze(-1) < 1e-8
        if torch.any(small):
            quat[small] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=quat.dtype)
        return quat

    def _standardize_quat(self, quat: torch.Tensor) -> torch.Tensor:
        quat = quat.clone()
        neg = quat[:, 0] < 0.0
        quat[neg] = -quat[neg]
        return quat

    def _compute_segment_twist(
        self,
        start_pos: torch.Tensor,
        start_quat: torch.Tensor,
        goal_pos: torch.Tensor,
        goal_quat: torch.Tensor,
    ) -> torch.Tensor:
        dt = self._env.physics_dt * float(self._env.cfg.decimation)

        lin_vel = (goal_pos - start_pos) / dt

        q_err = math_utils.quat_mul(goal_quat, math_utils.quat_inv(start_quat))
        q_err = self._standardize_quat(q_err)
        rot_vec = math_utils.axis_angle_from_quat(q_err)
        ang_vel = rot_vec / dt

        return torch.cat([lin_vel, ang_vel], dim=-1)
    
    def _quat_slerp_batch(
        self,
        q0: torch.Tensor,   # (N, 4)
        q1: torch.Tensor,   # (N, 4)
        t: torch.Tensor,    # (N,)
    ) -> torch.Tensor:
        q0 = math_utils.normalize(q0)
        q1 = math_utils.normalize(q1)

        # shortest path
        dot = torch.sum(q0 * q1, dim=-1, keepdim=True)  # (N, 1)
        q1 = torch.where(dot < 0.0, -q1, q1)
        dot = torch.sum(q0 * q1, dim=-1, keepdim=True).clamp(-1.0, 1.0)

        t = t.unsqueeze(-1)  # (N, 1)

        # if very close, fall back to normalized lerp
        close = dot > 0.9995

        theta_0 = torch.acos(dot)                      # angle between q0 and q1
        sin_theta_0 = torch.sin(theta_0).clamp_min(1e-8)

        theta = theta_0 * t
        sin_theta = torch.sin(theta)

        s0 = torch.sin(theta_0 - theta) / sin_theta_0
        s1 = sin_theta / sin_theta_0

        slerp = s0 * q0 + s1 * q1
        lerp = math_utils.normalize((1.0 - t) * q0 + t * q1)

        out = torch.where(close, lerp, slerp)
        return math_utils.normalize(out)
    


@configclass
class CartesianImpedanceTargetActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = CartesianImpedanceTargetAction # Added at after class definition

    asset_name: str = MISSING
    joint_names: list[str] = MISSING
    body_name: str = MISSING

    # command semantics
    command_frame: str = "base_link"   # "base_link" or "gripper/tcp"
    use_relative_mode: bool = True

    # action scaling
    position_scale: float = 0.005      # m per env step
    orientation_scale: float = 0.05    # rad per env step

    # workspace clamp in base frame
    clamp_workspace: bool = False
    pos_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (-0.70, -0.50, 0.02),
        (-0.20,  0.50, 0.50),
    )

    # fixed Cartesian impedance gains
    translational_stiffness: tuple[float, float, float] = (85.0, 85.0, 85.0)
    rotational_stiffness: tuple[float, float, float] = (85.0, 85.0, 85.0)
    translational_damping: tuple[float, float, float] = (75.0, 75.0, 75.0)
    rotational_damping: tuple[float, float, float] = (75.0, 75.0, 75.0)

    # torque safety
    torque_limit: float | None = None

    # whether the desired task-space twist should come from interpolation
    use_interpolated_twist: bool = True