from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.envs.mdp import JointEffortActionCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
)

from ..aic_task_base_env import AICTaskBaseEnv


@configclass
class RelCartesianOSPNoRefEnvCfg(AICTaskBaseEnv):
    """Configuration for Relative Cartesian Operational Space Controller environment."""

    # OSC parameters
    osc_ee_body: str = "gripper_tcp"
    osc_joint_names: list[str] = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    osc_stiffness: tuple[float, ...] = (300.0, 300.0, 300.0, 20.0, 20.0, 20.0)
    osc_damping: tuple[float, ...] = (35.0, 35.0, 35.0, 9.0, 9.0, 9.0)
    osc_inertial_dynamics_decoupling: bool = False
    osc_gravity_compensation: bool = True
    osc_effort_limit: float = 187.0
    
    # Action scaling
    action_delta_pos_scale: float = 0.0005   # 0.5 mm/step
    action_delta_ori_scale: float = 0.009    # ~0.5°/step

    def __post_init__(self) -> None:
        super().__post_init__()

        # Dummy joint action that does nothing
        self.actions.arm_action = JointEffortActionCfg(
            asset_name="robot",
            joint_names=self.osc_joint_names,
            scale=0.0, # This makes the action manager output 0 effort
        )

        # Replace implicit actuators with explicit torque actuators
        self.scene.robot.actuators["arm"] = IdealPDActuatorCfg(
            joint_names_expr=self.osc_joint_names,
            stiffness=0.0,
            damping=0.0,
            effort_limit=self.osc_effort_limit,
            effort_limit_sim=self.osc_effort_limit,
            velocity_limit_sim=100.0,
        )


class RelCartesianOSPEnv(ManagerBasedRLEnv):
    """RL env that applies Operational Space Control (OSC) manually in the step loop."""

    cfg: RelCartesianOSPNoRefEnvCfg

    def __init__(self, cfg: RelCartesianOSPNoRefEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._osc_setup()

    def _osc_setup(self) -> None:
        """Initialize OSC controller and cache robot indices."""
        self._robot = self.scene["robot"]

        # Resolve EE body index
        body_name = self.cfg.osc_ee_body
        try:
            self._ee_body_idx = self._robot.body_names.index(body_name)
        except ValueError:
            self._ee_body_idx = next(
                i for i, n in enumerate(self._robot.body_names) if body_name in n
            )

        # Resolve arm joint indices
        self._arm_joint_ids = torch.tensor(
            [self._robot.joint_names.index(n) for n in self.cfg.osc_joint_names],
            device=self.device,
            dtype=torch.long,
        )

        # PhysX Jacobian skips the base link for fixed-base robots.
        self._jacobi_body_idx = self._ee_body_idx - 1

        # Build OSC Controller
        Kp = list(self.cfg.osc_stiffness)
        Kd = list(self.cfg.osc_damping)
        ratios = [
            kd / (2.0 * math.sqrt(kp)) if kp > 0 else 1.0
            for kp, kd in zip(Kp, Kd)
        ]
        
        osc_cfg = OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            motion_control_axes_task=[1, 1, 1, 1, 1, 1],
            contact_wrench_control_axes_task=[0, 0, 0, 0, 0, 0],
            inertial_dynamics_decoupling=self.cfg.osc_inertial_dynamics_decoupling,
            gravity_compensation=self.cfg.osc_gravity_compensation,
            impedance_mode="fixed",
            motion_stiffness_task=Kp,
            motion_damping_ratio_task=ratios,
            nullspace_control="none",
        )
        self._osc = OperationalSpaceController(osc_cfg, self.num_envs, self.device)

        self._identity_pose = torch.zeros(self.num_envs, 7, device=self.device)
        self._identity_pose[:, 3] = 1.0

        self._target_pos_w: torch.Tensor | None = None
        self._target_quat_w: torch.Tensor | None = None

    def step(self, action: torch.Tensor):
        """Modified step that accumulates delta actions and applies OSC torques."""
        action = action.to(self.device)

        # Clip action to be within [-1, 1] for stability
        if torch.isnan(action).any():
            print("[CRITICAL] NaN Action detected!")
            action = torch.nan_to_num(action, nan=0.0)
        action = torch.clamp(action, -1.0, 1.0)

        # Initialize targets if they don't exist
        if self._target_pos_w is None:
            self._seed_target_from_ee()

        # 1. Update world-frame target based on delta action
        self._update_target(action)

        # 2. Standard manager-based steps (processing observations, etc.)
        self.action_manager.process_action(action)
        
        # We manually step the simulation to apply OSC at each physics step
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            
            # This writes the action manager's output (which is 0 effort due to our dummy cfg)
            self.action_manager.apply_action()
            
            # 3. Overwrite efforts with OSC output
            self._apply_osc()

            # Check for physics explosions
            joint_vel = self._robot.data.joint_vel
            if not torch.isfinite(joint_vel).all():
                bad_envs = ~torch.isfinite(joint_vel).all(dim=-1)
                reset_ids = bad_envs.nonzero(as_tuple=False).squeeze(-1)
                print(f"[CRITICAL] Physics exploded in {len(reset_ids)} envs. Resetting...")
                self._reset_idx(reset_ids)
                # Skip the current sim step update for these envs to avoid NaN propagation
                continue
            
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        # 4. Post-physics manager updates
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # Handle resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # Safety check for policy inputs/outputs
        # obs_buf is a dict of tensors, reward_buf is a tensor
        obs_has_nan = any(not torch.isfinite(v).all() for v in self.obs_buf.values())
        rew_has_nan = not torch.isfinite(self.reward_buf).all()
        if obs_has_nan or rew_has_nan:
            print(f"[CRITICAL] Non-finite values! Obs: {obs_has_nan}, Rew: {rew_has_nan}. Sanitizing...")
            for k in self.obs_buf:
                # Replace NaN and Inf with 0.0
                self.obs_buf[k] = torch.nan_to_num(self.obs_buf[k], nan=0.0, posinf=0.0, neginf=0.0)
            self.reward_buf = torch.nan_to_num(self.reward_buf, nan=0.0, posinf=0.0, neginf=0.0)

        return (
            self.obs_buf,
            self.reward_buf,
            self.termination_manager.terminated,
            self.termination_manager.time_outs,
            self.extras,
        )

    def _apply_osc(self) -> None:
        """Compute and apply OSC torques to the robot."""
        robot = self._robot

        # Get Jacobians
        jacs = robot.root_physx_view.get_jacobians()
        J_full = jacs[:, self._jacobi_body_idx, :, :].to(self.device)
        J_arm = J_full[:, :, self._arm_joint_ids]

        # Get current state
        ee_pos = robot.data.body_pos_w[:, self._ee_body_idx, :]
        ee_quat = robot.data.body_quat_w[:, self._ee_body_idx, :]
        ee_pose = torch.cat([ee_pos, ee_quat], dim=-1)
        ee_vel = robot.data.body_vel_w[:, self._ee_body_idx, :]

        # Optional dynamics terms
        mass_matrix = None
        if self.cfg.osc_inertial_dynamics_decoupling:
            M_full = robot.root_physx_view.get_generalized_mass_matrices()
            ids = self._arm_joint_ids.cpu()
            mass_matrix = M_full[:, ids][:, :, ids].to(self.device)

        gravity = None
        if self.cfg.osc_gravity_compensation:
            g_full = robot.root_physx_view.get_gravity_compensation_forces()
            ids = self._arm_joint_ids.cpu()
            gravity = g_full[:, ids].to(self.device)

        # Set OSC command and compute torque
        target_pose = torch.cat([self._target_pos_w, self._target_quat_w], dim=-1)
        # Ensure target pose is clean (no NaN/Inf)
        target_pose = torch.nan_to_num(target_pose, nan=0.0, posinf=0.0, neginf=0.0)
        
        self._osc.set_command(
            command=target_pose,
            current_ee_pose_b=ee_pose,
            current_task_frame_pose_b=self._identity_pose,
        )
        tau_arm = self._osc.compute(
            jacobian_b=J_arm,
            current_ee_pose_b=ee_pose,
            current_ee_vel_b=ee_vel,
            mass_matrix=mass_matrix,
            gravity=gravity,
        )
        
        # Clamp and apply (and sanitize NaNs/Infs)
        tau_arm = torch.nan_to_num(tau_arm, nan=0.0, posinf=0.0, neginf=0.0)
        tau_arm = tau_arm.clamp(-self.cfg.osc_effort_limit, self.cfg.osc_effort_limit)
        
        num_joints = robot.num_joints
        if num_joints == len(self.cfg.osc_joint_names):
            robot.set_joint_effort_target(tau_arm)
        else:
            efforts = torch.zeros(self.num_envs, num_joints, device=self.device)
            efforts[:, self._arm_joint_ids] = tau_arm
            robot.set_joint_effort_target(efforts)

    def _seed_target_from_ee(self) -> None:
        """Synchronize the target pose with the current EE pose."""
        self._target_pos_w = self._robot.data.body_pos_w[:, self._ee_body_idx, :].clone()
        self._target_quat_w = self._robot.data.body_quat_w[:, self._ee_body_idx, :].clone()

    def _update_target(self, action: torch.Tensor) -> None:
        """Accumulate delta actions into the world-frame target pose."""
        pos_scale = self.cfg.action_delta_pos_scale
        ori_scale = self.cfg.action_delta_ori_scale

        # delta_pos is expressed in EE local frame; rotate to world.
        delta_pos_ee = action[:, :3] * pos_scale
        ee_quat = self._robot.data.body_quat_w[:, self._ee_body_idx, :]
        delta_pos = quat_apply(ee_quat, delta_pos_ee)
        delta_rpy = action[:, 3:6] * ori_scale

        self._target_pos_w = self._target_pos_w + delta_pos

        dq = quat_from_euler_xyz(delta_rpy[:, 0], delta_rpy[:, 1], delta_rpy[:, 2])
        self._target_quat_w = quat_mul(dq, self._target_quat_w)
        self._target_quat_w = self._target_quat_w / self._target_quat_w.norm(dim=-1, keepdim=True)

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        """Reset environment indices and re-seed the OSC target."""
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return

        if self._target_pos_w is None:
            self._seed_target_from_ee()

        # Seed OSC target from actual EE pose (after reset perturbation).
        self._target_pos_w[env_ids] = self._robot.data.body_pos_w[env_ids, self._ee_body_idx, :].clone()
        self._target_quat_w[env_ids] = self._robot.data.body_quat_w[env_ids, self._ee_body_idx, :].clone()