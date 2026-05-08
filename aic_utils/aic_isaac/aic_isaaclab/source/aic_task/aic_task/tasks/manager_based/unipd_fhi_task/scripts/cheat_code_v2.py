# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified Cheat code v2 policy and Task configuration for insertion task in Isaac Lab."""

import argparse
from isaaclab.app import AppLauncher

# 1. Setup Argparse
parser = argparse.ArgumentParser(description="Cheat code v2 for insertion task.")
parser.add_argument("--task", type=str, default="CheatCode-v2-Task", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable fabric.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. Launch App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 3. Isaac Lab / Task Imports
import math
import torch
import numpy as np
import gymnasium as gym
from enum import Enum, auto

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from aic_task.tasks.manager_based.unipd_fhi_task import mdp

# Import the base OSP task and env
from aic_task.tasks.manager_based.unipd_fhi_task.config.rel_cart_no_ref import RelCartesianOSPNoRefEnvCfg, RelCartesianOSPNoRefEnv as RelCartesianOSPEnv

##
# Task Configuration
##

def reset(env, env_ids):
    device = env.device
    env_origins = env.scene.env_origins[env_ids]
    n = len(env_ids)

    # Reset robot
    robot = env.scene["robot"]
    robot_pos = torch.tensor((-0.2, 0.2, 1.14), device=device, dtype=torch.float32).unsqueeze(0).expand(n, -1)
    robot_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float32).unsqueeze(0).expand(n, -1)
    robot.write_root_pose_to_sim(torch.cat([robot_pos + env_origins, robot_rot], dim=-1), env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=device, dtype=torch.float32), env_ids=env_ids)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # Reset NIC card
    nic_card = env.scene["nic_card"]
    nic_card_pos = torch.tensor((0.11765, -0.17671, 1.2143), device=device, dtype=torch.float32).unsqueeze(0).expand(n, -1)
    nic_card_rot = torch.tensor([0.0, 0.0, -0.7068252, 0.7073883], device=device, dtype=torch.float32).unsqueeze(0).expand(n, -1)
    nic_card.write_root_pose_to_sim(torch.cat([nic_card_pos + env_origins, nic_card_rot], dim=-1), env_ids=env_ids)
    nic_card.write_root_velocity_to_sim(torch.zeros(n, 6, device=device, dtype=torch.float32), env_ids=env_ids)

@configclass
class CheatCodeV2TaskCfg(RelCartesianOSPNoRefEnvCfg):
    """Task configuration specifically for cheatcode v2 validation."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Modify OSC reference link to gripper_tcp
        self.osc_ee_body = "gripper_tcp"

        # Disable episode timeout
        self.terminations.time_out = None
        self.terminations.failed_insertion = None

        # Enable variable stiffness mode in the base env
        self.osc_impedance_mode = "variable_kp"

        # Delete the task board
        self.scene.task_board = None

        self.events.reset_scene = EventTerm(
            func=reset,
            mode="reset",
        )

##
# Registration
##

gym.register(
    id="CheatCode-v2-Task",
    entry_point=RelCartesianOSPEnv,
    disable_env_checker=True,
    kwargs={
        "cfg": CheatCodeV2TaskCfg(),
    },
)

##
# Policy Implementation
##

class InsertCableState(Enum):
    MOVE_ABOVE_PORT = auto()
    DESCEND_AND_INSERT = auto()
    STABILIZE_XY = auto()
    UNTILT_INSERTED_CABLE = auto()
    REDESCEND = auto()
    FIX_YAW = auto()
    STABILIZE = auto()
    DONE = auto()
    FAILED = auto()

class CheatCode:

    def __init__(self, env: RelCartesianOSPEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device

        # Sensor/Robot references
        self.tip_sensor = env.scene.sensors["sfp_tip_sensor"]
        self.port_sensor = env.scene.sensors["sfp_port_sensor"]
        self._robot = env.scene["robot"]
        self.command_term = env.command_manager.get_term("sfp_port_pose_command")
        
        # Zero action for the step method
        self.zero_action = torch.zeros(self.num_envs, 6, device=self.device)

        # State machine variables
        self._tip_x_error_integrator = torch.zeros(self.num_envs, device=self.device)
        self._tip_y_error_integrator = torch.zeros(self.num_envs, device=self.device)
        self._max_integrator_windup = 0.05

        # Fix yaw parameters
        self.fix_yaw_direction = 1.0
        self.fix_yaw_attempt = 0
        self.fix_yaw_max_attempts = 2
        self.fix_yaw_start_z = None
        self.fix_yaw_best_z = None
        self.fix_yaw_no_improve_steps = 0
        self.fix_yaw_max_no_improve_steps = 100
        self.fix_yaw_step = np.deg2rad(0.2)
        self.fix_yaw_limit = np.deg2rad(10.0)
        self.fix_yaw_min_z_improvement = 0.001

        # T_gripper_tip calculation
        self._pos_tg, self._quat_tg = self._get_T_tip_gripper()

    def add_noise_to_port(self, pos, quat, pos_std=0.003, rot_std=np.deg2rad(7)):
        """Adds Gaussian/Uniform noise to port pose as in emp_cheat.py."""
        num_envs = pos.shape[0]
        
        # Translation noise: positive uniform in X and Y
        noise_x = torch.rand(num_envs, device=self.device) * pos_std
        noise_y = torch.rand(num_envs, device=self.device) * pos_std
        
        pos[:, 0] += noise_x
        pos[:, 1] += noise_y
        
        # Rotation noise: uniform yaw noise
        yaw_noise = (torch.rand(num_envs, device=self.device) * 2.0 - 1.0) * rot_std
        q_noise = math_utils.quat_from_euler_xyz(
            torch.zeros(num_envs, device=self.device),
            torch.zeros(num_envs, device=self.device),
            yaw_noise
        )
        
        # In Isaac Lab, quat is (w, x, y, z)
        new_quat = math_utils.quat_mul(q_noise, quat)
        
        return pos, new_quat

    def _get_T_tip_gripper(self):
        """Calculates the relative transform from tip to gripper."""
        gripper_pos = self._robot.data.body_pos_w[:, self.env._ee_body_idx, :]
        gripper_quat = self._robot.data.body_quat_w[:, self.env._ee_body_idx, :]
        
        tip_pos = self.tip_sensor.data.target_pos_w[:, 0]
        tip_quat = self.tip_sensor.data.target_quat_w[:, 0]

        pos_rel, quat_rel = math_utils.subtract_frame_transforms(tip_pos, tip_quat, gripper_pos, gripper_quat)
        return pos_rel.to(dtype=torch.float32), quat_rel.to(dtype=torch.float32)

    def run(self):
        """Executes the multi-phase insertion sequence."""
        self.env.reset()
        env_ids = torch.arange(self.num_envs, device=self.device)
        
        print("--- [Cheat Code v2] Starting execution loop ---")
        
        pos_tg, quat_tg = self._pos_tg, self._quat_tg
        state = InsertCableState.MOVE_ABOVE_PORT
        
        z_offset = 0.10
        port_entrance_offset = 0.045
        z_force_threshold = -1.0  # [N]
        
        roll_angle = np.deg2rad(15)
        pitch_angle = np.deg2rad(15)
        yaw_angle = 0.0
        untilt_step = np.deg2rad(0.5)

        stabilize_step = 0
        target_stabilize_steps = 75

        target_idx = self.command_term.targets_idx
        port_pos = self.port_sensor.data.target_pos_w[env_ids, target_idx].to(dtype=torch.float32)
        port_quat = self.port_sensor.data.target_quat_w[env_ids, target_idx].to(dtype=torch.float32)

        # Add x,y noise to the port position for testing
        port_pos, port_quat = self.add_noise_to_port(port_pos, port_quat)

        obs = None
        while state not in [InsertCableState.DONE, InsertCableState.FAILED] and simulation_app.is_running():
            tip_pos = self.tip_sensor.data.target_pos_w[:, 0].to(dtype=torch.float32)
            tip_quat = self.tip_sensor.data.target_quat_w[:, 0].to(dtype=torch.float32)
            
            # Get force from observation
            if obs is not None and "policy" in obs:
                # In the new obs structure, body_forces might be at a different index.
                # In rel_cart_no_ref.py:
                # port_target (3) + ee_pos (3) + ee_rpy (3) + ee_lin_vel (3) + ee_ang_vel (3) + body_forces (6) + actions (6)
                # body_forces start at index 15.
                body_forces = obs["policy"][:, 15:21]
                tip_force_z = body_forces[:, 2]
            else:
                tip_force_z = torch.zeros(self.num_envs, device=self.device)

            # ---------------------------------------------------------
            # State Machine
            # ---------------------------------------------------------
            reset_xy_integrator = True

            if state == InsertCableState.MOVE_ABOVE_PORT:
                # 1. Orientation with initial tilt
                tilt_quat = math_utils.quat_from_euler_xyz(
                    torch.tensor(roll_angle, device=self.device, dtype=torch.float32),
                    torch.tensor(pitch_angle, device=self.device, dtype=torch.float32),
                    torch.tensor(yaw_angle, device=self.device, dtype=torch.float32)
                ).unsqueeze(0).expand(self.num_envs, -1)
                
                target_tip_quat = math_utils.quat_mul(port_quat, tilt_quat)
                target_tip_pos = port_pos.clone()
                target_tip_pos[:, 2] += z_offset

                # 2. Transform to gripper
                target_gripper_pos = target_tip_pos + math_utils.quat_apply(target_tip_quat, pos_tg)
                target_gripper_quat = math_utils.quat_mul(target_tip_quat, quat_tg)

                # 3. Set target and stiffness
                self.env._target_pos_w = target_gripper_pos
                self.env._target_quat_w = target_gripper_quat
                self.env._current_stiffness = torch.tensor([1000.0, 1000.0, 1000.0, 150.0, 150.0, 150.0], device=self.device).repeat(self.num_envs, 1)
                
                print(f"Moving to approach position above port...")
                for _ in range(100):
                    obs, _, _, _, _ = self.env.step(self.zero_action)
                
                state = InsertCableState.DESCEND_AND_INSERT
                print(f"State: {state.name}")
                continue

            elif state == InsertCableState.DESCEND_AND_INSERT:
                self.env._current_stiffness = torch.tensor([90.0, 90.0, 400.0, 50, 50, 50], device=self.device).repeat(self.num_envs, 1)
                contact_detected = tip_force_z > z_force_threshold
                
                if contact_detected:
                    print(f"Z contact detected: Fz={tip_force_z[0]:.3f} N, z_offset={z_offset:.3f}")
                    state = InsertCableState.STABILIZE_XY
                    print(f"State: {state.name}")
                    continue

                if z_offset < 0.0:
                    print("Reached max insertion depth without detecting Z contact.")
                    state = InsertCableState.STABILIZE
                    print(f"State: {state.name}")
                    continue

                z_offset -= 0.001
                reset_xy_integrator = True

            elif state == InsertCableState.STABILIZE_XY:
                self.env._current_stiffness = torch.tensor([10.0, 10.0, 90.0, 5.0, 5.0, 5.0], device=self.device).repeat(self.num_envs, 1)

                if z_offset < port_entrance_offset - 0.03:
                    print(f"Insertion depth reached: z_offset={z_offset:.3f}, port_entrance_z={port_pos[0, 2] + port_entrance_offset:.3f}, tip_pos_z={tip_pos[0, 2]:.3f}. Checking alignment...")
                    port_pos[:, 0] = tip_pos[:, 0]
                    port_pos[:, 1] = tip_pos[:, 1]
                    
                    state = InsertCableState.UNTILT_INSERTED_CABLE
                    
                    print(f"State: {state.name}")
                    continue

                z_offset -= 0.001
                reset_xy_integrator = False

            elif state == InsertCableState.FIX_YAW:
                self.env._current_stiffness = torch.tensor([10, 10, 90, 25, 25, 200], device=self.device).repeat(self.num_envs, 1)

                current_z = tip_pos[0, 2].item()
                z_distance = current_z - (port_pos[0, 2] + port_entrance_offset)
                plug_inserted = z_distance < 0
                print(f"Z_distance: {z_distance}, Plug inserted: {plug_inserted}")

                if self.fix_yaw_start_z is None:
                    self.fix_yaw_start_z = current_z
                    self.fix_yaw_best_z = current_z
                    self.fix_yaw_no_improve_steps = 0

                if plug_inserted:
                    print("Yaw fixed.")
                    self.fix_yaw_start_z = None
                    state = InsertCableState.UNTILT_INSERTED_CABLE
                    continue

                if current_z < self.fix_yaw_best_z - self.fix_yaw_min_z_improvement:
                    self.fix_yaw_best_z = current_z
                    self.fix_yaw_no_improve_steps = 0
                else:
                    self.fix_yaw_no_improve_steps += 1

                yaw_angle += self.fix_yaw_direction * self.fix_yaw_step

                yaw_limit_reached = abs(yaw_angle) >= self.fix_yaw_limit
                no_improvement = self.fix_yaw_no_improve_steps >= self.fix_yaw_max_no_improve_steps

                if yaw_limit_reached or no_improvement:
                    self.fix_yaw_attempt += 1
                    if self.fix_yaw_attempt < self.fix_yaw_max_attempts:
                        self.fix_yaw_direction *= -1.0
                        self.fix_yaw_start_z = None
                        self.fix_yaw_best_z = None
                        self.fix_yaw_no_improve_steps = 0
                        continue
                    
                    print("Yaw adjustment failed in both directions.")
                    state = InsertCableState.UNTILT_INSERTED_CABLE
                    print(f"State: {state.name}")
                    continue

                reset_xy_integrator = True

            elif state == InsertCableState.UNTILT_INSERTED_CABLE:
                self.env._current_stiffness = torch.tensor([20.0, 20.0, 50.0, 50.0, 50.0, 200.0], device=self.device).repeat(self.num_envs, 1)
                roll_done = abs(roll_angle) <= untilt_step
                pitch_done = abs(pitch_angle) <= untilt_step
                if not roll_done:
                    roll_angle -= np.sign(roll_angle) * untilt_step
                if not pitch_done:
                    pitch_angle -= np.sign(pitch_angle) * untilt_step

                if roll_done and pitch_done and z_offset >= (port_entrance_offset - 0.005):
                    print(f"Untilt completed. z_offset: {z_offset}")
                    state = InsertCableState.REDESCEND
                    print(f"State: {state.name}")
                    continue

                z_offset += 0.0005
                reset_xy_integrator = False

            elif state == InsertCableState.REDESCEND:
                self.env._current_stiffness = torch.tensor([90.0, 90.0, 90.0, 200.0, 200.0, 200.0], device=self.device).repeat(self.num_envs, 1)
                z_offset -= 0.001
                reset_xy_integrator = False
                if z_offset < -0.015:
                    print("Redescend completed.")
                    state = InsertCableState.STABILIZE
                    print(f"State: {state.name}")
                    continue

            elif state == InsertCableState.STABILIZE:
                stabilize_step += 1
                if stabilize_step > target_stabilize_steps:
                    state = InsertCableState.DONE
                    print(f"State: {state.name}")

            # ---------------------------------------------------------
            # Compute Target Pose for TIP
            # ---------------------------------------------------------
            tilt_quat = math_utils.quat_from_euler_xyz(
                torch.tensor(roll_angle, device=self.device, dtype=torch.float32),
                torch.tensor(pitch_angle, device=self.device, dtype=torch.float32),
                torch.tensor(yaw_angle, device=self.device, dtype=torch.float32)
            ).unsqueeze(0).expand(self.num_envs, -1)
            
            target_tip_quat = math_utils.quat_mul(port_quat, tilt_quat)

            if reset_xy_integrator:
                self._tip_x_error_integrator.zero_()
                self._tip_y_error_integrator.zero_()
            else:
                tip_x_error = port_pos[:, 0] - tip_pos[:, 0]
                tip_y_error = port_pos[:, 1] - tip_pos[:, 1]
                self._tip_x_error_integrator = torch.clamp(
                    self._tip_x_error_integrator + tip_x_error,
                    -self._max_integrator_windup,
                    self._max_integrator_windup
                )
                self._tip_y_error_integrator = torch.clamp(
                    self._tip_y_error_integrator + tip_y_error,
                    -self._max_integrator_windup,
                    self._max_integrator_windup
                )

            i_gain = 0.15
            target_tip_pos = port_pos.clone()
            target_tip_pos[:, 2] += z_offset
            target_tip_pos[:, 0] += i_gain * self._tip_x_error_integrator
            target_tip_pos[:, 1] += i_gain * self._tip_y_error_integrator

            target_gripper_pos = target_tip_pos + math_utils.quat_apply(target_tip_quat, pos_tg)
            target_gripper_quat = math_utils.quat_mul(target_tip_quat, quat_tg)

            self.env._target_pos_w = target_gripper_pos
            self.env._target_quat_w = target_gripper_quat
            
            obs, _, _, _, _ = self.env.step(self.zero_action)

        print("--- [Cheat Code v2] Finished ---")
        while simulation_app.is_running():
            self.env.step(self.zero_action)

def main():
    env_cfg = CheatCodeV2TaskCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = RelCartesianOSPEnv(env_cfg)

    cheat = CheatCode(env)
    cheat.run()

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
