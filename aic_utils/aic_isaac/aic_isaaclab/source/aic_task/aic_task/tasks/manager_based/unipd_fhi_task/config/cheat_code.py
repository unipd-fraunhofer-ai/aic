# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified Cheat code policy and Task configuration for insertion task in Isaac Lab."""

import argparse
from isaaclab.app import AppLauncher

# 1. Setup Argparse
parser = argparse.ArgumentParser(description="Cheat code for insertion task.")
parser.add_argument("--task", type=str, default="CheatCode-Task", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable fabric.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. Launch App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 3. Isaac Lab / Task Imports
import torch
import numpy as np
import gymnasium as gym

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from aic_task.tasks.manager_based.unipd_fhi_task import mdp

# Import the base OSP task and env
from aic_task.tasks.manager_based.unipd_fhi_task.config.rel_cart_osp_no_ref import RelCartesianOSPNoRefEnvCfg, RelCartesianOSPEnv

##
# Task Configuration
##

def reset(env, env_ids):
    device = env.device
    env_origins = env.scene.env_origins[env_ids]
    n = len(env_ids)

    # Reset robot
    robot = env.scene["robot"]
    robot_pos = torch.tensor([0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(n, -1)
    robot_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(n, -1)
    robot.write_root_pose_to_sim(torch.cat([robot_pos + env_origins, robot_rot], dim=-1), env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # Reset NIC card
    nic_card = env.scene["nic_card"]
    nic_card_pos = torch.tensor([-0.397, 0.208, 0.102], device=device).unsqueeze(0).expand(n, -1)
    nic_card_rot = torch.tensor([0.0, 0.0, -0.7068252, 0.7073883], device=device).unsqueeze(0).expand(n, -1)
    nic_card.write_root_pose_to_sim(torch.cat([nic_card_pos + env_origins, nic_card_rot], dim=-1), env_ids=env_ids)
    nic_card.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=env_ids)

@configclass
class CheatCodeTaskCfg(RelCartesianOSPNoRefEnvCfg):
    """Task configuration specifically for cheatcode validation."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Modify OSC reference link
        self.osc_ee_body = "sfp_tip_link"

        # Disable episode timeout
        self.terminations.time_out = None
        self.terminations.failed_insertion = None

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
    id="CheatCode-Task",
    entry_point=RelCartesianOSPEnv,
    disable_env_checker=True,
    kwargs={
        "cfg": CheatCodeTaskCfg(),
    },
)

##
# Policy Implementation
##

class CheatCode:

    def __init__(self, env: RelCartesianOSPEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device

        # Sensor/Robot references
        self.tip_sensor = env.scene.sensors["sfp_tip_sensor"]
        self.port_sensor = env.scene.sensors["sfp_port_sensor"]
        self.robot = env.scene["robot"]
        self.command_term = env.command_manager.get_term("sfp_port_pose_command")
        
        # Zero action for the step method (RelCartesianOSPEnv expects a 6D delta action)
        self.zero_action = torch.zeros(self.num_envs, 6, device=self.device)

    def run(self):
        """Executes the two-phase insertion sequence."""
        self.env.reset()
        env_ids = torch.arange(self.num_envs, device=self.device)
        
        print("--- [Cheat Code] Starting execution loop ---")
                
        target_idx = self.command_term.targets_idx
        port_pos = self.port_sensor.data.target_pos_w[env_ids, target_idx]
        port_quat = self.port_sensor.data.target_quat_w[env_ids, target_idx]

        z_offset = 0.07
        
        # Move to approach position 
        target_pos_w = port_pos.clone()
        target_pos_w[:, 2] += z_offset
        
        # Directly set target pose for the OSC controller (without delta action)
        self.env._target_pos_w = target_pos_w.clone()
        self.env._target_quat_w = port_quat.clone()
        for i in range(100):
            obs, _, _, _, _ = self.env.step(self.zero_action)
        
        # Compute errors
        tip_pos = self.tip_sensor.data.target_pos_w[:, 0]
        tip_quat = self.tip_sensor.data.target_quat_w[:, 0]
        pos_error_w = target_pos_w - tip_pos
        quat_error_w = math_utils.quat_mul(port_quat, math_utils.quat_inv(tip_quat))
        axis_angle_error_w = math_utils.axis_angle_from_quat(quat_error_w)
        pos_error_mag = torch.norm(pos_error_w[0])
        rot_error_mag = torch.norm(axis_angle_error_w[0])
        print(f"[APPROACH] Pos Error Mag: {pos_error_mag:.4f}, Rot Error Mag: {rot_error_mag:.4f}")
        
        # Wait a bit
        for i in range(100):
            obs, _, _, _, _ = self.env.step(self.zero_action)

        # Move to insertion position
        while z_offset >= 0.0:
            z_offset -= 0.0005
            
            target_pos_w = port_pos.clone()
            target_pos_w[:, 2] += z_offset
            
            self.env._target_pos_w = target_pos_w.clone()
            self.env._target_quat_w = port_quat.clone()
            obs, _, _, _, _ = self.env.step(self.zero_action)
        
        tip_pos = self.tip_sensor.data.target_pos_w[:, 0]
        tip_quat = self.tip_sensor.data.target_quat_w[:, 0]
        pos_error_w = port_pos - tip_pos
        quat_error_w = math_utils.quat_mul(port_quat, math_utils.quat_inv(tip_quat))
        axis_angle_error_w = math_utils.axis_angle_from_quat(quat_error_w)
        pos_error_mag = torch.norm(pos_error_w[0])
        rot_error_mag = torch.norm(axis_angle_error_w[0])
        print(f"[INSERTION] Pos Error Mag: {pos_error_mag:.4f}, Rot Error Mag: {rot_error_mag:.4f}")

        print("--- [Cheat Code] Finished ---")

        # Keep running until simulation is closed
        while simulation_app.is_running():
            obs, _, _, _, _ = self.env.step(self.zero_action)


def main():
    env_cfg = CheatCodeTaskCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = RelCartesianOSPEnv(env_cfg)

    cheat = CheatCode(env)
    cheat.run()

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()