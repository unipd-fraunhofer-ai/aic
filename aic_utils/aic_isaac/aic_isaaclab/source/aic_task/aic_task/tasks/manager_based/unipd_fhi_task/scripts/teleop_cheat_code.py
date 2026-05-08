# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleoperation script for insertion task in Isaac Lab."""

import argparse
from isaaclab.app import AppLauncher

# 1. Setup Argparse
parser = argparse.ArgumentParser(description="Teleop for insertion task.")
parser.add_argument("--task", type=str, default="Teleop-Task", help="Name of the task.")
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
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.devices.keyboard import Se3Keyboard, Se3KeyboardCfg
from aic_task.tasks.manager_based.unipd_fhi_task import mdp

# Import the base OSP task and env
from aic_task.tasks.manager_based.unipd_fhi_task.config.rel_cart_no_ref import RelCartesianOSPNoRefEnvCfg, RelCartesianOSPNoRefEnv as RelCartesianOSPEnv

##
# Task Configuration
##

@configclass
class TeleopTaskCfg(RelCartesianOSPNoRefEnvCfg):
    """Task configuration specifically for teleop validation."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Modify OSC reference link
        self.osc_ee_body = "sfp_tip_link"
        self.osc_stiffness = (300.0, 300.0, 300.0, 50.0, 50.0, 50.0)

        # Increase sensitivity for teleoperation
        self.action_delta_pos_scale = 1.0
        self.action_delta_ori_scale = 1.0

        # Disable episode timeout for teleop validation
        self.terminations.time_out = None
        self.terminations.failed_insertion = None

        # Disable randomization to keep robot behavior consistent
        self.events.robot_joint_stiffness_and_damping = None
        self.events.joint_friction = None
        self.events.nic_card_physics_material = None

        self.events.reset_scene = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"]),
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
            },
        )

##
# Registration
##

gym.register(
    id="Teleop-Task",
    entry_point=RelCartesianOSPEnv,
    disable_env_checker=True,
    kwargs={
        "cfg": TeleopTaskCfg(),
    },
)

##
# Teleop Implementation
##

class TeleopCheatCode:
    """Heuristic policy that combines programmatic approach with manual teleop."""

    def __init__(self, env: RelCartesianOSPEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device

        # Sensor/Robot references
        self.tip_sensor = env.scene.sensors["sfp_tip_sensor"]
        self.port_sensor = env.scene.sensors["sfp_port_sensor"]
        self.robot = env.scene["robot"]
        self.command_term = env.command_manager.get_term("sfp_port_pose_command")
        
        # Zero action for programmatic steps
        self.zero_action = torch.zeros(self.num_envs, 6, device=self.device)

        # Initialize keyboard
        self.keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.0005, rot_sensitivity=0.0001))
        print(self.keyboard)


    def run(self):
        """Main execution loop."""
        self.env.reset()
        env_ids = torch.arange(self.num_envs, device=self.device)

        print("--- [Teleop Cheat Code] Phase 1: Programmatic Approach ---")

        target_idx = self.command_term.targets_idx
        port_pos = self.port_sensor.data.target_pos_w[env_ids, target_idx]
        port_quat = self.port_sensor.data.target_quat_w[env_ids, target_idx]

        z_offset = 0.07
        
        # Move to approach position 
        target_pos_w = port_pos.clone()
        target_pos_w[:, 2] += z_offset
        
        self.env._target_pos_w = target_pos_w.clone()
        self.env._target_quat_w = port_quat.clone()
        for i in range(75):
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
        
        
        print("--- [Teleop Cheat Code] Phase 2: Keyboard Teleoperation ---")
        print("Use W/S/A/D/Q/E for Translation, Z/X/T/G/C/V for Rotation")

        while simulation_app.is_running():
            # Get keyboard delta
            delta = self.keyboard.advance() # [dx, dy, dz, drx, dry, drz, gripper]
            
            # Action for RelCartesianOSPEnv is [dx, dy, dz, drx, dry, drz]
            # Keyboard returns [dx, dy, dz, drx, dry, drz, gripper]
            action = delta[:6].to(self.device).unsqueeze(0)
            
            # Step the environment with the delta action
            obs, _, _, _, _ = self.env.step(action)
            # print(f"Forces: {obs}")

        print("--- [Teleop Cheat Code] Finished ---")


def main():
    env_cfg = TeleopTaskCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = RelCartesianOSPEnv(env_cfg)

    teleop = TeleopCheatCode(env)
    teleop.run()
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
