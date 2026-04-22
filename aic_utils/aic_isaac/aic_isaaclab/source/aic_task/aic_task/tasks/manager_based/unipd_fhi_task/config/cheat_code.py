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
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.envs.mdp import DifferentialInverseKinematicsActionCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from aic_task.tasks.manager_based.unipd_fhi_task import mdp

# Import the base task environment
from aic_task.tasks.manager_based.unipd_fhi_task.aic_task_base_env import AICTaskBaseEnv

##
# Task Configuration
##

@configclass
class CheatCodeTaskCfg(AICTaskBaseEnv):
    """Task configuration specifically for cheatcode validation."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder.*", "elbow.*", "wrist.*"],
            body_name="sfp_tip_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            ),            
            scale=1.0,
        )

        # Disable episode timeout
        self.terminations.time_out = None

        # Disable randomization
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
    id="CheatCode-Task",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CheatCodeTaskCfg,
    },
)

##
# Policy Implementation
##

class CheatCode:
    """Refactored Heuristic policy for Peg-in-Hole task."""

    def __init__(self, env: ManagerBasedRLEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device

        # Sensor/Robot references
        self.tip_sensor = env.scene.sensors["sfp_tip_sensor"]
        self.port_sensor = env.scene.sensors["sfp_port_sensor"]
        self.robot = env.scene["robot"]
        self.command_term = env.command_manager.get_term("sfp_port_pose_command")

    def get_action(self, target_pos_w, target_quat_w):
        """Converts world target pose to robot base frame and formats for IK."""
        base_pos = self.robot.data.root_pos_w
        base_quat = self.robot.data.root_quat_w

        # Transform target to robot base frame
        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
            base_pos, base_quat, target_pos_w, target_quat_w
        )

        # Concatenate for command
        action = torch.cat([target_pos_b, target_quat_b], dim=-1)
        return action

    def print_debug_poses(self):
        """Prints world poses of wrist_3_link and the tip sensor."""

        self.ee_body_id = self.robot.find_bodies("wrist_3_link")[0][0]

        # Wrist 3 link pose
        wrist_pos = self.robot.data.body_pos_w[:, self.ee_body_id]
        wrist_quat = self.robot.data.body_quat_w[:, self.ee_body_id]
        
        # Tip sensor pose
        tip_pos = self.tip_sensor.data.target_pos_w[:, 0]
        tip_quat = self.tip_sensor.data.target_quat_w[:, 0]

        # Tip relative to wrist
        tip_pos_b, tip_quat_b = math_utils.subtract_frame_transforms(
            wrist_pos,
            wrist_quat,
            tip_pos,
            tip_quat,
        )
        print(f"--- [POSE DEBUG] ---")
        print(f"  Tip (Body):     pos={tip_pos_b[0].cpu().numpy()}, quat={tip_quat_b[0].cpu().numpy()}")

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
        action = self.get_action(target_pos_w, port_quat)
        for i in range(75):
            obs, _, _, _, _ = self.env.step(action)
        
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
            obs, _, _, _, _ = self.env.step(action)

        # Move to insertion position
        while z_offset > -0.005:
            z_offset -= 0.0005
            
            target_pos_w = port_pos.clone()
            target_pos_w[:, 2] += z_offset
            action = self.get_action(target_pos_w, port_quat)
            obs, _, _, _, _ = self.env.step(action)
        
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
            obs, _, _, _, _ = self.env.step(action)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    cheat = CheatCode(env)
    cheat.run()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()