# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect near-completion states for the insertion task."""

import os
import torch
import argparse
from isaaclab.app import AppLauncher

# 1. Setup Argparse
parser = argparse.ArgumentParser(description="Collect near-completion states for insertion task.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to spawn.")
parser.add_argument("--max_samples", type=int, default=10000, help="Total number of samples to collect.")
parser.add_argument("--output", type=str, default="near_completion_states.pt", help="Output file name.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. Launch App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 3. Isaac Lab / Task Imports
import torch
import gymnasium as gym
import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm

from aic_task.tasks.manager_based.unipd_fhi_task import mdp
from aic_task.tasks.manager_based.unipd_fhi_task.config.rel_cart_osp_no_ref import RelCartesianOSPNoRefEnvCfg, RelCartesianOSPEnv

##
# Task Configuration for Collection
##

@configclass
class CollectionTaskCfg(RelCartesianOSPNoRefEnvCfg):
    """Task configuration specifically for state collection.
    Uses the base environment's randomized reset but with a higher robot Z offset.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

       # Modify OSC reference link
        self.osc_ee_body = "sfp_tip_link"

        # Disable terminations during collection
        self.terminations.time_out = None
        self.terminations.failed_insertion = None

        # Use the randomized reset from mdp
        self.events.reset_scene = EventTerm(
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
                "target_ee_offset_asset_name": "nic_card",
                "ee_offset_range": {
                    "x": (0.00, 0.00), 
                    "y": (0.00, 0.00), 
                    "z": (0.13, 0.13),  # Increased Z offset to be higher
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            },
        )

##
# Registration
##

gym.register(
    id="Collection-Task",
    entry_point=RelCartesianOSPEnv,
    disable_env_checker=True,
    kwargs={
        "cfg": CollectionTaskCfg(),
    },
)

##
# State Collector
##

class StateCollector:
    def __init__(self, env: RelCartesianOSPEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device

        # Sensor/Robot references
        self.tip_sensor = env.scene.sensors["sfp_tip_sensor"]
        self.port_sensor = env.scene.sensors["sfp_port_sensor"]
        self.robot = env.scene["robot"]
        self.nic_card = env.scene["nic_card"]
        self.command_term = env.command_manager.get_term("sfp_port_pose_command")
        
        self.zero_action = torch.zeros(self.num_envs, 6, device=self.device)

    def collect_batch(self, noise_xy=0.0015, noise_yaw=10.0, z_offset_range=(0.01, 0.03), approach_z_offset=0.07, target_dist=0.005):
        """Performs a single reset-approach-capture sequence across all envs."""
        self.env.reset()
        env_ids = torch.arange(self.num_envs, device=self.device)
        env_origins = self.env.scene.env_origins
        
        target_idx = self.command_term.targets_idx
        port_pos = self.port_sensor.data.target_pos_w[env_ids, target_idx]
        port_quat = self.port_sensor.data.target_quat_w[env_ids, target_idx]

        # 1. Randomized near-completion target
        xy_noise = torch.empty((self.num_envs, 2), device=self.device).uniform_(-noise_xy, noise_xy)
        yaw_noise = torch.empty((self.num_envs, 1), device=self.device).uniform_(-noise_yaw, noise_yaw)
        z_final_offset = torch.empty((self.num_envs, 1), device=self.device).uniform_(*z_offset_range)

        # Final Target
        target_pos_w = port_pos.clone()
        target_pos_w[:, 2] += z_final_offset.squeeze(-1)
        
        # Approach Target
        approach_pos_w = target_pos_w.clone()
        approach_pos_w[:, :2] += xy_noise
        approach_pos_w[:, 2] += approach_z_offset

        yaw_noise_rad = torch.deg2rad(yaw_noise.squeeze(-1))
        noise_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(yaw_noise_rad), 
            torch.zeros_like(yaw_noise_rad), 
            yaw_noise_rad
        )
        approach_rot_w = math_utils.quat_mul(port_quat, noise_quat)
        
        # 2. Approach Phase (Higher)
        print("[Batch] Moving to high approach...")        
        self.env._target_pos_w = approach_pos_w.clone()
        self.env._target_quat_w = approach_rot_w.clone()
        for _ in range(100):
            self.env.step(self.zero_action)
            
        # 3. Near-Completion Phase (gradual descent)
        print("[Batch] Moving to near-completion target...")
        z_descent_step = 0.0005
        num_steps = int(approach_z_offset / z_descent_step)
        for _ in range(num_steps):
            approach_pos_w[:, 2] -= z_descent_step
            
            self.env._target_pos_w = approach_pos_w.clone()
            self.env._target_quat_w = approach_rot_w.clone()
            self.env.step(self.zero_action)
        
        # Phase 4: Wait to settle
        print("[Batch] Waiting to settle...")
        for _ in range(50):
            self.env.step(self.zero_action)
            
        # 5. Capture States
        print("[Batch] Capturing states...")
        # Success: distance from final target <= target_dist
        tip_pos = self.tip_sensor.data.target_pos_w[:, 0]
        dist_to_target = torch.norm(target_pos_w - tip_pos, dim=-1)
        success = (dist_to_target <= target_dist).float()
        print(f"[Batch] Sample success rate: {success.mean().item():.4f}, avg dist: {dist_to_target.mean().item():.4f}, min dist: {dist_to_target.min().item():.4f}, max dist: {dist_to_target.max().item():.4f} ")
        
        # Robot states
        robot_joint_pos = self.robot.data.joint_pos.clone()
        robot_joint_vel = self.robot.data.joint_vel.clone()
        robot_root_state = self.robot.data.root_state_w.clone()
        robot_root_state[:, :3] -= env_origins
        
        # NIC Card states
        nic_card_root_state = self.nic_card.data.root_state_w.clone()
        nic_card_root_state[:, :3] -= env_origins
        
        return {
            "robot_joint_pos": robot_joint_pos,
            "robot_joint_vel": robot_joint_vel,
            "robot_root_state": robot_root_state,
            "nic_card_root_state": nic_card_root_state,
            "target_idx": target_idx.clone(),
            "success": success.clone(),
            "dist_to_target": dist_to_target.clone()
        }

def main():
    # Setup environment
    env_cfg = CollectionTaskCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # We need to register the task first if we want to use gym.make
    # Or we can just instantiate the env class
    env = RelCartesianOSPEnv(env_cfg)
    
    collector = StateCollector(env)
    
    all_data = {
        "robot_joint_pos": [],
        "robot_joint_vel": [],
        "robot_root_state": [],
        "nic_card_root_state": [],
        "target_idx": [],
        "success": [],
        "dist_to_target": []
    }
    
    collected_count = 0
    while collected_count < args_cli.max_samples:
        batch_data = collector.collect_batch(noise_xy=0.0015, noise_yaw=5.0)
        
        for key in all_data:
            all_data[key].append(batch_data[key].cpu())
            
        collected_count += args_cli.num_envs
        print(f"--- [Progress] Collected {collected_count}/{args_cli.max_samples} samples ---")
    
    # Concatenate all batches
    final_data = {}
    for key in all_data:
        final_data[key] = torch.cat(all_data[key], dim=0)[:args_cli.max_samples]
    final_data["num_samples"] = args_cli.max_samples
    
    num_samples = final_data['num_samples']
    success_rate = final_data['success'].mean().item()
    print(f"--- [Results] Collected {num_samples} states with {success_rate*100:.2f}% success rate.")
    
    # Save data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, args_cli.output)
    torch.save(final_data, output_path)
    print(f"--- [Finished] Saved states to {output_path} ---")
    
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
