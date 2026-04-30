# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect near-completion states for the insertion task."""

import argparse
import os
import torch
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
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.envs.mdp import OperationalSpaceControllerActionCfg
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg

from aic_task.tasks.manager_based.unipd_fhi_task import mdp
from aic_task.tasks.manager_based.unipd_fhi_task.aic_task_base_env import AICTaskBaseEnv

##
# Task Configuration for Collection
##

@configclass
class CollectionTaskCfg(AICTaskBaseEnv):
    """Task configuration specifically for state collection.
    Uses the base environment's randomized reset but with a higher robot Z offset.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        self.actions.arm_action = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            body_name="sfp_tip_link",
            body_offset=None,
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_abs"],
                impedance_mode="fixed",
                motion_control_axes_task=(1, 1, 1, 1, 1, 1),
                contact_wrench_control_axes_task=(0, 0, 0, 0, 0, 0),
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=True,
              
                # Kp
                motion_stiffness_task=(1500.0, 1500.0, 1500.0, 300.0, 300.0, 300.0),

                # choose zeta so that d = 2*sqrt(Kp)*zeta
                motion_damping_ratio_task=(0.5, 0.5, 0.5, 0.25, 0.25, 0.25),
            ),
            position_scale=1.0,
            orientation_scale=1.0,
        )

        # replace implicit actuators with explicit torque actuators
        self.scene.robot.actuators["arm"] = IdealPDActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            stiffness=0.0,
            damping=0.0,
            effort_limit=187.0,
            effort_limit_sim=187.0,
            velocity_limit_sim=100.0,
        )

        # Disable terminations during collection
        self.terminations.time_out = None
        self.terminations.failed_insertion = None

        # Use the randomized reset from mdp
        self.events.reset_scene = EventTerm(
            func=mdp.reset_board_and_robot,
            mode="reset",
            params={
                "board_scene_name": "task_board",
                "board_default_pos": (0.35, -0.30, 0.0),
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
                    "x": (-0.03, 0.03), 
                    "y": (-0.03, 0.03), 
                    "z": (0.10, 0.10),  # Increased Z offset to be higher
                    "roll": (-10.0, 10.0),
                    "pitch": (-10.0, 10.0),
                    "yaw": (-10.0, 10.0),
                },
            },
        )

##
# Registration
##

gym.register(
    id="Collection-Task",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CollectionTaskCfg,
    },
)

##
# State Collector
##

class StateCollector:
    def __init__(self, env: ManagerBasedRLEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device

        # Sensor/Robot references
        self.tip_sensor = env.scene.sensors["sfp_tip_sensor"]
        self.port_sensor = env.scene.sensors["sfp_port_sensor"]
        self.robot = env.scene["robot"]
        self.nic_card = env.scene["nic_card"]
        self.command_term = env.command_manager.get_term("sfp_port_pose_command")

    def get_action(self, target_pos_w, target_quat_w):
        """Converts world target pose to robot base frame and formats for OSC."""
        base_pos = self.robot.data.root_pos_w
        base_quat = self.robot.data.root_quat_w
        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
            base_pos, base_quat, target_pos_w, target_quat_w
        )
        return torch.cat([target_pos_b, target_quat_b], dim=-1)

    def collect_batch(self, noise_xy=0.0015, z_offset_range=(0.01, 0.02), approach_z_offset=0.07, target_dist=0.01):
        """Performs a single reset-approach-capture sequence across all envs."""
        self.env.reset()
        env_ids = torch.arange(self.num_envs, device=self.device)
        env_origins = self.env.scene.env_origins
        
        target_idx = self.command_term.targets_idx
        port_pos = self.port_sensor.data.target_pos_w[env_ids, target_idx]
        port_quat = self.port_sensor.data.target_quat_w[env_ids, target_idx]

        # 1. Randomized near-completion target
        noise = torch.empty((self.num_envs, 2), device=self.device).uniform_(-noise_xy, noise_xy)
        z_final_offset = torch.empty((self.num_envs, 1), device=self.device).uniform_(*z_offset_range)

        target_port_pos = port_pos.clone()
        target_port_pos[:, :2] += noise
        target_port_pos[:, 2] += z_final_offset.squeeze(-1)
        
        # 2. Approach Phase (Higher)
        print("[Batch] Moving to high approach...")
        approach_pos_w = target_port_pos.clone()
        approach_pos_w[:, 2] += approach_z_offset
        action = self.get_action(approach_pos_w, port_quat)
        for _ in range(100):
            self.env.step(action)
            
        # 3. Near-Completion Phase (gradual descent)
        print("[Batch] Moving to near-completion target...")
        current_extra_z = torch.full((self.num_envs, 1), approach_z_offset, device=self.device)
        while torch.any(current_extra_z > 0.0):
            current_extra_z -= 0.0005 
            current_extra_z = torch.clamp(current_extra_z, min=0.0)
            
            target_pos_w = target_port_pos.clone()
            target_pos_w[:, 2] += current_extra_z.squeeze(-1)
            
            action = self.get_action(target_pos_w, port_quat)
            self.env.step(action)
        
        # Phase 4: Wait to settle
        print("[Batch] Waiting to settle...")
        for _ in range(50):
            self.env.step(action)
            
        # 5. Capture States
        print("[Batch] Capturing states...")
        # Success condition: distance from original target <= target_dist
        tip_pos = self.tip_sensor.data.target_pos_w[:, 0]
        target_pos_w[:, :2] -= noise # go back to original pose
        dist_to_target = torch.norm(target_port_pos - tip_pos, dim=-1)
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
            "success": success
        }

def main():
    # Setup environment
    env_cfg = CollectionTaskCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make("Collection-Task", cfg=env_cfg).unwrapped
    
    collector = StateCollector(env)
    
    all_data = {
        "robot_joint_pos": [],
        "robot_joint_vel": [],
        "robot_root_state": [],
        "nic_card_root_state": [],
        "target_idx": [],
        "success": []
    }
    
    collected_count = 0
    while collected_count < args_cli.max_samples:
        batch_data = collector.collect_batch(noise_xy=0.0015)
        
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
