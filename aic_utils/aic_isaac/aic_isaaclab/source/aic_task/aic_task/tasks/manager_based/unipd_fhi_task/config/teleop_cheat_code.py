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
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.envs.mdp import OperationalSpaceControllerActionCfg
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.devices.keyboard import Se3Keyboard, Se3KeyboardCfg
from aic_task.tasks.manager_based.unipd_fhi_task import mdp

# Import the base task environment
from aic_task.tasks.manager_based.unipd_fhi_task.aic_task_base_env import AICTaskBaseEnv

##
# Task Configuration
##

@configclass
class TeleopTaskCfg(AICTaskBaseEnv):
    """Task configuration specifically for teleop validation."""

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

        # Disable episode timeout for teleop validation
        self.terminations.time_out = None

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
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TeleopTaskCfg,
    },
)

##
# Teleop Implementation
##

class TeleopCheatCode:
    """Heuristic policy that combines programmatic approach with manual teleop."""

    def __init__(self, env: ManagerBasedRLEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device

        # XY Error Integrators
        self._tip_x_error_integrator = torch.zeros(self.num_envs, device=self.device)
        self._tip_y_error_integrator = torch.zeros(self.num_envs, device=self.device)
        self._max_integrator_windup = 0.05
        self.i_gain = 0.15

        # Action scale
        self.action_scale = 1.0
        
        # Sensor/Robot references
        self.tip_sensor = env.scene.sensors["sfp_tip_sensor"]
        self.port_sensor = env.scene.sensors["sfp_port_sensor"]
        self.robot = env.scene["robot"]
        self.command_term = env.command_manager.get_term("sfp_port_pose_command")
        
        # Body ID for the control link (wrist_3_link)
        self.ee_body_id = self.robot.find_bodies("wrist_3_link")[0][0]

        # Initialize keyboard
        self.keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.0005, rot_sensitivity=0.001))
        print(self.keyboard)


    def get_action_from_target(self, target_pos, target_quat):
        """Compute the absolute target for the Differential IK controller."""
        base_pos = self.robot.data.root_pos_w
        base_quat = self.robot.data.root_quat_w

        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
            base_pos,
            base_quat,
            target_pos,
            target_quat,
        )
     
        # Action = target positions in base frame
        action = torch.cat([target_pos_b, target_quat_b], dim=-1)
        return action

    def run(self):
        """Main execution loop."""
        obs, _ = self.env.reset()
        env_ids = torch.arange(self.num_envs, device=self.device)

        print("--- [Teleop Cheat Code] Phase 1: Programmatic Approach ---")

        target_idx = self.command_term.targets_idx
        port_pos = self.port_sensor.data.target_pos_w[env_ids, target_idx]
        port_quat = self.port_sensor.data.target_quat_w[env_ids, target_idx]

        z_offset = 0.07
        
        # Move to approach position 
        target_pos_w = port_pos.clone()
        target_pos_w[:, 2] += z_offset
        action = self.get_action_from_target(target_pos_w, port_quat)
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
        
        
        print("--- [Teleop Cheat Code] Phase 2: Keyboard Teleoperation ---")
        print("Use W/S/A/D/Q/E for Translation, Z/X/T/G/C/V for Rotation")

        # Current target pose (starting from where approach left off)
        cur_target_p = target_pos_w.clone()
        cur_target_q = port_quat.clone()

        while simulation_app.is_running():
            # Get keyboard delta
            delta = self.keyboard.advance() # [dx, dy, dz, drx, dry, drz, gripper]
            
            # Apply translation delta (in world frame for simplicity, or we could rotate it)
            # Keyboard returns delta in a generic frame. Let's assume it's aligned with world for now.
            cur_target_p += delta[:3].to(self.device).unsqueeze(0)
            
            # Apply rotation delta
            d_quat = math_utils.quat_from_euler_xyz(delta[3], delta[4], delta[5]).to(self.device).unsqueeze(0)
            cur_target_q = math_utils.quat_mul(d_quat, cur_target_q)

            # Compute action
            actions = self.get_action_from_target(cur_target_p, cur_target_q)
            obs, _, _, _, _ = self.env.step(actions)
            print(f"Forces: {obs}")

        print("--- [Teleop Cheat Code] Finished ---")


def main():
    # Load task config
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # Instantiate environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    # Initialize and run teleop cheat code
    teleop = TeleopCheatCode(env)
    teleop.run()
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
