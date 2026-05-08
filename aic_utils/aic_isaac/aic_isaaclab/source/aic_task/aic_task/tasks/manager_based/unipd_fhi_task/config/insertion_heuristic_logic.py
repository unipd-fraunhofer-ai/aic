import torch
import numpy as np
from enum import IntEnum
import isaaclab.utils.math as math_utils

class InsertCableState(IntEnum):
    MOVE_ABOVE_PORT = 0
    DESCEND_AND_INSERT = 1
    STABILIZE_XY = 2
    UNTILT_INSERTED_CABLE = 3
    REDESCEND = 4
    FIX_YAW = 5
    STABILIZE = 6
    DONE = 7

class InsertionHeuristicLogic:
    """Vectorized state-machine heuristic for plug insertion."""

    def __init__(self, num_envs, device, pos_tg, quat_tg):
        self.num_envs = num_envs
        self.device = device
        
        # Relative transform from tip to gripper (pre-computed in env)
        self.pos_tg = pos_tg
        self.quat_tg = quat_tg

        # Approach constants
        self.approach_z_offset = 0.07
        self.approach_roll_offset = np.deg2rad(15)
        self.approach_pitch_offset = np.deg2rad(15)

        # --- State Buffers ---
        self.states = torch.full((num_envs,), InsertCableState.MOVE_ABOVE_PORT, device=device, dtype=torch.long)
        self.z_offsets = torch.full((num_envs,), self.approach_z_offset, device=device)
        self.roll_angles = torch.full((num_envs,), self.approach_roll_offset, device=device)
        self.pitch_angles = torch.full((num_envs,), self.approach_pitch_offset, device=device)
        self.yaw_angles = torch.zeros(num_envs, device=device)
        self.port_pos = torch.zeros(num_envs, 3, device=device)
        self.port_quat = torch.zeros(num_envs, 4, device=device)
        
        # Step counters for time-based transitions
        self.step_counts = torch.zeros(num_envs, device=device, dtype=torch.long)
        
        # Fix Yaw helpers
        self.fix_yaw_direction = torch.ones(num_envs, device=device)
        self.fix_yaw_attempt = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.fix_yaw_start_z = torch.full((num_envs,), 10.0, device=device)
        self.fix_yaw_best_z = torch.full((num_envs,), 10.0, device=device)
        self.fix_yaw_no_improve_steps = torch.zeros(num_envs, device=device, dtype=torch.long)

        # Logic constants
        self.port_entrance_offset = 0.045
        self.z_force_threshold = -1.0 # [N]
        self.untilt_step = np.deg2rad(0.5)
        self.target_stabilize_steps = 75
        self.fix_yaw_limit = np.deg2rad(10.0)
        self.fix_yaw_step = np.deg2rad(0.2)
        self.fix_yaw_max_no_improve_steps = 100
        self.fix_yaw_min_z_improvement = 0.001

    def reset_idx(self, env_ids):
        """Resets the state of specific environments."""
        self.states[env_ids] = InsertCableState.MOVE_ABOVE_PORT
        self.z_offsets[env_ids] = self.approach_z_offset
        self.roll_angles[env_ids] = self.approach_roll_offset
        self.pitch_angles[env_ids] = self.approach_pitch_offset
        self.yaw_angles[env_ids] = 0.0
        self.step_counts[env_ids] = 0
        self.fix_yaw_direction[env_ids] = 1.0
        self.fix_yaw_attempt[env_ids] = 0
        self.fix_yaw_start_z[env_ids] = 10.0
        self.fix_yaw_best_z[env_ids] = 10.0
        self.fix_yaw_no_improve_steps[env_ids] = 0
        self.port_pos[env_ids] = torch.zeros((env_ids.shape[0], 3), device=self.device)
        self.port_quat[env_ids] = torch.zeros((env_ids.shape[0], 4), device=self.device)

    def set_target(self, port_pos, port_quat):
        """Set the target port pose (world frame)."""
        self.port_pos = port_pos
        self.port_quat = port_quat

    def compute(self, tip_pos, tip_force_z):
        """
        Computes the base target pose and stiffness for the current step.
        
        Args:
            tip_pos: (N, 3) World-frame tip position.
            tip_force_z: (N,) Z-axis contact force on the tip.
            port_pos: (N, 3) World-frame port position.
            port_quat: (N, 4) World-frame port orientation.
            
        Returns:
            dict: Containing 'target_pos', 'target_quat', and 'stiffness'.
        """
        self.step_counts += 1

        # --- State Transitions ---
        
        # 0. MOVE_ABOVE_PORT -> DESCEND_AND_INSERT
        mask = (self.states == InsertCableState.MOVE_ABOVE_PORT)
        if mask.any():
            # Wait for 100 steps to reach the noisy target above the port
            finished_move = self.step_counts >= 30
            self.states[mask & finished_move] = InsertCableState.DESCEND_AND_INSERT
            self.step_counts[mask & finished_move] = 0 # Reset counter for next state

        # 1. DESCEND_AND_INSERT -> STABILIZE_XY
        mask = (self.states == InsertCableState.DESCEND_AND_INSERT)
        if mask.any():
            contact = tip_force_z > self.z_force_threshold
            print("Tip force z: ", tip_force_z)
            depth_reached = self.z_offsets < 0.0
                        
            # If contact, go to STABILIZE_XY to try and find the hole
            self.states[mask & contact] = InsertCableState.STABILIZE_XY
            # If depth reached we are likely fully inserted
            self.states[mask & depth_reached] = InsertCableState.STABILIZE
            self.z_offsets[mask] -= 0.001

        # 3. STABILIZE_XY -> UNTILT / FIX_YAW
        mask = (self.states == InsertCableState.STABILIZE_XY)
        if mask.any():
            # Current depth check relative to port
            depth_threshold = self.port_entrance_offset - 0.03
            finished_stabilize = self.z_offsets < depth_threshold
            if finished_stabilize.any():
                m = mask & finished_stabilize

                # Update port position in local frame
                self.port_pos[m, 0] = tip_pos[m, 0]
                self.port_pos[m, 1] = tip_pos[m, 1]

                # TODO: here we should check if we are aligned, if not we should go to FIX_YAW
                self.states[m] = InsertCableState.UNTILT_INSERTED_CABLE

            self.z_offsets[mask] -= 0.001

        # 4. FIX_YAW logic (TODO: FIX THIS)
        mask = (self.states == InsertCableState.FIX_YAW)
        if mask.any():
            current_z = tip_pos[:, 2]
            port_z = self.port_pos[:, 2] + self.port_entrance_offset
            z_dist = current_z - port_z
            plug_inserted = z_dist < 0

            # Init trackers for new entries
            new_fix_yaw = mask & (self.fix_yaw_start_z == 10.0)
            self.fix_yaw_start_z[new_fix_yaw] = current_z[new_fix_yaw]
            self.fix_yaw_best_z[new_fix_yaw] = current_z[new_fix_yaw]
            self.fix_yaw_no_improve_steps[new_fix_yaw] = 0

            # Success check
            self.states[mask & plug_inserted] = InsertCableState.UNTILT_INSERTED_CABLE
            
            # Improvement check
            improved = mask & (current_z < self.fix_yaw_best_z - self.fix_yaw_min_z_improvement)
            self.fix_yaw_best_z[improved] = current_z[improved]
            self.fix_yaw_no_improve_steps[improved] = 0
            self.fix_yaw_no_improve_steps[mask & ~improved] += 1
            
            # Update yaw
            self.yaw_angles[mask] += self.fix_yaw_direction[mask] * self.fix_yaw_step
            
            # Failure / Direction Flip
            limit_reached = torch.abs(self.yaw_angles) >= self.fix_yaw_limit
            no_improve = self.fix_yaw_no_improve_steps >= self.fix_yaw_max_no_improve_steps
            failure = mask & (limit_reached | no_improve)
            
            if failure.any():
                self.fix_yaw_attempt[failure] += 1
                # Flip direction if first attempt
                first_try = failure & (self.fix_yaw_attempt < 2)
                self.fix_yaw_direction[first_try] *= -1.0
                self.fix_yaw_start_z[first_try] = 10.0 # Trigger re-init
                self.yaw_angles[first_try] = 0.0 # Optional: reset yaw? Reference doesn't reset yaw angle, just continues
                
                # Full failure
                final_fail = failure & (self.fix_yaw_attempt >= 2)
                self.states[final_fail] = InsertCableState.UNTILT_INSERTED_CABLE

        # 5. UNTILT_INSERTED_CABLE -> REDESCEND
        mask = (self.states == InsertCableState.UNTILT_INSERTED_CABLE)
        if mask.any():
            self.roll_angles[mask] -= torch.sign(self.roll_angles[mask]) * self.untilt_step
            self.pitch_angles[mask] -= torch.sign(self.pitch_angles[mask]) * self.untilt_step
            # Clamp to 0
            self.roll_angles[mask] = torch.where(torch.abs(self.roll_angles[mask]) < self.untilt_step, 0.0, self.roll_angles[mask])
            self.pitch_angles[mask] = torch.where(torch.abs(self.pitch_angles[mask]) < self.untilt_step, 0.0, self.pitch_angles[mask])
            
            done_untilt = (self.roll_angles == 0.0) & (self.pitch_angles == 0.0) & (self.z_offsets >= (self.port_entrance_offset - 0.005))
            self.states[mask & done_untilt] = InsertCableState.REDESCEND
            self.z_offsets[mask] += 0.0005

        # 6. REDESCEND -> STABILIZE
        mask = (self.states == InsertCableState.REDESCEND)
        if mask.any():
            self.z_offsets[mask] -= 0.001
            self.states[mask & (self.z_offsets < -0.015)] = InsertCableState.STABILIZE
            self.step_counts[mask & (self.z_offsets < -0.015)] = 0 # Reset for stabilize

        # 7. STABILIZE -> DONE
        mask = (self.states == InsertCableState.STABILIZE)
        if mask.any():
            self.states[mask & (self.step_counts >= self.target_stabilize_steps)] = InsertCableState.DONE

        # --- Compute Outputs ---

        # Stiffness
        stiffness = torch.tensor([90.0, 90.0, 90.0, 50.0, 50.0, 50.0], device=self.device).repeat(self.num_envs, 1)
        stiffness[self.states == InsertCableState.MOVE_ABOVE_PORT] = torch.tensor([1000.0, 1000.0, 1000.0, 150.0, 150.0, 150.0], device=self.device)
        stiffness[self.states == InsertCableState.DESCEND_AND_INSERT] = torch.tensor([90.0, 90.0, 400.0, 50.0, 50.0, 50.0], device=self.device)
        stiffness[self.states == InsertCableState.STABILIZE_XY] = torch.tensor([10.0, 10.0, 90.0, 5.0, 5.0, 5.0], device=self.device)
        stiffness[self.states == InsertCableState.FIX_YAW] = torch.tensor([10.0, 10.0, 90.0, 25.0, 25.0, 200.0], device=self.device)
        stiffness[self.states == InsertCableState.UNTILT_INSERTED_CABLE] = torch.tensor([20.0, 20.0, 50.0, 50.0, 50.0, 200.0], device=self.device)
        stiffness[self.states == InsertCableState.REDESCEND] = torch.tensor([90.0, 90.0, 90.0, 200.0, 200.0, 200.0], device=self.device)

        # Target Pose
        tilt_quat = math_utils.quat_from_euler_xyz(self.roll_angles, self.pitch_angles, self.yaw_angles)
        target_tip_quat = math_utils.quat_mul(self.port_quat, tilt_quat)
        
        target_tip_pos = self.port_pos.clone()
        target_tip_pos[:, 2] += self.z_offsets

        # Final Gripper Pose
        target_ee_pos = target_tip_pos + math_utils.quat_apply(target_tip_quat, self.pos_tg)
        target_ee_quat = math_utils.quat_mul(target_tip_quat, self.quat_tg)

        print("current states: ", self.states)

        return {
            "target_pos": target_ee_pos,
            "target_quat": target_ee_quat,
            "stiffness": stiffness,
            "state": self.states
        }
