# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the aic task (UR5e assembly with task board).

Includes:
- Command-tracking rewards with exponential / tanh kernels (inspired by the
  gear-assembly deploy environment).
- A sparse reaching bonus.
- Smoothness and safety penalties (torques, joint acceleration, action rate).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Insertion-pose tracking (positon)
# ---------------------------------------------------------------------------


def insertion_position_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    tip_sensor_cfg: SceneEntityCfg,
    port_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward tracking of the peg tip position using the tanh kernel."""
    # Get sensors
    tip_sensor: FrameTransformer = env.scene.sensors[tip_sensor_cfg.name]
    port_sensor: FrameTransformer = env.scene.sensors[port_sensor_cfg.name]

    # Get active target index from command generator
    command_term = env.command_manager.get_term(command_name)
    target_idx = command_term.targets_idx

    # Get world poses
    tip_pos_w = tip_sensor.data.target_pos_w[:, 0]
    port_pos_w = port_sensor.data.target_pos_w[torch.arange(env.num_envs), target_idx]

    # Calculate distance and reward
    distance = torch.norm(tip_pos_w - port_pos_w, dim=1)
    return torch.clamp(1 - distance, min=0.0)

def insertion_position_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    tip_sensor_cfg: SceneEntityCfg,
    port_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward tracking of the peg tip position using the tanh kernel."""
    # Get sensors
    tip_sensor: FrameTransformer = env.scene.sensors[tip_sensor_cfg.name]
    port_sensor: FrameTransformer = env.scene.sensors[port_sensor_cfg.name]

    # Get active target index from command generator
    command_term = env.command_manager.get_term(command_name)
    target_idx = command_term.targets_idx

    # Get world poses
    tip_pos_w = tip_sensor.data.target_pos_w[:, 0]
    port_pos_w = port_sensor.data.target_pos_w[torch.arange(env.num_envs), target_idx]

    # Calculate distance and reward
    distance = torch.norm(tip_pos_w - port_pos_w, dim=1)
    return 1.0 - torch.tanh(distance / std)

def insertion_completed(
    env: ManagerBasedRLEnv,
    command_name: str,
    tip_sensor_cfg: SceneEntityCfg,
    port_sensor_cfg: SceneEntityCfg,
    threshold: float = 0.005,
) -> torch.Tensor:
    """Sparse reward for completing the insertion (distance < threshold)."""
    # Get sensors
    tip_sensor: FrameTransformer = env.scene.sensors[tip_sensor_cfg.name]
    port_sensor: FrameTransformer = env.scene.sensors[port_sensor_cfg.name]

    # Get active target index
    command_term = env.command_manager.get_term(command_name)
    target_idx = command_term.targets_idx

    # Get world positions
    tip_pos_w = tip_sensor.data.target_pos_w[:, 0]
    port_pos_w = port_sensor.data.target_pos_w[torch.arange(env.num_envs), target_idx]

    # Calculate distance
    distance = torch.norm(tip_pos_w - port_pos_w, dim=1)
    
    return (distance < threshold).float()


# ---------------------------------------------------------------------------
# Insertion-pose tracking (orientation)
# ---------------------------------------------------------------------------


def insertion_orientation_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    tip_sensor_cfg: SceneEntityCfg,
    port_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward tracking of the peg tip orientation using the tanh kernel."""
    # Get sensors
    tip_sensor: FrameTransformer = env.scene.sensors[tip_sensor_cfg.name]
    port_sensor: FrameTransformer = env.scene.sensors[port_sensor_cfg.name]

    # Get active target index from command generator
    command_term = env.command_manager.get_term(command_name)
    target_idx = command_term.targets_idx

    # Get world orientations
    tip_quat_w = tip_sensor.data.target_quat_w[:, 0]
    port_quat_w = port_sensor.data.target_quat_w[torch.arange(env.num_envs), target_idx]

    # Calculate error and reward
    ang_error = quat_error_magnitude(tip_quat_w, port_quat_w)
    return torch.clamp(1 - ang_error, min=0.0)


def insertion_orientation_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    tip_sensor_cfg: SceneEntityCfg,
    port_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward tracking of the peg tip orientation using the tanh kernel."""
    # Get sensors
    tip_sensor: FrameTransformer = env.scene.sensors[tip_sensor_cfg.name]
    port_sensor: FrameTransformer = env.scene.sensors[port_sensor_cfg.name]

    # Get active target index from command generator
    command_term = env.command_manager.get_term(command_name)
    target_idx = command_term.targets_idx

    # Get world orientations
    tip_quat_w = tip_sensor.data.target_quat_w[:, 0]
    port_quat_w = port_sensor.data.target_quat_w[torch.arange(env.num_envs), target_idx]

    # Calculate error and reward
    ang_error = quat_error_magnitude(tip_quat_w, port_quat_w)
    return 1.0 - torch.tanh(ang_error / std)


# ---------------------------------------------------------------------------
# Pose tracking (4-point distance)
# ---------------------------------------------------------------------------


def pose_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    tip_sensor_cfg: SceneEntityCfg,
    port_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Compute mean distance between 4 frame keypoints (origin + 3 axes at 0.15m) for insertion."""
    # Get sensors
    tip_sensor: FrameTransformer = env.scene.sensors[tip_sensor_cfg.name]
    port_sensor: FrameTransformer = env.scene.sensors[port_sensor_cfg.name]

    # Get active target index
    command_term = env.command_manager.get_term(command_name)
    target_idx = command_term.targets_idx
    env_ids = torch.arange(env.num_envs, device=env.device)

    # Get world poses
    curr_pos = tip_sensor.data.target_pos_w[:, 0]
    curr_quat = tip_sensor.data.target_quat_w[:, 0]
    des_pos = port_sensor.data.target_pos_w[env_ids, target_idx]
    des_quat = port_sensor.data.target_quat_w[env_ids, target_idx]

    # Define 4 points in local frame: origin + 3 axes at 0.15m
    offsets = torch.tensor(
        [[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [0.0, 0.15, 0.0], [0.0, 0.0, 0.15]],
        device=env.device,
    )
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs * 4, 1)

    # Expand poses
    curr_pos_exp = curr_pos.repeat_interleave(4, dim=0)
    curr_quat_exp = curr_quat.repeat_interleave(4, dim=0)
    des_pos_exp = des_pos.repeat_interleave(4, dim=0)
    des_quat_exp = des_quat.repeat_interleave(4, dim=0)
    offsets_exp = offsets.repeat(env.num_envs, 1)

    # Transform points
    curr_pts, _ = combine_frame_transforms(curr_pos_exp, curr_quat_exp, offsets_exp, identity_quat)
    des_pts, _ = combine_frame_transforms(des_pos_exp, des_quat_exp, offsets_exp, identity_quat)

    # Compute mean distance
    return torch.norm(curr_pts - des_pts, p=2, dim=-1).view(-1, 4).mean(dim=-1)


def pose_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    tip_sensor_cfg: SceneEntityCfg,
    port_sensor_cfg: SceneEntityCfg,
    kp_exp_coeffs: list[tuple[float, float]] = [(50, 0.0001), (300, 0.0001)],
) -> torch.Tensor:
    """Compute exponential reward based on 4 frame keypoints."""
    # Compute distances (reusing logic from pose_error for brevity in this minimal implementation)
    # Get sensors
    tip_sensor: FrameTransformer = env.scene.sensors[tip_sensor_cfg.name]
    port_sensor: FrameTransformer = env.scene.sensors[port_sensor_cfg.name]

    # Get active target index
    command_term = env.command_manager.get_term(command_name)
    target_idx = command_term.targets_idx
    env_ids = torch.arange(env.num_envs, device=env.device)

    # Get world poses
    curr_pos = tip_sensor.data.target_pos_w[:, 0]
    curr_quat = tip_sensor.data.target_quat_w[:, 0]
    des_pos = port_sensor.data.target_pos_w[env_ids, target_idx]
    des_quat = port_sensor.data.target_quat_w[env_ids, target_idx]

    # Points logic
    offsets = torch.tensor(
        [[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [0.0, 0.15, 0.0], [0.0, 0.0, 0.15]],
        device=env.device,
    )
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs * 4, 1)

    curr_pos_exp = curr_pos.repeat_interleave(4, dim=0)
    curr_quat_exp = curr_quat.repeat_interleave(4, dim=0)
    des_pos_exp = des_pos.repeat_interleave(4, dim=0)
    des_quat_exp = des_quat.repeat_interleave(4, dim=0)
    offsets_exp = offsets.repeat(env.num_envs, 1)

    curr_pts, _ = combine_frame_transforms(curr_pos_exp, curr_quat_exp, offsets_exp, identity_quat)
    des_pts, _ = combine_frame_transforms(des_pos_exp, des_quat_exp, offsets_exp, identity_quat)

    dists = torch.norm(curr_pts - des_pts, p=2, dim=-1).view(-1, 4)
    mean_dist = dists.mean(dim=-1)

    # Apply exponential reward
    reward = torch.zeros(env.num_envs, device=env.device)
    for a, b in kp_exp_coeffs:
        reward += 1.0 / (torch.exp(a * mean_dist) + b + torch.exp(-a * mean_dist))
    return reward


# ---------------------------------------------------------------------------
# Smoothness / safety penalties
# ---------------------------------------------------------------------------


def joint_torques_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize applied joint torques (L2 squared)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1
    )


def joint_acc_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint accelerations (L2 squared) for smoother motion."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_pos_limits(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joints that exceed their soft position limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def body_lin_acc_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize linear acceleration of selected bodies (encourages gentle motion)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1
    )