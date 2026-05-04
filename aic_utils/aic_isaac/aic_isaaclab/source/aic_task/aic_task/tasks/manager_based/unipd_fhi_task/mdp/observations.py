# Copyright (c) 2022-2024, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def contact_net_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Net contact forces (world frame) from the contact sensor, flattened for policy obs.

    Uses the current timestep net forces (no history). Body selection is via sensor_cfg.body_ids
    if set by the manager, or sensor_cfg.body_names matched against the sensor's body_names.

    Returns:
        Tensor of shape (num_envs, num_bodies * 3) in world frame (x,y,z per body).
    """
    from isaaclab.sensors import ContactSensor

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net = contact_sensor.data.net_forces_w  # (N, B, 3)
    body_ids = sensor_cfg.body_ids
    if body_ids is None or body_ids == slice(None):
        if getattr(sensor_cfg, "body_names", None) is not None:
            names = (
                [sensor_cfg.body_names]
                if isinstance(sensor_cfg.body_names, str)
                else sensor_cfg.body_names
            )
            pattern = re.compile(names[0] if len(names) == 1 else "|".join(names))
            body_ids = [
                i for i, b in enumerate(contact_sensor.body_names) if pattern.search(b)
            ]
            if body_ids:
                net = net[:, body_ids, :]
    else:
        net = net[:, body_ids, :]
    return net.reshape(env.num_envs, -1)


# ---------------------------------------------------------------------------
# Target port pose in robot base frame
# ---------------------------------------------------------------------------
def target_port_base(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "sfp_port_pose_command",
) -> torch.Tensor:
    """Minimal target port pose in robot base frame — (x,y,yaw)."""
    robot = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    target_pos_w = command_term.pose_command_w[:, :3]
    target_quat_w = command_term.pose_command_w[:, 3:]
    
    pos_rel, quat_rel = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_w, target_quat_w
    )

    _, _, yaw = euler_xyz_from_quat(quat_rel)
    return torch.cat([pos_rel[:, :2], yaw.unsqueeze(-1)], dim=-1)


def target_port_pos_base(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "sfp_port_pose_command",
) -> torch.Tensor:
    """Target port position relative to robot base frame — 3-D (x,y,z)."""
    robot = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    target_pos_w = command_term.pose_command_w[:, :3]
    target_quat_w = command_term.pose_command_w[:, 3:]
    
    pos_rel, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_w, target_quat_w
    )
    return pos_rel


def target_port_rpy_base(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "sfp_port_pose_command",
) -> torch.Tensor:
    """Target port orientation as (roll, pitch, yaw) in robot base frame — 3-D (radians)."""
    robot = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    target_pos_w = command_term.pose_command_w[:, :3]
    target_quat_w = command_term.pose_command_w[:, 3:]
    
    _, quat_rel = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_w, target_quat_w
    )
    roll, pitch, yaw = euler_xyz_from_quat(quat_rel)
    return torch.stack([roll, pitch, yaw], dim=-1)


# ---------------------------------------------------------------------------
# EE pose & velocity in robot base frame
# ---------------------------------------------------------------------------

def ee_pos_base(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """EE position relative to robot base frame — 3-D (x, y, z)."""
    asset = env.scene[asset_cfg.name]
    ee_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :]
    return quat_apply_inverse(
        asset.data.root_quat_w, ee_pos_w - asset.data.root_pos_w
    )


def ee_rpy_base(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """EE orientation as (roll, pitch, yaw) in robot base frame — 3-D (radians)."""
    asset = env.scene[asset_cfg.name]
    ee_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    q_rel = quat_mul(quat_conjugate(asset.data.root_quat_w), ee_quat_w)
    roll, pitch, yaw = euler_xyz_from_quat(q_rel)
    return torch.stack([roll, pitch, yaw], dim=-1)


def ee_lin_vel_base(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """EE linear velocity in robot base frame — 3-D."""
    asset = env.scene[asset_cfg.name]
    vel_w = asset.data.body_vel_w[:, asset_cfg.body_ids[0], :3]
    return quat_apply_inverse(asset.data.root_quat_w, vel_w)


def ee_ang_vel_base(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """EE angular velocity in robot base frame — 3-D."""
    asset = env.scene[asset_cfg.name]
    omega_w = asset.data.body_vel_w[:, asset_cfg.body_ids[0], 3:]
    return quat_apply_inverse(asset.data.root_quat_w, omega_w)
