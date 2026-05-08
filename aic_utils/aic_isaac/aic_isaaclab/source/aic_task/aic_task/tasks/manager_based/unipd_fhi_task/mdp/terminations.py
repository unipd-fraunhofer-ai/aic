# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def failed_insertion(
    env: ManagerBasedRLEnv,
    command_name: str,
    tip_sensor_cfg: SceneEntityCfg,
    port_sensor_cfg: SceneEntityCfg,
    card_max_xy_offset: float = 0.05,
    below_port_offset: float = 0.03,
    port_max_xy_offset: float = 0.01,
) -> torch.Tensor:
    # Get sensors
    tip_sensor: FrameTransformer = env.scene.sensors[tip_sensor_cfg.name]
    port_sensor: FrameTransformer = env.scene.sensors[port_sensor_cfg.name]

    # Get active target index from command generator
    command_term = env.command_manager.get_term(command_name)
    target_idx = command_term.targets_idx

    # Get world poses
    tip_pos_w = tip_sensor.data.target_pos_w[:, 0]
    port_pos_w = port_sensor.data.target_pos_w[torch.arange(env.num_envs), target_idx]

    # Out of insertion workspace termination
    card_xy_dist_exceeded = torch.norm(tip_pos_w[:, :2] - port_pos_w[:, :2], dim=1) > card_max_xy_offset
    
    below_port_entrance = tip_pos_w[:, 2] < (port_pos_w[:, 2] + below_port_offset)
    port_xy_dist_exceeded = torch.norm(tip_pos_w[:, :2] - port_pos_w[:, :2], dim=1) > port_max_xy_offset
    port_failure = torch.logical_and(below_port_entrance, port_xy_dist_exceeded)
    
    return torch.logical_or(card_xy_dist_exceeded, port_failure)