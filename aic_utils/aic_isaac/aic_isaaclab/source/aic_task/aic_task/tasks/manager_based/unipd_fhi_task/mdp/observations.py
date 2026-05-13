# Copyright (c) 2022-2024, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
import torch
from typing import TYPE_CHECKING


from isaaclab.managers import ManagerTermBase
from isaaclab.managers import ObservationTermCfg
from isaaclab.managers import SceneEntityCfg

from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.observations import image

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
    target_pos_w = command_term.poses_w[:, :3]
    target_quat_w = command_term.poses_w[:, 3:]
    
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
    target_pos_w = command_term.poses_w[:, :3]
    target_quat_w = command_term.poses_w[:, 3:]
    
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
    target_pos_w = command_term.poses_w[:, :3]
    target_quat_w = command_term.poses_w[:, 3:]
    
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


# ---------------------------------------------------------------------------
# Camera features
# ---------------------------------------------------------------------------

class image_resnet_features(ManagerTermBase):
    """Frozen ResNet penultimate features from camera RGB.

    Output:
        resnet18/resnet34  -> (num_envs, 512)
        resnet50/resnet101 -> (num_envs, 2048)
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.model_name = cfg.params.get("model_name", "resnet18")
        self.model_device = env.device

        self._model = self._make_resnet(self.model_name, self.model_device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
        data_type: str = "rgb",
        convert_perspective_to_orthogonal: bool = False,
        model_name: str = "resnet18",
    ) -> torch.Tensor:

        image_data = image(
            env=env,
            sensor_cfg=sensor_cfg,
            data_type=data_type,
            convert_perspective_to_orthogonal=convert_perspective_to_orthogonal,
            normalize=False,
        )

        image_device = image_data.device

        # [N, H, W, 3] -> [N, 3, H, W]
        image_proc = image_data[..., :3].to(self.model_device)
        image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.model_device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.model_device).view(1, 3, 1, 1)
        image_proc = (image_proc - mean) / std

        with torch.no_grad():
            features = self._model(image_proc)

        return features.detach().to(image_device)

    def reset(self, env_ids: torch.Tensor | None = None):
        pass

    def _make_resnet(self, model_name: str, model_device: str) -> torch.nn.Module:
        from torchvision import models

        weights_map = {
            "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
            "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
            "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
            "resnet101": models.ResNet101_Weights.IMAGENET1K_V1,
        }

        if model_name not in weights_map:
            raise ValueError(
                f"Unsupported ResNet model: {model_name}. "
                f"Available: {list(weights_map.keys())}"
            )

        model = getattr(models, model_name)(weights=weights_map[model_name])

        # Remove ImageNet classifier, keep avgpool + flatten behavior.
        model.fc = torch.nn.Identity()

        model.eval()
        model.to(model_device)

        for param in model.parameters():
            param.requires_grad_(False)

        return model


# ---------------------------------------------------------------------------
# Heuristic target
# ---------------------------------------------------------------------------


def target_port_base_heuristic(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Minimal target port pose in robot base frame — (x,y,yaw)."""
    robot = env.scene[asset_cfg.name]

    if not hasattr(env, "_target_pos_w") or not hasattr(env, "_target_quat_w"):
        env._target_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        env._target_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 4)

    pos_rel, quat_rel = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, env._target_pos_w, env._target_quat_w
    )

    _, _, yaw = euler_xyz_from_quat(quat_rel)
    return torch.cat([pos_rel[:, :2], yaw.unsqueeze(-1)], dim=-1)
