from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from ..aic_task_base_env_cfg import AICTaskSceneCfg
from .rel_cart_residual import RelCartesianOSPResidualEnvCfg, RelCartesianOSPResidualEnv
from .. import mdp


_cam_spawn = sim_utils.PinholeCameraCfg(
    focal_length=22.48,
    focus_distance=0.0,
    horizontal_aperture=20.955,
    vertical_aperture=18.627,
    clipping_range=(0.07, 20.0),
)

@configclass
class AICTaskSceneRGBCfg(AICTaskSceneCfg):
    
    center_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/aic_unified_robot/center_camera_optical/center_camera",
        spawn=_cam_spawn,
        height=384,
        width=384,
        data_types=["rgb"],
        update_period=1.0 / 20.0, # 20 FPS 
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP: robot state, ee pose, pose command."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy: joint state, ee pose, pose command."""

        # Minimal target port position and orientation (x, y, yaw = 3 dims)
        port_target = ObsTerm(
            func=mdp.target_port_base_heuristic,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # End effector position, orientation, linear velocity, angular velocity (12 dims)
        ee_pos = ObsTerm(
            func=mdp.ee_pos_base,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_tcp")},
        )
        ee_rpy = ObsTerm(
            func=mdp.ee_rpy_base,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_tcp")},
        )
        ee_lin_vel = ObsTerm(
            func=mdp.ee_lin_vel_base,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_tcp")},
        )
        ee_ang_vel = ObsTerm(
            func=mdp.ee_ang_vel_base,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_tcp")},
        )

        # Camera observations
        center_rgb = ObsTerm(
            func=mdp.image_resnet_features,
            params={
                "sensor_cfg": SceneEntityCfg("center_camera"),
                "data_type": "rgb",
                "model_name": "resnet18",
            },
        )

        # Last action (6 dims)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True # Total obs dim = 3 + 12 + (1007 * 3) + 6 = 3027

    # observation groups
    policy: PolicyCfg = PolicyCfg()


##
# Env definition
##


@configclass
class RelCartesianOSPResidualRGBEnvCfg(RelCartesianOSPResidualEnvCfg):
    """Configuration for Residual RL Insertion environment with images."""

    # MDP settings
    scene: AICTaskSceneRGBCfg = AICTaskSceneRGBCfg(num_envs=200, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # 240 Hz sim / render every 12 sim steps = 20 Hz render cadence
        self.sim.render_interval = 12

        # Balanced rendering for faster training
        self.sim.render = sim_utils.RenderCfg(
            rendering_mode="balanced",
            antialiasing_mode="FXAA",
            samples_per_pixel=6,
            enable_dl_denoiser=True,
            enable_reflections=False,
            enable_translucency=False,
            enable_global_illumination=True,
            enable_shadows=True,
            enable_ambient_occlusion=True,
        )

        # Phtorealistic rendering
        # self.sim.render = sim_utils.RenderCfg(
        #     rendering_mode="quality",
        #     antialiasing_mode="DLAA",
        #     samples_per_pixel=32,
        #     enable_dl_denoiser=True,
        #     enable_reflections=True,
        #     enable_translucency=True,
        #     enable_global_illumination=True,
        #     enable_shadows=True,
        #     enable_ambient_occlusion=True,
        # )

class RelCartesianOSPResidualRGBEnv(RelCartesianOSPResidualEnv):
    """RL env that combines a heuristic base command with policy residual corrections and images."""

    cfg: RelCartesianOSPResidualRGBEnvCfg

    def __init__(self, cfg: RelCartesianOSPResidualRGBEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)