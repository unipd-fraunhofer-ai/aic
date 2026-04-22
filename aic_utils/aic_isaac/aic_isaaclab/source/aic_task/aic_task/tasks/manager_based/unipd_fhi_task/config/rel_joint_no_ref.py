from isaaclab.utils import configclass
from ..aic_task_base_env import AICTaskBaseEnv
from isaaclab.envs.mdp import RelativeJointPositionActionCfg


@configclass
class RelJointNoRefEnvCfg(AICTaskBaseEnv):
    """Environment for relative joint position control"""


    def __post_init__(self) -> None:
        super().__post_init__()

        self.actions.arm_action = RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder.*", "elbow.*", "wrist.*"],
            scale=0.025,
            use_zero_offset=True,
        )