import gymnasium as gym

from .. import agents

##
# Register Gym environments.
##


gym.register(
    id="Rel-Joint-No-Ref",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rel_joint_no_ref:RelJointNoRefEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Rel-Cart-DiffIK-No-Ref",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rel_cart_diff_ik_no_ref:RelCartesianDiffIKNoRefEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Rel-Cart-OSP-No-Ref",
    entry_point=f"{__name__}.rel_cart_osp_no_ref:RelCartesianOSPEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rel_cart_osp_no_ref:RelCartesianOSPNoRefEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
