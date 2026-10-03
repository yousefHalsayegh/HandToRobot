import gymnasium as gym

gym.register(
    id="SOArm-PickPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:SOArmPickPlaceEnvCfg"},
    disable_env_checker=True,
)
