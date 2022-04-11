from gym.envs.registration import register
register(
    id='SimpleDriving-v0',
    entry_point='simple_driving.envs:SimpleDrivingEnv',
    max_episode_steps=500
)