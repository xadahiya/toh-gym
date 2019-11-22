from gym.envs.registration import register

register(
    id='toh-v0',
    entry_point='toh_gym.envs:TohEnv',
    kwargs={},
)
