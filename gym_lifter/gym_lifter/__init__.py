from gym.envs.registration import register


register(
    id='Lifter-v0',
    entry_point='gym_lifter.envs:LifterEnv',
    max_episode_steps=20000,
)

register(
    id='LifterDiscrete-v0',
    entry_point='gym_lifter.envs:DiscreteLifterEnv',
    max_episode_steps=20000,
)


register(
    id='LifterPOD-v0',
    entry_point='gym_lifter.envs:LifterPODEnv',
    max_episode_steps=20000,
)

register(
    id='LifterCAPA-v0',
    entry_point='gym_lifter.envs:LifterCAPAEnv',
    max_episode_steps=20000,
)

