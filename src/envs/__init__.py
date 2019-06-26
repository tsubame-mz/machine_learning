from gym.envs.registration import register

register(
    id="SimpleMaze-v0", entry_point="envs.simple_maze:SimpleMaze", max_episode_steps=200
)
