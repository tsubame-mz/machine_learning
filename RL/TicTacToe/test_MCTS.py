import gym
import gym_tictactoe  # NOQA
import numpy as np

# from agent import RandomAgent, MCTSAgent, AlphaZeroAgent
from agent import MCTSAgent


def set_seed(seed):
    np.random.seed(seed)


if __name__ == "__main__":
    set_seed(0)

    env = gym.make("tictactoe-v0")
    obs = env.reset()
    mcts_agent = MCTSAgent()  # type: ignore

    # actions = [0, 1, 3, 5, 8, 4]  # o
    # actions = [0, 1, 3, 5, 8, 4, 2]  # x
    actions = [8, 3, 4]  # x

    for action in actions:
        obs, _, _, _ = env.step(action)

    env.render()
    print(obs)
    action = mcts_agent.get_action(env, obs)
    print(action)
