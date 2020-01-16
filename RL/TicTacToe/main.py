import numpy as np
import torch

from TicTacToe import TicTacToeEnv
from agent import RandomAgent


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    seed = 0
    set_seed(seed)

    env = TicTacToeEnv()
    agent = RandomAgent()

    env.reset()
    while not env.done:
        print("-" * 80)
        env.render()
        obs = env.board
        action = agent.get_action(env)
        print("obs: ", obs)
        print("action: ", action)
        env.step(action)
        print("next obs: ", env.board)
        print("winner: ", env.winner)
        print("done: ", env.done)
    print("-" * 80)
    env.render()


if __name__ == "__main__":
    main()
