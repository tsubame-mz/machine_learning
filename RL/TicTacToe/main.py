import numpy as np
import torch

from TicTacToe import TicTacToeEnv
from agent import RandomAgent, MCTSAgent


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def play_game(env, agent_b, agent_w):
    env.reset()
    while not env.done:
        # print("-" * 80)
        # env.render()
        # obs = env.board
        if env.player == env.BLACK:
            action = agent_b.get_action(env)
        else:
            action = agent_w.get_action(env)
        # print("obs: ", obs)
        # print("action: ", action)
        env.step(action)
        # print("next obs: ", env.board)
        # print("winner: ", env.winner)
        # print("done: ", env.done)
    # print("-" * 80)
    env.render()
    return env.winner


def main():
    seed = 0
    set_seed(seed)

    env = TicTacToeEnv()
    agent_map = {"Random": RandomAgent, "MCTS": MCTSAgent}
    agent_b = agent_map["MCTS"]()
    agent_w = agent_map["MCTS"]()

    win_b_cnt = 0
    win_w_cnt = 0
    draw_cnt = 0
    for i in range(10):
        winner = play_game(env, agent_b, agent_w)
        if winner == env.BLACK:
            win_b_cnt += 1
        elif winner == env.WHITE:
            win_w_cnt += 1
        else:
            draw_cnt += 1
    print(f"Win: B[{win_b_cnt}]/W[{win_w_cnt}], Draw[{draw_cnt}]")


if __name__ == "__main__":
    main()
