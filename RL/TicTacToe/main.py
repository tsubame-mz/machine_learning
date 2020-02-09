import numpy as np
import torch

from TicTacToe import TicTacToeEnv
from agent import RandomAgent, MCTSAgent, AlphaZeroAgent


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


def match(num, env, agent_b, agent_w):
    win_b_cnt = 0
    win_w_cnt = 0
    draw_cnt = 0
    for i in range(num):
        print(f"Game[{i+1}]")
        winner = play_game(env, agent_b, agent_w)
        if winner == env.BLACK:
            win_b_cnt += 1
        elif winner == env.WHITE:
            win_w_cnt += 1
        else:
            draw_cnt += 1
    return win_b_cnt, win_w_cnt, draw_cnt


def main():
    # seed = 0
    # set_seed(seed)

    env = TicTacToeEnv()
    agent_map = {"Random": RandomAgent, "MCTS": MCTSAgent, "AlphaZero": AlphaZeroAgent}
    agent_b = agent_map["MCTS"]()
    agent_w = agent_map["MCTS"]()

    if isinstance(agent_b, AlphaZeroAgent):
        agent_b.load_model("alphazero_model.pth")
    if isinstance(agent_w, AlphaZeroAgent):
        agent_w.load_model("alphazero_model.pth")

    win_agent_b = 0
    win_agent_w = 0
    draw_cnt_total = 0

    win_b_cnt, win_w_cnt, draw_cnt = match(50, env, agent_b, agent_w)
    win_agent_b += win_b_cnt
    win_agent_w += win_w_cnt
    draw_cnt_total += draw_cnt

    win_b_cnt, win_w_cnt, draw_cnt = match(50, env, agent_w, agent_b)  # 手番反転
    win_agent_b += win_w_cnt
    win_agent_w += win_b_cnt
    draw_cnt_total += draw_cnt

    print(f"Win: B[{win_agent_b}]/W[{win_agent_w}], Draw[{draw_cnt_total}]")


if __name__ == "__main__":
    main()
