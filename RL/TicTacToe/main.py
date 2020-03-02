import copy
import numpy as np
import torch
import gym
import gym_tictactoe  # NOQA

from agent import RandomAgent, MCTSAgent, AlphaZeroAgent


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def play_game(env, agent_b, agent_w):
    obs = env.reset()
    done = False
    while not done:
        # print("-" * 80)
        # env.render()
        player = obs["to_play"]
        if player == 0:
            action = agent_b.get_action(env, obs)
        else:
            action = agent_w.get_action(env, obs)
        obs, _, done, _ = env.step(action)
    # print("-" * 80)
    env.render()
    return obs["winner"]


def match(num, env, agent_b, agent_w):
    win_b_cnt = 0
    win_w_cnt = 0
    draw_cnt = 0
    for i in range(num):
        print(f"--- Game[{i+1}] ---")
        winner = play_game(env, agent_b, agent_w)
        print(f"Winner[{winner if winner is not None else 'None'}]")
        if winner is None:
            draw_cnt += 1
        elif winner == env.BLACK:
            win_b_cnt += 1
        else:
            win_w_cnt += 1
    return win_b_cnt, win_w_cnt, draw_cnt


def main():
    # set_seed(0)

    env = gym.make("TicTacToe-v0")
    agent_map = {"Random": RandomAgent, "MCTS": MCTSAgent, "AlphaZero": AlphaZeroAgent}
    agent_b = agent_map["MCTS"]()
    agent_w = agent_map["AlphaZero"]()

    if isinstance(agent_b, AlphaZeroAgent):
        agent_b.load_model("./pretrained/alphazero_model_support.pth")
    if isinstance(agent_w, AlphaZeroAgent):
        agent_w.load_model("./pretrained/alphazero_model_support.pth")

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
