import copy
import logging

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import gym_tictactoe  # NOQA
import radam  # NOQA
from agent import AlphaZeroAgent
from agent.AlphaZero import GameBuffer, ReplayBuffer, AlphaZeroConfig
from logger import setup_logger

logger = setup_logger(__name__, logging.INFO)


def play_game(env, agent, config: AlphaZeroConfig):
    obs = env.reset()
    done = False
    game = GameBuffer(copy.deepcopy(obs["board"]), obs["to_play"], config.action_space)
    while not done:
        # env.render()
        action, root = agent.get_action(env, obs, True)
        obs, _, done, _ = env.step(action)
        game.append(copy.deepcopy(obs["board"]), obs["to_play"], action)
        game.store_search_statistics(root)
    # env.render()
    game.set_winner(obs["winner"], config.discount, config.terminate_value)
    # game.print_buffer()
    return game


def run_selfplay(env, agent, replay, config: AlphaZeroConfig):
    for i in range(config.self_play_num):
        game = play_game(env, agent, config)
        replay.append(game)


def validate(env, agent, is_render=False):
    obs = env.reset()
    done = False
    while not done:
        if is_render:
            env.render()
        action, root = agent.get_action(env, obs, True)
        if is_render:
            root.print_node(limit_depth=1)
        obs, _, done, _ = env.step(action)
    env.render()


def calc_win_rate(replay):
    win_b_cnt = 0
    win_w_cnt = 0
    draw_cnt = 0
    for game in replay.buffer:
        if game.winner is None:
            draw_cnt += 1
        elif game.winner == 0:
            win_b_cnt += 1
        else:
            win_w_cnt += 1

    replay_size = len(replay.buffer)
    win_b_rate = win_b_cnt / replay_size
    win_w_rate = win_w_cnt / replay_size
    draw_rate = draw_cnt / replay_size

    return win_b_rate, win_w_rate, draw_rate


def alphazero(env, agent, replay, optimizer, writer, model_file, summary_tag, config: AlphaZeroConfig):
    for i in range(config.max_training_step):
        run_selfplay(env, agent, replay, config)
        p_loss, v_loss = agent.train(replay.sample_batch(), optimizer)
        win_b_rate, win_w_rate, draw_rate = calc_win_rate(replay)
        logger.info(
            f"{i}: Loss:P[{p_loss:.6f}]/V[{v_loss:.6f}], Win:B[{win_b_rate:.6f}]/W[{win_w_rate:.6f}], Draw[{draw_rate:.6f}]"
        )
        writer.add_scalar(summary_tag + "/p_loss", p_loss, i)
        writer.add_scalar(summary_tag + "/v_loss", v_loss, i)
        writer.add_scalar(summary_tag + "/win_b_rate", win_b_rate, i)
        writer.add_scalar(summary_tag + "/win_w_rate", win_w_rate, i)
        writer.add_scalar(summary_tag + "/draw_rate", draw_rate, i)

        if (i > 0) and (i % config.validate_interval) == 0:
            validate(env, agent)
            agent.save_model("./pretrained/" + model_file)


def main():
    config = AlphaZeroConfig()
    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)

    model_suffix = "_support"
    model_file = "alphazero_model" + model_suffix + ".pth"
    summary_tag = "AlphaZero" + model_suffix

    env = gym.make("TicTacToe-v0")
    env.seed(config.seed)
    agent = AlphaZeroAgent(config)
    replay = ReplayBuffer(config.replay_buffer_size, config.batch_size)
    optimizer = radam.RAdam(agent.network.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    agent.load_model("./pretrained/" + model_file)
    try:
        writer = SummaryWriter("./logs/" + summary_tag)
        alphazero(env, agent, replay, optimizer, writer, model_file, summary_tag, config)
    except KeyboardInterrupt:
        print("Keyboard interrupt")
    print("Train complete")
    validate(env, agent, True)
    agent.save_model("./pretrained/" + model_file)


if __name__ == "__main__":
    main()
