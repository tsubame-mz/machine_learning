import copy
import logging

import gym
from torch.utils.tensorboard import SummaryWriter

import gym_tictactoe  # NOQA
import radam
from agent import AlphaZeroAgent
from agent.AlphaZero import GameBuffer, ReplayBuffer
from logger import setup_logger

logger = setup_logger(__name__, logging.INFO)


def play_game(env, agent):
    obs = env.reset()
    done = False
    game = GameBuffer(copy.deepcopy(obs["board"]), obs["to_play"])
    while not done:
        # env.render()
        action, root = agent.get_action(env, obs, True)
        obs, _, done, _ = env.step(action)
        game.append(copy.deepcopy(obs["board"]), obs["to_play"], action)
        game.store_search_statistics(root)
    # env.render()
    game.set_winner(obs["winner"])
    # game.print_buffer()
    return game


def run_selfplay(env, agent, replay):
    for i in range(10):
        game = play_game(env, agent)
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


def alphazero(env, agent, replay, optimizer, writer, model_file, summary_tag):
    for i in range(10000):
        run_selfplay(env, agent, replay)
        p_loss, v_loss = agent.train(replay.sample_batch(), optimizer)
        win_b_rate, win_w_rate, draw_rate = calc_win_rate(replay)
        logger.info(
            f"{i}: Loss:P[{p_loss:.6f}]/V[{v_loss:.6f}], Win:B[{win_b_rate:.6f}]/W[{win_w_rate:.6f}], Draw[{draw_rate:.6f}]"
        )
        writer.add_scalar(summary_tag + "/p_loss", p_loss, i + 1)
        writer.add_scalar(summary_tag + "/v_loss", v_loss, i + 1)
        writer.add_scalar(summary_tag + "/win_b_rate", win_b_rate, i + 1)
        writer.add_scalar(summary_tag + "/win_w_rate", win_w_rate, i + 1)
        writer.add_scalar(summary_tag + "/draw_rate", draw_rate, i + 1)

        if (i > 0) and (i % 100) == 0:
            validate(env, agent)
            agent.save_model("./pretrained/" + model_file)


def main():
    replay_buffer_size = 1000
    batch_size = 128
    model_suffix = "_support"
    model_file = "alphazero_model" + model_suffix + ".pth"
    summary_tag = "AlphaZero" + model_suffix

    env = gym.make("TicTacToe-v0")
    agent = AlphaZeroAgent()
    replay = ReplayBuffer(replay_buffer_size, batch_size)
    optimizer = radam.RAdam(agent.network.parameters(), lr=1e-2, weight_decay=1e-6)

    agent.load_model("./pretrained/" + model_file)
    try:
        writer = SummaryWriter("./logs/" + summary_tag)
        alphazero(env, agent, replay, optimizer, writer, model_file, summary_tag)
    except KeyboardInterrupt:
        print("Keyboard interrupt")
    print("Train complete")
    validate(env, agent, True)
    agent.save_model("./pretrained/" + model_file)


if __name__ == "__main__":
    main()
