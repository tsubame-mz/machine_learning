import copy
from torch.utils.tensorboard import SummaryWriter

from TicTacToe import TicTacToeEnv
from agent import MuZeroAgent
from agent.MuZero import GameBuffer, ReplayBuffer
from ralamb import Ralamb


def play_game(env, agent):
    env.reset()
    game = GameBuffer()
    while not env.done:
        # env.render()
        action, root = agent.get_action(env, True)
        obs = copy.deepcopy(env.observation)
        player = env.player
        env.step(action)
        game.append(obs, player, action)
        game.store_search_statistics(root)
    # env.render()
    game.set_winner(env.winner)
    # game.print_buffer()
    return game


def run_selfplay(env, agent, replay):
    for i in range(10):
        game = play_game(env, agent)
        replay.append(game)


def validate(env, agent, is_render=False):
    env.reset()
    while not env.done:
        if is_render:
            env.render()
        action, root = agent.get_action(env, True)
        if is_render:
            root.print_node(limit_depth=1)
        env.step(action)
    env.render()


def calc_win_rate(replay):
    win_b_cnt = 0
    win_w_cnt = 0
    draw_cnt = 0
    for game in replay.buffer:
        if game.winner == TicTacToeEnv.BLACK:
            win_b_cnt += 1
        elif game.winner == TicTacToeEnv.WHITE:
            win_w_cnt += 1
        else:
            draw_cnt += 1
    win_b_rate = win_b_cnt / len(replay.buffer)
    win_w_rate = win_w_cnt / len(replay.buffer)
    draw_rate = draw_cnt / len(replay.buffer)

    return win_b_rate, win_w_rate, draw_rate


def muzero(env, agent, replay, optimizer, writer):
    for i in range(1000):
        run_selfplay(env, agent, replay)
        batch = replay.sample_batch()
        p_loss, v_loss = agent.train(batch, optimizer)
        win_b_rate, win_w_rate, draw_rate = calc_win_rate(replay)
        print(
            f"{i}: Loss:P[{p_loss:.6f}]/V[{v_loss:.6f}], Win:B[{win_b_rate:.6f}]/W[{win_w_rate:.6f}], Draw[{draw_rate:.6f}]"
        )
        writer.add_scalar("MuZero/p_loss", p_loss, i + 1)
        writer.add_scalar("MuZero/v_loss", v_loss, i + 1)
        writer.add_scalar("MuZero/win_b_rate", win_b_rate, i + 1)
        writer.add_scalar("MuZero/win_w_rate", win_w_rate, i + 1)
        writer.add_scalar("MuZero/draw_rate", draw_rate, i + 1)

        if (i % 10) == 0:
            validate(env, agent)
            agent.save_model("muzero_model.pth")


def main():
    env = TicTacToeEnv()
    agent = MuZeroAgent()
    replay = ReplayBuffer(100, 32)
    optimizer = Ralamb(agent.network.parameters(), lr=1e-2, weight_decay=1e-4)

    agent.load_model("muzero_model.pth")
    try:
        writer = SummaryWriter("./logs/MuZero")
        muzero(env, agent, replay, optimizer, writer)
    except KeyboardInterrupt:
        print("Keyboard interrupt")
    print("Train complete")
    validate(env, agent, True)
    agent.save_model("muzero_model.pth")


if __name__ == "__main__":
    main()
