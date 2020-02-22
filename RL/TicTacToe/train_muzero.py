import copy
import torch
from torch.utils.tensorboard import SummaryWriter

from TicTacToe import TicTacToeEnv
from agent import MuZeroAgent
from agent.MuZero import GameBuffer, ReplayBuffer
import radam


def play_game(env, agent):
    env.reset()
    game = GameBuffer(copy.deepcopy(env.observation), env.player, agent.discount)
    # game.print_buffer()
    while not env.done:
        # env.render()
        action, root = agent.get_action(env, True)
        env.step(action)
        game.append(copy.deepcopy(env.observation), env.player, action)
        game.store_search_statistics(root)
    # env.render()
    # game.print_buffer()
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
    replay_size = len(replay.buffer)
    win_b_rate = win_b_cnt / replay_size
    win_w_rate = win_w_cnt / replay_size
    draw_rate = draw_cnt / replay_size

    return win_b_rate, win_w_rate, draw_rate


def muzero(env, agent, replay, optimizer, writer):
    for i in range(1000000):
        run_selfplay(env, agent, replay)
        batch = replay.sample_batch()
        total_loss, p_loss, v_loss, r_loss = agent.train(batch, optimizer)
        win_b_rate, win_w_rate, draw_rate = calc_win_rate(replay)
        print(
            f"{i}: Loss:Total[{total_loss:.6f}]/P[{p_loss:.6f}]/V[{v_loss:.6f}]/R[{r_loss:.6f}], Win:B[{win_b_rate:.6f}]/W[{win_w_rate:.6f}], Draw[{draw_rate:.6f}]"
        )
        writer.add_scalar("MuZero/Loss/1.Total loss", total_loss, i)
        writer.add_scalar("MuZero/Loss/2.Policy loss", p_loss, i)
        writer.add_scalar("MuZero/Loss/3.Value loss", v_loss, i)
        writer.add_scalar("MuZero/Loss/4.Reward loss", r_loss, i)
        writer.add_scalar("MuZero/Rate/1.Black win rate", win_b_rate, i)
        writer.add_scalar("MuZero/Rate/2.White win rate", win_w_rate, i)
        writer.add_scalar("MuZero/Rate/3.Draw rate", draw_rate, i)

        if (i > 0) and (i % 100) == 0:
            validate(env, agent)
            agent.save_model("muzero_model.pth")


def main():
    discount = 0.995
    unroll_steps = 5
    replay_buffer_size = 1000
    batch_size = 128

    env = TicTacToeEnv()
    agent = MuZeroAgent(discount=discount)
    replay = ReplayBuffer(replay_buffer_size, batch_size, unroll_steps)
    # optimizer = torch.optim.SGD(agent.network.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-6, nesterov=True)
    optimizer = radam.RAdam(agent.network.parameters(), lr=1e-2, weight_decay=1e-6)

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
