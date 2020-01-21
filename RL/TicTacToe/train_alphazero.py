import copy
from typing import List
import numpy as np
import torch

from TicTacToe import TicTacToeEnv
from agent import AlphaZeroAgent
from ralamb import Ralamb


class GameBuffer:
    def __init__(self):
        self.boards = []
        self.players = []
        self.actions = []
        self.child_visits = []
        self.winner = TicTacToeEnv.EMPTY
        self.num_actions = 9

    def append(self, board, player, action):
        self.boards.append(board)
        self.players.append(player)
        self.actions.append(action)

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.values())
        child_visit = [
            root.children[action].visit_count / sum_visits if action in root.children else 0
            for action in range(self.num_actions)
        ]
        self.child_visits.append(child_visit)

    def set_winner(self, wineer):
        self.winner = wineer

    def make_target(self, state_index: int):
        player = self.players[state_index]
        value = 0
        if self.winner != 0:
            value = +1 if self.winner == player else -1
        return value, np.array(self.child_visits[state_index])

    def print_buffer(self):
        for i in range(len(self.boards)):
            print(
                f"board[{self.boards[i]}]/player[{self.players[i]}]/action[{self.actions[i]}]/child_visits[{self.child_visits[i]}]"
            )
        print(f"winner[{self.winner}]")


class ReplayBuffer:
    def __init__(self, window_size: int, batch_size: int):
        self.window_size = window_size
        self.batch_size = batch_size
        self.buffer: List[GameBuffer] = []

    def append(self, game: GameBuffer):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self):
        games = np.random.choice(self.buffer, self.batch_size)
        game_pos = [(g, np.random.randint(len(g.boards))) for g in games]
        return [(g.boards[i], g.players[i], g.make_target(i)) for (g, i) in game_pos]


def play_game(env, agent):
    env.reset()
    game = GameBuffer()
    while not env.done:
        # env.render()
        action, root = agent.get_action(env)
        # root.print_node()
        board = copy.deepcopy(env.board)
        player = env.player
        env.step(action)
        game.append(board, player, action)
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
        action, root = agent.get_action(env)
        if is_render:
            root.print_node()
        env.step(action)
    env.render()


def alphazero(env, agent, replay, optimizer):
    for i in range(1000):
        run_selfplay(env, agent, replay)
        p_loss, v_loss = agent.train(replay.sample_batch(), optimizer)
        print(f"{i}: Loss:P[{p_loss}]/V[{v_loss}]")

        if (i % 10) == 0:
            validate(env, agent)
            agent.load_model("alphazero_model.pth")


def main():
    env = TicTacToeEnv()
    agent = AlphaZeroAgent()
    replay = ReplayBuffer(100, 32)
    optimizer = Ralamb(agent.network.parameters(), lr=1e-3, weight_decay=1e-6)

    agent.load_model("alphazero_model.pth")
    try:
        alphazero(env, agent, replay, optimizer)
    except KeyboardInterrupt:
        print("Keyboard interrupt")
    print("Train complete")
    validate(env, agent, True)
    agent.save_model("alphazero_model.pth")


if __name__ == "__main__":
    main()
