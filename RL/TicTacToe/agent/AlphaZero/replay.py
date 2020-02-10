from typing import List
import numpy as np
import logging

from TicTacToe import TicTacToeEnv
from logger import setup_logger

logger = setup_logger(__name__, logging.INFO)


class GameBuffer:
    def __init__(self):
        self.observations = []
        self.players = []
        self.actions = []
        self.values = []
        self.child_visits = []
        self.winner = TicTacToeEnv.EMPTY
        self.num_actions = 9

    def append(self, obs, player, action):
        self.observations.append(obs)
        self.players.append(player)
        self.actions.append(action)

    def store_search_statistics(self, root):
        total_visit = sum([edge.visit_count for edge in root.edges])
        edge_map = {edge.action: edge.visit_count for edge in root.edges}
        child_visit = [
            edge_map[action] / total_visit if action in edge_map else 0 for action in range(self.num_actions)
        ]
        self.child_visits.append(child_visit)

    def set_winner(self, wineer):
        self.winner = wineer
        for i in range(len(self.observations)):
            if self.winner != 0:
                player = self.players[i]
                value = +1 if self.winner == player else -1
            else:
                value = 0
            self.values.append(value)

    def make_target(self, state_index: int):
        return self.values[state_index], np.array(self.child_visits[state_index])

    def print_buffer(self):
        for i in range(len(self.observations)):
            print(
                f"obs[{self.observations[i]}]/player[{self.players[i]}]/action[{self.actions[i]}]/value[{self.values[i]}]/child_visits[{self.child_visits[i]}]"
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
        game_pos = [(g, np.random.randint(len(g.observations))) for g in games]
        return [(g.observations[i], g.make_target(i)) for (g, i) in game_pos]
