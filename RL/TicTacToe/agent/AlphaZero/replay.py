import logging
from typing import List

import numpy as np

import gym_tictactoe  # NOQA
from logger import setup_logger

logger = setup_logger(__name__, logging.INFO)


class GameBuffer:
    def __init__(self, obs, player):
        self.observations = [obs]
        self.players = [player]
        self.actions = []
        self.values = []
        self.child_visits = []
        self.winner = None
        self.num_actions = 9
        self.discount = 0.95  # TODO

    def append(self, next_obs, next_player, action):
        self.observations.append(next_obs)
        self.players.append(next_player)
        self.actions.append(action)

    def store_search_statistics(self, root):
        total_visit = sum([edge.visit_count for edge in root.edges])
        edge_map = {edge.action: edge.visit_count for edge in root.edges}
        child_visit = [
            edge_map[action] / total_visit if action in edge_map else 0 for action in range(self.num_actions)
        ]
        self.child_visits.append(child_visit)

    def set_winner(self, winner):
        self.child_visits.append([1.0 / self.num_actions for _ in range(self.num_actions)])
        self.winner = winner
        for i in range(len(self.observations)):
            if self.winner is not None:
                player = self.players[i]
                value = +10 * (self.discount ** (len(self.observations) - i - 1))
                value = value if self.winner == player else -value
            else:
                value = 0
            self.values.append(value)

    def make_target(self, state_index: int):
        return self.values[state_index], np.array(self.child_visits[state_index])

    def print_buffer(self):
        print("--- Buffer ---")
        print("observations: ", self.observations)
        print("players: ", self.players)
        print("actions: ", self.actions)
        print("values: ", self.values)
        print("child_visits: ", self.child_visits)
        print("winner: ", self.winner)
        print("--------------")


class ReplayBuffer:
    def __init__(self, window_size: int, batch_size: int):
        self.window_size = window_size
        self.batch_size = batch_size
        self.buffer: List[GameBuffer] = []

    def append(self, game: GameBuffer):
        if len(self.buffer) >= self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self):
        games = np.random.choice(self.buffer, self.batch_size)
        game_pos = [(g, np.random.randint(len(g.observations))) for g in games]
        return [(g.observations[i], g.make_target(i)) for (g, i) in game_pos]
