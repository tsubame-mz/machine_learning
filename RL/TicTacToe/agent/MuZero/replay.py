from typing import List
import numpy as np
import logging

from TicTacToe import TicTacToeEnv
from logger import setup_logger

logger = setup_logger(__name__, logging.INFO)


class GameBuffer:
    def __init__(self, obs, player, discount):
        self.observations = [obs]
        self.players = [player]
        self.actions = []
        self.child_visits = []
        self.values = []
        self.rewards = []
        self.winner = TicTacToeEnv.EMPTY
        self.num_actions = 9
        self.discount = discount

    def append(self, obs, player, action):
        self.observations.append(obs)
        self.players.append(player)
        self.actions.append(action)
        self.values.append(0.0)
        self.rewards.append(0.0)

    def store_search_statistics(self, root):
        total_visit = sum([edge.visit_count for edge in root.edges])
        edge_map = {edge.action: edge.visit_count for edge in root.edges}
        child_visit = [
            edge_map[action] / total_visit if action in edge_map else 0 for action in range(self.num_actions)
        ]
        # print(child_visit)
        self.child_visits.append(child_visit)

    def set_winner(self, wineer):
        self.winner = wineer
        self.values.append(0.0)
        self.child_visits.append([1.0 / self.num_actions for _ in range(self.num_actions)])
        for i in range(len(self.observations)):
            if self.winner != 0:
                player = self.players[i]
                value = +1 if self.winner == player else -1
                value *= self.discount ** (len(self.observations) - (i + 1))
            else:
                value = 0
            self.values[i] = value

        if wineer != 0:
            if self.players[-1] != wineer:
                self.rewards[-1] = +1.0
            else:
                self.rewards[-1] = -1.0

    def make_target(self, state_index: int, unroll_steps: int):
        return list(
            zip(
                self.child_visits[state_index : state_index + unroll_steps + 1],
                self.values[state_index : state_index + unroll_steps + 1],
                [0.0] + self.rewards[state_index : state_index + unroll_steps],
            )
        )

    def print_buffer(self):
        print("--- Buffer ---")
        print("observations: ", self.observations)
        print("players: ", self.players)
        print("actions: ", self.actions)
        print("values: ", self.values)
        print("rewards: ", self.rewards)
        print("child_visits: ", self.child_visits)
        print("winner: ", self.winner)
        print("--------------")


class ReplayBuffer:
    def __init__(self, window_size: int, batch_size: int, unroll_steps: int):
        self.window_size = window_size
        self.batch_size = batch_size
        self.buffer: List[GameBuffer] = []
        self.unroll_steps = unroll_steps

    def append(self, game: GameBuffer):
        if len(self.buffer) >= self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self):
        games = np.random.choice(self.buffer, self.batch_size)
        game_pos = [(g, np.random.randint(len(g.observations))) for g in games]
        return [
            (g.observations[i], g.actions[i : i + self.unroll_steps], g.make_target(i, self.unroll_steps))
            for (g, i) in game_pos
        ]
