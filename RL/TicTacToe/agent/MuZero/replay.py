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
        self.rewards = []
        self.child_visits = []
        self.winner = TicTacToeEnv.EMPTY
        self.num_actions = 9

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
        self.child_visits.append(child_visit)

    def set_winner(self, wineer):
        self.winner = wineer
        discount = 0.95  # TODO
        for i in range(len(self.observations)):
            if self.winner != 0:
                player = self.players[i]
                value = +1 if self.winner == player else -1
                value *= discount ** (len(self.observations) - (i + 1))
            else:
                value = 0
            self.values[i] = value

        if wineer != 0:
            if self.players[-1] == wineer:
                self.rewards[-1] = 1.0
            else:
                self.rewards[-1] = -1.0

    def make_target(self, state_index: int, unroll_steps: int):
        target_values = self.values[state_index : state_index + unroll_steps + 1]
        target_policies = self.child_visits[state_index : state_index + unroll_steps + 1]
        target_rewards = [0.0] + self.rewards[state_index : state_index + unroll_steps]
        if len(target_values) < (unroll_steps + 1):
            target_values.append(0.0)
            target_policies.append([1.0 / self.num_actions for _ in range(self.num_actions)])
        assert len(target_values) == len(target_policies) == len(target_rewards)
        return target_values, np.array(target_policies), target_rewards

    def print_buffer(self):
        for i in range(len(self.observations)):
            print(
                f"obs[{self.observations[i]}]/player[{self.players[i]}]/action[{self.actions[i]}]/value[{self.values[i]:.4f}]/reward[{self.rewards[i]}]/policy[{self.child_visits[i]}]"
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
        unroll_steps = 5  # TODO
        games = np.random.choice(self.buffer, self.batch_size)
        game_pos = [(g, np.random.randint(len(g.observations))) for g in games]
        return [
            (g.observations[i], g.actions[i : i + unroll_steps], g.make_target(i, unroll_steps)) for (g, i) in game_pos
        ]
