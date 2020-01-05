from typing import List, Tuple
import numpy as np


class GameBuffer:
    def __init__(self):
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.policy: List[List[float]] = []

    def append(self, obs: np.ndarray, action: int, reward: float, values: float, policy: List[float]):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(values)
        self.policy.append(policy)

    def print_buffer(self):
        for i, (obs, action, reward, value, policy) in enumerate(
            zip(self.observations, self.actions, self.rewards, self.values, self.policy)
        ):
            print(f"[{i+1:3d}]: obs[{obs}]/action[{action}]/reward[{reward:.3f}]/value[{value:.3f}], policy[{policy}]")

    def sample_target(self, num_unroll_steps: int, td_steps: int, discount: float):
        pos = np.random.randint(0, len(self.rewards) - num_unroll_steps)
        return (
            self.observations[pos],
            self.actions[pos : pos + num_unroll_steps],
            self._make_target(pos, num_unroll_steps, td_steps, discount),
        )

    def _make_target(
        self, pos: int, num_unroll_steps: int, td_steps: int, discount: float
    ) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        target_values = []
        target_rewards = []
        target_policies = []
        for current_pos in range(pos, pos + num_unroll_steps + 1):
            bootstrap_index = current_pos + td_steps
            if bootstrap_index < len(self.values):
                value = self.values[bootstrap_index] * (discount ** td_steps)
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_pos:bootstrap_index]):
                value += reward * (discount ** i)

            if current_pos < len(self.rewards):
                # (割引報酬和(Value), 報酬(Reward), 方策(Policy))
                target_values.append([value])
                target_rewards.append([self.rewards[current_pos]])
                target_policies.append(self.policy[current_pos])
            else:
                # 終了状態
                target_values.append([0])
                target_rewards.append([0])
                target_policies.append([])
        return target_values, target_rewards, target_policies


class ReplayBuffer:
    def __init__(self, window_size: int, batch_size: int):
        self.window_size = window_size
        self.games: List[GameBuffer] = []
        self.batch_size = batch_size

    def append(self, game: GameBuffer):
        if len(self.games) > self.window_size:
            self.games.pop(0)
        self.games.append(game)

    def sample_batch(self, num_unroll_steps: int, td_steps: int, discount: float):
        batch = []
        games = np.random.choice(self.games, self.batch_size)
        batch = [game.sample_target(num_unroll_steps, td_steps, discount) for game in games]
        return batch
