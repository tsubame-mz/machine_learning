import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np


class Buffer:
    def __init__(self):
        self.initialize()

    def initialize(self):
        self._data_map = dict(obs=[], actions=[], rewards=[], values=[], llhs=[])

    def append(self, ob, action, reward, value, llh):
        self._data_map["obs"].append(ob)
        self._data_map["actions"].append(action)
        self._data_map["rewards"].append(reward)
        self._data_map["values"].append(value)
        self._data_map["llhs"].append(llh)

    def finish_path(self, gamma, lam):
        # ndarray化
        for key in self._data_map.keys():
            self._data_map[key] = np.array(self._data_map[key], dtype=float)

        values = np.append(self.data_map["values"], 0)
        last_delta = 0
        last_rew = 0
        rewards = self._data_map["rewards"]
        reward_size = len(rewards)
        returns = np.zeros(reward_size)
        advantages = np.zeros(reward_size)
        for t in reversed(range(reward_size)):
            # Advantage
            # GAE(General Advantage Estimation)
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            last_delta = delta + (gamma * lam) * last_delta
            advantages[t] = last_delta

            # Return
            last_rew = rewards[t] + gamma * last_rew
            returns[t] = last_rew
        self._data_map["advantages"] = advantages
        self._data_map["returns"] = returns

    @property
    def data_map(self):
        return self._data_map


class Trajectory:
    def __init__(self):
        self.initialize()

    def initialize(self):
        self._data_map = dict()
        self.total_step = 0

    def append(self, local_buf):
        data_map = local_buf.data_map
        for key in data_map.keys():
            if key not in self._data_map.keys():
                self._data_map[key] = data_map[key]
            else:
                self._data_map[key] = np.concatenate([self._data_map[key], data_map[key]])

    def finish_path(self, eps, device):
        # tensor化
        for key in self._data_map.keys():
            self._data_map[key] = torch.tensor(self._data_map[key], dtype=torch.float, device=device)

        # Advantageを標準化(平均0, 分散1)
        advs = self._data_map["advantages"]
        self._data_map["advantages"] = (advs - advs.mean()) / (advs.std() + eps)

        self.total_step = len(self._data_map["rewards"])

    def iterate(self, batch_size, epoch, drop_last=False):
        sampler = BatchSampler(SubsetRandomSampler(range(self.total_step)), batch_size, drop_last=drop_last)
        for _ in range(epoch):
            for indices in sampler:
                batch = dict()
                for k in self._data_map.keys():
                    batch[k] = self._data_map[k][indices]
                yield batch

    @property
    def data_map(self):
        return self._data_map
