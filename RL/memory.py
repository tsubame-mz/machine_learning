from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np


class PPOBuffer:
    def __init__(self, args):
        self.gamma = args.gamma
        self.lam = args.lam
        self.initialize()

    def initialize(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_pis = []
        self.returns = []
        self.advantages = []

    def add(self, obs, action, reward, value, log_pi):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_pis.append(log_pi)

    def finish_path(self):
        values = np.append(self.values, 0)
        last_delta = 0
        last_rew = 0
        reward_size = len(self.rewards)
        self.returns = np.zeros(reward_size)
        self.advantages = np.zeros(reward_size)
        for t in reversed(range(reward_size)):
            # Advantage
            # GAE(General Advantage Estimation)
            delta = self.rewards[t] + self.gamma * values[t + 1] - values[t]
            last_delta = delta + (self.gamma * self.lam) * last_delta
            self.advantages[t] = last_delta

            # Return
            last_rew = self.rewards[t] + self.gamma * last_rew
            self.returns[t] = last_rew

    def get_all(self):
        return (self.obs, self.actions, self.rewards, self.values, self.log_pis, self.returns, self.advantages)


class RolloutStorage:
    def __init__(self, args):
        self.initialize()
        self.adv_eps = args.adv_eps

    def initialize(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_pis = []
        self.returns = []
        self.advantages = []
        self.seq_indices = []

    def append(self, local_buf):
        obs, actions, rewards, values, log_pis, returns, advantages = local_buf.get_all()

        # シーケンスの開始/終了インデックスを保持
        self.seq_indices.append([len(self.rewards), len(self.rewards) + len(rewards)])

        self.obs.extend(obs)
        self.actions.extend(actions)
        self.rewards.extend(rewards)
        self.values.extend(values)
        self.log_pis.extend(log_pis)
        self.returns.extend(returns)
        self.advantages.extend(advantages)

    def finish_path(self):
        self.to_ndarray()
        # Advantageを標準化(平均0, 分散1)
        if len(self.advantages) > 1:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + self.adv_eps)

    def to_ndarray(self):
        self.obs = np.array(self.obs)
        self.actions = np.array(self.actions)
        self.rewards = np.array(self.rewards)
        self.values = np.array(self.values)
        self.log_pis = np.array(self.log_pis)
        self.returns = np.array(self.returns)
        self.advantages = np.array(self.advantages)
        self.seq_indices = np.array(self.seq_indices)

    def get_all(self):
        return (self.obs, self.actions, self.rewards, self.values, self.log_pis, self.returns, self.advantages)

    def iterate(self, batch_size, epoch, drop_last=False):
        sampler = BatchSampler(SubsetRandomSampler(range(len(self.rewards))), batch_size, drop_last=drop_last)
        for _ in range(epoch):
            for indices in sampler:
                batch = (
                    self.obs[indices],
                    self.actions[indices],
                    self.rewards[indices],
                    self.values[indices],
                    self.log_pis[indices],
                    self.returns[indices],
                    self.advantages[indices],
                )
                yield batch

    def iterate_seq(self, batch_size, epoch):
        self.to_ndarray()
        sampler = BatchSampler(SubsetRandomSampler(range(len(self.seq_indices))), batch_size, drop_last=False)
        for _ in range(epoch):
            for indices in sampler:
                batch_obs = []
                batch_actions = []
                batch_rewards = []
                batch_values = []
                batch_log_pis = []
                batch_returns = []
                batch_advantages = []
                for seq_indices in self.seq_indices[indices]:
                    # ランダムで選ばれたシーケンスだけ取り出す
                    batch_obs.extend(self.obs[seq_indices[0] : seq_indices[1]])
                    batch_actions.extend(self.actions[seq_indices[0] : seq_indices[1]])
                    batch_rewards.extend(self.rewards[seq_indices[0] : seq_indices[1]])
                    batch_values.extend(self.values[seq_indices[0] : seq_indices[1]])
                    batch_log_pis.extend(self.log_pis[seq_indices[0] : seq_indices[1]])
                    batch_returns.extend(self.returns[seq_indices[0] : seq_indices[1]])
                    batch_advantages.extend(self.advantages[seq_indices[0] : seq_indices[1]])
                # ndarrayに変換
                batch_obs = np.array(batch_obs)
                batch_actions = np.array(batch_actions)
                batch_rewards = np.array(batch_rewards)
                batch_values = np.array(batch_values)
                batch_log_pis = np.array(batch_log_pis)
                batch_returns = np.array(batch_returns)
                batch_advantages = np.array(batch_advantages)
                batch = (
                    batch_obs,
                    batch_actions,
                    batch_rewards,
                    batch_values,
                    batch_log_pis,
                    batch_returns,
                    batch_advantages,
                )
                yield batch
