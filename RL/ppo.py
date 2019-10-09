import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
import gym
from gym.core import Wrapper
from gym.core import RewardWrapper, ObservationWrapper
from collections import deque
import argparse
import os


class EnvMonitor(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._reset_state()

    def reset(self, **kwargs):
        self._reset_state()
        return self.env.reset(**kwargs)

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        self._update(ob, rew, done, info)
        return (ob, rew, done, info)

    def _reset_state(self):
        self.rewards = []

    def _update(self, ob, rew, done, info):
        self.rewards.append(rew)
        if done:
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {"r": round(eprew, 6), "l": eplen}
            if isinstance(info, dict):
                info["episode"] = epinfo


class CartPoleRewardWrapper(RewardWrapper):
    def reset(self, **kwargs):
        self.step_cnt = 0
        self.done = False
        return super().reset(**kwargs)

    def step(self, action):
        ob, rew, self.done, info = self.env.step(action)
        self.step_cnt += 1
        return ob, self.reward(rew), self.done, info

    def reward(self, reward):
        reward = 0
        if self.done:
            if self.step_cnt >= 195:
                # 195ステップ以上立てていたらOK
                reward = +1
            else:
                # 途中で転んでいたらNG
                reward = -1
        return reward


class OnehotObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.observation_space, gym.spaces.Discrete)
        self.in_features = self.observation_space.n

    def reset(self, **kwargs):
        return super().reset(**kwargs)

    def observation(self, obs):
        return self.one_hot(obs)

    def one_hot(self, index):
        return np.eye(self.in_features)[index]


class PVNet(nn.Module):
    def __init__(self, observation_space, action_space, args):
        super(PVNet, self).__init__()

        hid_num = args.hid_num
        droprate = args.droprate

        # in
        if isinstance(observation_space, gym.spaces.Box):
            in_features = observation_space.shape[0]
        elif isinstance(observation_space, gym.spaces.Discrete):
            in_features = observation_space.n
        # out
        if isinstance(action_space, gym.spaces.Discrete):
            out_features = action_space.n
        print("in_features[{}], out_features[{}]".format(in_features, out_features))

        # 共通ネットワーク
        self.layer = nn.Sequential(
            nn.Linear(in_features, hid_num),
            nn.ReLU(inplace=True),
            nn.Dropout(p=droprate),
            nn.Linear(hid_num, hid_num),
            nn.ReLU(inplace=True),
            nn.Dropout(p=droprate),
            nn.Linear(hid_num, hid_num),
            nn.ReLU(inplace=True),
            nn.Dropout(p=droprate),
        )

        # Policy : 各行動の確率を出力
        self.policy = nn.Sequential(nn.Linear(hid_num, out_features), nn.Softmax(dim=-1))

        # Value : 状態の価値を出力
        self.value = nn.Sequential(nn.Linear(hid_num, 1))

        # ネットワークの重みを初期化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.layer(x)
        return self.policy(h), self.value(h)


class PPOAgent:
    def __init__(self, model, optimizer, device, args):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.minibatch_size = args.minibatch_size
        self.opt_epochs = args.opt_epochs
        self.clip_ratio = args.clip_ratio
        self.v_loss_c = args.v_loss_c
        self.ent_c = args.ent_c
        self.max_grad_norm = args.max_grad_norm
        self.adv_eps = args.adv_eps
        self.minibatch_size_seq = args.minibatch_size_seq
        self.use_minibatch_seq = args.use_minibatch_seq

    def get_action(self, obs):
        obs = torch.from_numpy(np.array(obs)).float().to(self.device)
        pi, value = self.model(obs)
        c = Categorical(pi)
        action = c.sample()
        # Action, Value, llh(pi(a|s)の対数)
        return action.squeeze().item(), value.item(), c.log_prob(action).item()

    def train(self, rollouts):
        if self.use_minibatch_seq:
            iterate = rollouts.iterate_seq(self.minibatch_size, self.opt_epochs)
        else:
            iterate = rollouts.iterate(self.minibatch_size, self.opt_epochs)
        loss_vals = []
        for batch in iterate:
            loss_vals.append(self.train_batch(batch))
        loss_vals = np.mean(loss_vals, axis=0)
        return loss_vals

    def train_batch(self, batch):
        obs, actions, _, _, log_pis, returns, advantages, = batch
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        log_pis = torch.FloatTensor(log_pis).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Advantageを標準化(平均0, 分散1)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.adv_eps)

        pi, value = self.model(obs)
        c = Categorical(pi)
        new_log_pi = c.log_prob(actions)
        ratio = (new_log_pi - log_pis).exp()  # pi(a|s) / pi_old(a|s)
        clip_adv = torch.where(advantages >= 0, (1 + self.clip_ratio) * advantages, (1 - self.clip_ratio) * advantages)
        pi_loss = -(torch.min(ratio * advantages, clip_adv)).mean()
        v_loss = self.v_loss_c * 0.5 * (returns - value).pow(2).mean()
        entropy = self.ent_c * c.entropy().mean()
        toral_loss = pi_loss + v_loss + entropy

        self.optimizer.zero_grad()
        toral_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return pi_loss.item(), v_loss.item(), entropy.item(), toral_loss.item()


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
            self.returns[t] = self.advantages[t] + values[t]

    def get_all(self):
        return (self.obs, self.actions, self.rewards, self.values, self.log_pis, self.returns, self.advantages)


class RolloutStorage:
    def __init__(self):
        self.initialize()

    def initialize(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_pis = []
        self.returns = []
        self.advantages = []
        self.seq_indices = []

    def append(self, local_buf: PPOBuffer):
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
        self.to_ndarray()
        return (self.obs, self.actions, self.rewards, self.values, self.log_pis, self.returns, self.advantages)

    def iterate(self, batch_size, epoch):
        self.to_ndarray()
        sampler = BatchSampler(SubsetRandomSampler(range(len(self.rewards))), batch_size, drop_last=False)
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


def main():
    parser = argparse.ArgumentParser()
    # 環境関係
    # parser.add_argument("--env", type=str, default="CartPole-v0")
    # parser.add_argument("--env", type=str, default="FrozenLake-v0")
    parser.add_argument("--env", type=str, default="Taxi-v2")
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--use_seed", type=int, default=False)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--resume", type=bool, default=True)
    parser.add_argument("--test_epochs", type=int, default=10)
    parser.add_argument("--render_env", type=bool, default=False)
    # ネットワーク関係
    parser.add_argument("--hid_num", type=int, default=128)
    parser.add_argument("--droprate", type=float, default=0.2)
    # メモリ関係
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--adv_eps", type=float, default=1e-8)
    # 学習関係
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--train_epochs", type=int, default=1000)
    parser.add_argument("--sample_episodes", type=int, default=10)
    parser.add_argument("--opt_epochs", type=int, default=3)
    parser.add_argument("--use_minibatch_seq", type=bool, default=False)
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--minibatch_size_seq", type=int, default=4)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--v_loss_c", type=float, default=0.9)
    parser.add_argument("--ent_c", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    # ログ関係
    parser.add_argument("--log", type=str, default="logs")
    args = parser.parse_args(args=[])

    # サポート対象のGPUがあれば使う
    if args.use_gpu:
        print("Check GPU available")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print("Use device: {}".format(device))

    # ログ用フォルダ生成
    if not os.path.exists(args.log):
        print("Create directory: {0}".format(args.log))
        os.mkdir(args.log)
    if not os.path.exists(os.path.join(args.log, "models")):
        print("Create directory: {0}".format(os.path.join(args.log, "models")))
        os.mkdir(os.path.join(args.log, "models"))

    # 乱数固定化
    env_seed = None
    if args.use_seed:
        print("Use random seed: {0}".format(args.seed))
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        env_seed = args.seed

    env = gym.make(args.env)
    env.seed(env_seed)
    if args.env == "CartPole-v0":
        env = CartPoleRewardWrapper(env)
    if args.env in ["FrozenLake-v0", "Taxi-v2"]:
        env = OnehotObservationWrapper(env)
    env = EnvMonitor(env)

    model = PVNet(env.observation_space, env.action_space, args)
    print(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    agent = PPOAgent(model, optimizer, device, args)
    local_buf = PPOBuffer(args)
    rollouts = RolloutStorage()

    # モデルの途中状態を読込む
    max_rew = -1e6
    last_model_filename = os.path.join(args.log, "models", "model_last.pkl")
    best_model_filename = os.path.join(args.log, "models", "model_best.pkl")
    if args.resume:
        print("Load model params")
        if os.path.exists(last_model_filename):
            # load last model
            load_data = torch.load(last_model_filename)
            model.load_state_dict(load_data["state_dict"])
            max_rew = load_data["max_rew"]
            print("Max reward: {0}".format(max_rew))
        else:
            print("Model file not found")

    print("Train start")
    try:
        episode_rewards = deque(maxlen=args.sample_episodes)
        for epoch in range(args.train_epochs):
            model.eval()  # 評価モード
            for episode in range(args.sample_episodes):
                obs = env.reset()
                done = False
                while not done:
                    action, value, log_pi = agent.get_action(obs)
                    next_obs, reward, done, info = env.step(action)
                    local_buf.add(obs, action, reward, value, log_pi)
                    obs = next_obs

                    if "episode" in info.keys():
                        eprew = info["episode"]["r"]
                        episode_rewards.append(eprew)

                local_buf.finish_path()
                rollouts.append(local_buf)
                local_buf.initialize()

            model.train()  # 学習モード
            pi_loss, v_loss, entropy, total_loss = agent.train(rollouts)
            rollouts.initialize()

            mean_rew = np.mean(episode_rewards)
            print("Epoch[{:3d}], ".format(epoch + 1), end="")
            print(
                "Loss(P/V/E/T)[{:+.6f}/{:+.6f}/{:+.6f}/{:+.6f}], ".format(pi_loss, v_loss, entropy, total_loss), end=""
            )
            reward_format = "Reward(Mean/Median/Min/Max)[{:+3.3f}/{:+3.3f}/{:+3.3f}/{:+3.3f}]"
            print(
                reward_format.format(
                    mean_rew, np.median(episode_rewards), np.min(episode_rewards), np.max(episode_rewards)
                ),
                end="",
            )
            print()

            if mean_rew > max_rew:
                print("Reward Update: Old[{:+3.3f}] -> New[{:+3.3f}]".format(max_rew, mean_rew))
                max_rew = mean_rew
                # save best model
                print("Save best model")
                save_data = {"state_dict": model.state_dict(), "max_rew": max_rew}
                torch.save(save_data, best_model_filename)

            if (args.env in ["CartPole-v0", "FrozenLake-v0"]) and (mean_rew > 0.95):
                break
    except KeyboardInterrupt:
        print("Keyboard interrupt")
    print("Train complete")

    # save last model
    print("Save last model")
    save_data = {"state_dict": model.state_dict(), "max_rew": max_rew}
    torch.save(save_data, last_model_filename)

    # test
    test_reward = []
    print("Test start")
    try:
        model.eval()  # 評価モード
        for epoch in range(args.test_epochs):
            obs = env.reset()
            done = False
            episode_reward = 0
            while not done:
                if args.render_env:
                    env.render()
                action, value, log_pi = agent.get_action(obs)
                next_obs, reward, done, info = env.step(action)
                episode_reward += reward
                obs = next_obs
            if args.render_env:
                env.render()
            episode_reward = np.mean(episode_reward)
            test_reward.append(episode_reward)
            print("Epoch[{:3d}], ".format(epoch + 1), end="")
            print("Reward[{0}]".format(episode_reward), end="")
            print()
        print("Mean reward[{0}]".format(np.mean(test_reward)))
    except KeyboardInterrupt:
        print("Keyboard interrupt")
    print("Test complete")


if __name__ == "__main__":
    main()
