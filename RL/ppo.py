import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import gym
import argparse


class PVNet(nn.Module):
    def __init__(self, in_features, out_features, hid_num, droprate):
        super(PVNet, self).__init__()

        # 共通ネットワーク
        self.layer = nn.Sequential(
            nn.Linear(in_features, hid_num),
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
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.layer(x)
        return self.policy(h), self.value(h)


class Agent:
    def __init__(self, model):
        super(Agent, self).__init__()
        self.model = model

    def get_action(self, state):
        pi, value = self.model(state)
        c = Categorical(pi)
        action = c.sample()
        # Action, Value, llh(pi(a|s)の対数)
        return action.squeeze().item(), value.item(), c.log_prob(action).item()

    def update(self, batch, optimizer, clip_ratio, v_loss_c, ent_c, target_kl, train_iters):
        state_tsr, action_tsr, log_pi_tsr, return_tsr, advantage_tsr = batch

        for _ in range(train_iters):
            pi, value = self.model(state_tsr)
            c = Categorical(pi)
            new_log_pi = c.log_prob(action_tsr)

            ratio = (new_log_pi - log_pi_tsr).exp()  # pi(a|s) / pi_old(a|s)
            clip_adv = torch.where(
                advantage_tsr >= 0, (1 + clip_ratio) * advantage_tsr, (1 - clip_ratio) * advantage_tsr
            )
            pi_loss = -(torch.min(ratio * advantage_tsr, clip_adv)).mean()
            v_loss = v_loss_c * (return_tsr - value).pow(2).mean()
            entropy = ent_c * c.entropy().mean()
            toral_loss = pi_loss + v_loss + entropy

            optimizer.zero_grad()
            toral_loss.backward()
            optimizer.step()

            # pi(a|s)が学習に使用するものから離れてきたら学習を打ち切る
            kl = (new_log_pi - log_pi_tsr).mean().item()  # KL-divergence
            if kl > 1.5 * target_kl:
                # Early stopping
                break

        return pi_loss.item(), v_loss.item(), entropy.item(), toral_loss.item()


class Memory:
    def __init__(self):
        self.initialize()

    def initialize(self):
        self.state_lst = []
        self.action_lst = []
        self.reward_lst = []
        self.value_lst = []
        self.log_pi_lst = []
        self.return_lst = []
        self.advantage_lst = []

    def add(self, state, action, reward, value, log_pi):
        self.state_lst.append(state)
        self.action_lst.append(action)
        self.reward_lst.append(reward)
        self.value_lst.append(value)
        self.log_pi_lst.append(log_pi)

    def finish_path(self, gamma, lam):
        value_lst = np.append(self.value_lst, 0)
        last_return = 0
        last_delta = 0
        reward_size = len(self.reward_lst)
        self.return_lst = np.zeros(reward_size)
        self.advantage_lst = np.zeros(reward_size)
        for t in reversed(range(reward_size)):
            # Return
            last_return = self.reward_lst[t] + gamma * last_return
            self.return_lst[t] = last_return

            # Advantage
            # GAE(generalized advantage estimation)
            delta = self.reward_lst[t] + gamma * value_lst[t + 1] - value_lst[t]
            last_delta = delta + (gamma * lam) * last_delta
            self.advantage_lst[t] = last_delta

    def get_batch(self, adv_eps):
        # Advantageを標準化(平均0, 分散1)
        advantage_tsr = torch.FloatTensor(self.advantage_lst)
        advantage_tsr = (advantage_tsr - advantage_tsr.mean()) / (advantage_tsr.std() + adv_eps)
        batch = [
            torch.FloatTensor(self.state_lst),
            torch.FloatTensor(self.action_lst),
            torch.FloatTensor(self.log_pi_lst),
            torch.FloatTensor(self.return_lst),
            advantage_tsr,
        ]
        self.initialize()
        return batch


def reward_modify(step, done):
    # only CartPole-v0
    reward = 0
    if done:
        if (step + 1) >= 195:
            # 195ステップ以上立てていたらOK
            reward = +1
        else:
            # 途中で転んでいたらNG
            reward = -1
    return reward


def main():
    parser = argparse.ArgumentParser()
    # 環境関係
    parser.add_argument("--env", type=str, default="CartPole-v0")
    parser.add_argument("--max_episodes", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=200)
    # ネットワーク関係
    parser.add_argument("--hid_num", type=int, default=64)
    parser.add_argument("--droprate", type=float, default=0.2)
    # メモリ関係
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--adv_eps", type=float, default=1e-6)
    # 学習関係
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_iters", type=int, default=80)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--v_loss_c", type=float, default=0.9)
    parser.add_argument("--ent_c", type=float, default=1e-3)
    parser.add_argument("--target_kl", type=float, default=0.01)
    args = parser.parse_args()

    env = gym.make(args.env)
    in_features = env.observation_space.shape[0]
    out_features = env.action_space.n

    model = PVNet(in_features, out_features, args.hid_num, args.droprate)
    agent = Agent(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    memory = Memory()

    for episode in range(args.max_episodes):
        model.eval()  # 評価モード
        state = env.reset()
        memory.initialize()

        for step in range(args.max_steps):
            state_tsr = torch.from_numpy(state).reshape(1, -1).float()  # バッチ形式のテンソル化
            action, value, log_pi = agent.get_action(state_tsr)
            next_state, _, done, _ = env.step(action)

            reward = reward_modify(step, done)  # 報酬を手動調整

            memory.add(state, action, reward, value, log_pi)

            if done:
                memory.finish_path(args.gamma, args.lam)
                break

            state = next_state

        model.train()  # 学習モード
        batch = memory.get_batch(args.adv_eps)
        pi_loss, v_loss, entropy, total_loss = agent.update(
            batch, optimizer, args.clip_ratio, args.v_loss_c, args.ent_c, args.target_kl, args.train_iters
        )

        print(
            "Episode[{:3d}], Step[{:3d}], Loss(P/V/E/T)[{:+.6f}/{:+.6f}/{:+.6f}/{:+.6f}]".format(
                episode + 1, step + 1, pi_loss, v_loss, entropy, total_loss
            )
        )


if __name__ == "__main__":
    main()
