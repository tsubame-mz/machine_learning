import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import gym


class PVNet(nn.Module):
    def __init__(self, in_features, out_features, h1, h2, droprate):
        super(PVNet, self).__init__()

        # 共通ネットワーク
        self.layer = nn.Sequential(
            nn.Linear(in_features, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=droprate),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=droprate),
        )

        # Policy : 各行動の確率を出力
        self.policy = nn.Sequential(nn.Linear(h2, out_features), nn.Softmax(dim=-1))

        # Value : 状態の価値を出力
        self.value = nn.Sequential(nn.Linear(h2, 1))

    def forward(self, x):
        h = self.layer(x)
        return self.policy(h), self.value(h)


class ActorCritic(nn.Module):
    def __init__(self, in_features, out_features, h1, h2, droprate):
        super(ActorCritic, self).__init__()
        self.net = PVNet(in_features, out_features, h1, h2, droprate)

        # ネットワークの重みを初期化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def eval_forward(self, state):
        # Action : 確率でサンプリング
        # Value
        # llh : pi(a|s)の対数尤度
        pi, value = self.net(state)
        c = Categorical(pi)
        action = c.sample()
        return action.squeeze().item(), value, c.log_prob(action)

    def train_forward(self, state, action):
        # llh
        # Entropy : pi * log(pi)
        # Value
        pi, value = self.net(state)
        c = Categorical(pi)
        return c.log_prob(action), c.entropy(), value


class Memory:
    def __init__(self):
        self.initialize()

    def initialize(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_pis = []
        self.returns = []
        self.advantages = []

    def add(self, state, action, reward, value, log_pi):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_pis.append(log_pi)

    def calculate_returns(self, gamma, lam):
        values = np.append(self.values, 0)
        last_return = 0
        last_delta = 0
        self.returns = np.zeros(len(self.rewards))
        self.advantages = np.zeros(len(self.rewards))
        for t in reversed(range(len(self.rewards))):
            # Return
            last_return = self.rewards[t] + gamma * last_return
            self.returns[t] = last_return

            # Advantage
            delta = self.rewards[t] + gamma * values[t + 1] - values[t]
            last_delta = delta + (gamma * lam) * last_delta
            self.advantages[t] = last_delta

    def get_batch(self, eps):
        # Advantageを中心化(平均0, 分散1)
        advantages_var = torch.FloatTensor(self.advantages)
        advantages_var = (advantages_var - advantages_var.mean()) / (advantages_var.std() + eps)
        batch = [
            torch.FloatTensor(self.states),
            torch.FloatTensor(self.actions),
            torch.FloatTensor(self.log_pis),
            torch.FloatTensor(self.returns),
            advantages_var,
        ]
        self.initialize()
        return batch


def update(memory, actor_critic, optimizer, clip_ratio, beta, eps, train_iters):
    actor_critic.train()  # 学習モード
    states_var, actions_var, log_pis_var, returns_var, advantages_var = memory.get_batch(eps)

    for _ in range(train_iters):
        new_log_pis, entropy, values = actor_critic.train_forward(states_var, actions_var)
        ratio = (new_log_pis - log_pis_var).exp()  # pi(a|s) / pi_old(a|s)
        min_adv = torch.where(advantages_var >= 0, (1 + clip_ratio) * advantages_var, (1 - clip_ratio) * advantages_var)
        pi_loss = -(torch.min(ratio * advantages_var, min_adv)).mean() - (beta * entropy.mean())
        v_loss = (returns_var - values).pow(2).mean()
        toral_loss = pi_loss + v_loss

        optimizer.zero_grad()
        toral_loss.backward()
        optimizer.step()

    return pi_loss.item(), v_loss.item(), toral_loss.item()


def reward_modify(step, done):
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
    import argparse

    parser = argparse.ArgumentParser()
    # 環境関係
    parser.add_argument("--env", type=str, default="CartPole-v0")
    parser.add_argument("--max_episodes", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=200)
    # ネットワーク関係
    parser.add_argument("--hid_num", type=int, default=64)
    parser.add_argument("--droprate", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    # メモリ関係
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("--eps", type=float, default=1e-6)
    # 学習関係
    parser.add_argument("--train_iters", type=int, default=80)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=1e-3)
    args = parser.parse_args()

    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    actor_critic = ActorCritic(obs_dim, act_dim, args.hid_num, args.hid_num, args.droprate)
    print(actor_critic)
    optimizer = torch.optim.Adam(actor_critic.net.parameters(), lr=args.lr)

    memory = Memory()

    for episode in range(args.max_episodes):
        actor_critic.eval()  # 評価モード
        state = env.reset()
        memory.initialize()

        for step in range(args.max_steps):
            state_var = torch.from_numpy(state).reshape(1, -1).float()  # バッチ形式のテンソル化
            action, value, log_pi = actor_critic.eval_forward(state_var)
            next_state, reward, done, _ = env.step(action)

            reward = reward_modify(step, done)  # 報酬を手動調整

            memory.add(state, action, reward, value.item(), log_pi.item())

            if done:
                memory.calculate_returns(args.gamma, args.lam)
                break

            state = next_state

        pi_loss, v_loss, total_loss = update(
            memory, actor_critic, optimizer, args.clip_ratio, args.beta, args.eps, args.train_iters
        )
        print(
            "Episode[{:3d}], Step[{:3d}], Loss(P/V/T)[{:+.6f} / {:+.6f} / {:+.6f}]".format(
                episode + 1, step + 1, pi_loss, v_loss, total_loss
            )
        )


if __name__ == "__main__":
    main()
