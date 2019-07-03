import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import gym


class PolNet(nn.Module):
    def __init__(self, in_features, out_features, h1, h2):
        super(PolNet, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_features, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, out_features),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.layer(x)


class ValNet(nn.Module):
    def __init__(self, in_features, h1, h2):
        super(ValNet, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_features, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        return self.layer(x).squeeze()


class ActorCritic(nn.Module):
    def __init__(self, in_features, out_features, h1, h2):
        super(ActorCritic, self).__init__()
        self.pol_net = PolNet(in_features, out_features, h1, h2)
        self.v_net = ValNet(in_features, h1, h2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def sampling_action(self, state):
        pi = self.pol_net(state)
        action = Categorical(pi).sample()
        return action.squeeze(0).item()

    def get_value(self, state):
        return self.v_net(state)

    def llh(self, state, action):
        # pi(a|s) : 対数尤度
        pi = self.pol_net(state)
        c = Categorical(pi)
        return c.log_prob(action), c.entropy()


class Memory:
    def __init__(self):
        self.initialize()

    def initialize(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_ps = []
        self.returns = []
        self.advantages = []

    def add(self, state, action, reward, value, log_p):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_ps.append(log_p)

    def calculate_returns(self, gamma=0.99, lam=0.97):
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

    def get_batch(self, eps=1e-6):
        # Advantageを中心化
        advantages_var = torch.FloatTensor(self.advantages)
        advantages_var = (advantages_var - advantages_var.mean()) / (advantages_var.std() + eps)
        batch = [
            torch.FloatTensor(self.states),
            torch.FloatTensor(self.actions),
            torch.FloatTensor(self.log_ps),
            torch.FloatTensor(self.returns),
            advantages_var,
        ]
        self.initialize()
        return batch


def update(memory, actor_critic, optim_pol, optim_val, clip_ratio=0.2, target_kl=0.01, beta=1e-3):
    actor_critic.train()  # 学習モード
    states_var, actions_var, log_ps_var, returns_var, advantages_var = memory.get_batch()

    train_pi_iters = 80
    for _ in range(train_pi_iters):
        new_log_ps, entropy = actor_critic.llh(states_var, actions_var)
        ratio = (new_log_ps - log_ps_var).exp()  # pi(a|s) / pi_old(a|s)
        min_adv = torch.where(advantages_var > 0, (1 + clip_ratio) * advantages_var, (1 - clip_ratio) * advantages_var)
        pi_loss = -(torch.min(ratio * advantages_var, min_adv)).mean() - (beta * entropy.mean())
        optim_pol.zero_grad()
        pi_loss.backward()
        optim_pol.step()

        kl = (new_log_ps - log_ps_var).mean()  # KL-divergence
        if kl > 1.5 * target_kl:
            # print("Early stopping")
            break

    train_v_iters = 80
    for _ in range(train_v_iters):
        values = actor_critic.get_value(states_var)
        v_loss = (returns_var - values).pow(2).mean()
        optim_val.zero_grad()
        v_loss.backward()
        optim_val.step()

    return pi_loss.item(), v_loss.item()


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
    env = gym.make("CartPole-v0")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    actor_critic = ActorCritic(obs_dim, act_dim, 32, 32)
    print(actor_critic)
    optim_pol = torch.optim.Adam(actor_critic.pol_net.parameters(), lr=3e-4)
    optim_val = torch.optim.Adam(actor_critic.v_net.parameters(), lr=1e-4)

    memory = Memory()

    max_episodes = 500
    max_steps = 200
    for episode in range(max_episodes):
        actor_critic.eval()  # 評価モード
        state = env.reset()
        memory.initialize()

        for step in range(max_steps):
            state_var = torch.from_numpy(state).reshape(1, -1).float()  # バッチ形式のテンソル化
            action = actor_critic.sampling_action(state_var)
            next_state, reward, done, _ = env.step(action)

            reward = reward_modify(step, done)  # 報酬を手動調整

            value = actor_critic.get_value(state_var).item()
            log_p, _ = actor_critic.llh(state_var, torch.tensor(action))
            memory.add(state, action, reward, value, log_p.item())

            if done:
                memory.calculate_returns()
                break

            state = next_state

        pi_loss, v_loss = update(memory, actor_critic, optim_pol, optim_val)
        print("Episode[{:3d}], Step[{:3d}], Loss(P/L)[{:.6f} / {:.6f}]".format(episode + 1, step + 1, pi_loss, v_loss))


if __name__ == "__main__":
    main()
