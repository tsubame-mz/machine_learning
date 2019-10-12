import torch
import numpy as np
from torch.distributions import Categorical


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
