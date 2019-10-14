import torch
import numpy as np
from torch.distributions import Categorical
import torch.nn.functional as F
import math


class PPOAgent:
    def __init__(self, pol_net, val_net, optim_pol, optim_val, device, args):
        self.pol_net = pol_net
        self.val_net = val_net
        self.optim_pol = optim_pol
        self.optim_val = optim_val
        self.device = device
        self.minibatch_size = args.minibatch_size
        self.opt_epochs = args.opt_epochs
        self.clip_ratio = args.clip_ratio
        self.v_loss_c = args.v_loss_c
        self.start_ent_c = args.start_ent_c
        self.end_ent_c = args.end_ent_c
        self.tau_ent_c = args.tau_ent_c
        self.max_grad_norm = args.max_grad_norm
        self.adv_eps = args.adv_eps
        self.minibatch_size_seq = args.minibatch_size_seq
        self.use_minibatch_seq = args.use_minibatch_seq

        self.train_count = 0

    def get_action(self, obs):
        obs = torch.from_numpy(np.array(obs)).to(self.device)
        with torch.no_grad():
            pi = self.pol_net(obs)
            value = self.val_net(obs)
        c = Categorical(pi)
        action = c.sample()
        # Action, Value, llh(pi(a|s)の対数)
        return action.squeeze().item(), value.item(), c.log_prob(action).item(), pi.detach().cpu().numpy()

    def train(self, rollouts):
        ent_c = self.calc_exp_val(self.train_count, self.start_ent_c, self.end_ent_c, self.tau_ent_c)

        if self.use_minibatch_seq:
            iterate = rollouts.iterate_seq(self.minibatch_size, self.opt_epochs)
        else:
            iterate = rollouts.iterate(self.minibatch_size, self.opt_epochs)
        loss_vals = []
        for batch in iterate:
            loss_vals.append(self.train_batch(batch, ent_c))

        self.train_count += 1
        loss_vals = np.mean(loss_vals, axis=0)
        return loss_vals

    def train_batch(self, batch, ent_c):
        obs, actions, _, _, log_pis, returns, advantages, = batch
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        log_pis = torch.FloatTensor(log_pis).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Policy
        pi = self.pol_net(obs)
        c = Categorical(pi)
        new_log_pi = c.log_prob(actions)
        ratio = torch.exp(new_log_pi - log_pis)  # pi(a|s) / pi_old(a|s)
        clip_adv = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        pi_loss = (torch.max(-ratio * advantages, -clip_adv)).mean()
        # Entropy
        entropy = -ent_c * c.entropy().mean()
        pol_loss = pi_loss + entropy

        self.optim_pol.zero_grad()
        pol_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pol_net.parameters(), self.max_grad_norm)
        self.optim_pol.step()

        # Value
        value = self.val_net(obs)
        v_loss = self.v_loss_c * F.smooth_l1_loss(value.squeeze(1), returns).mean()
        self.optim_val.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.val_net.parameters(), self.max_grad_norm)
        self.optim_val.step()

        return pi_loss.item(), v_loss.item(), entropy.item()

    def calc_exp_val(self, step, max_val, min_val, tau):
        return (max_val - min_val) * math.exp(-step / tau) + min_val

