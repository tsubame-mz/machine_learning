import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import os


class Agent:
    def __init__(self, p_net, v_net, optim_p, optim_v, device):
        super(Agent, self).__init__()
        self.p_net = p_net
        self.v_net = v_net
        self.optim_p = optim_p
        self.optim_v = optim_v
        self.device = device

    def get_action(self, obs):
        obs = obs.long()
        with torch.no_grad():
            pi = self.p_net(obs)
            value = self.v_net(obs)
        c = Categorical(pi)
        action = c.sample()
        # Action, Value, llh(対数尤度(log likelihood))
        return action.squeeze().item(), value.item(), c.log_prob(action).item(), pi.detach().cpu().numpy()

    def get_llhs(self, obs, actions):
        with torch.no_grad():
            pi = self.p_net(obs)
            llhs = Categorical(pi).log_prob(actions)
        return llhs.detach()

    def update(self, traj, batch_size, epochs, clip_param, v_loss_c, ent_c, max_grad_norm):
        iterate = traj.iterate(batch_size, epochs)
        loss_vals = []
        for batch in iterate:
            loss_vals.append(self._update_batch(batch, clip_param, v_loss_c, ent_c, max_grad_norm))

        loss_vals = np.mean(loss_vals, axis=0)
        return loss_vals

    def save_model(self, filename, info, suffix):
        if suffix:
            filename += "_" + suffix
        filename += ".pkl"
        print("save model: {}".format(filename))
        save_data = {"state_dict_p": self.p_net.state_dict(), "state_dict_v": self.v_net.state_dict(), "info": info}
        torch.save(save_data, filename)

    def load_model(self, filename, suffix):
        if suffix:
            filename += "_" + suffix
        filename += ".pkl"
        if not os.path.exists(filename):
            return None
        print("load model: {}".format(filename))
        load_data = torch.load(filename, map_location=self.device)
        self.p_net.load_state_dict(load_data["state_dict_p"])
        self.v_net.load_state_dict(load_data["state_dict_v"])
        return load_data["info"]

    def train(self):
        self.p_net.train()
        self.v_net.train()

    def eval(self):
        self.p_net.eval()
        self.v_net.eval()

    def _update_batch(self, batch: dict, clip_param: float, v_loss_c: float, ent_c: float, max_grad_norm: float):
        obs = batch["obs"].long()
        actions = batch["actions"]
        llhs = batch["llhs"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        pol_loss, entropy = self._update_pol(obs, actions, llhs, advantages, clip_param, ent_c, max_grad_norm)
        v_loss = self._update_val(obs, returns, v_loss_c, max_grad_norm)

        return pol_loss, v_loss, entropy

    def _update_pol(self, obs, actions, llhs, advantages, clip_param, ent_c, max_grad_norm):
        # Policy
        pi = self.p_net(obs)
        c = Categorical(pi)
        new_llhs = c.log_prob(actions)
        ratio = torch.exp(new_llhs - llhs)  # pi(a|s) / pi_old(a|s)
        clip_ratio = ratio.clamp(1.0 - clip_param, 1.0 + clip_param)
        pol_loss = (torch.max(-ratio * advantages, -clip_ratio * advantages)).mean()
        # Entropy
        entropy = -ent_c * c.entropy().mean()

        total_loss = pol_loss + entropy
        self.optim_p.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.p_net.parameters(), max_grad_norm)
        self.optim_p.step()

        return pol_loss.item(), entropy.item()

    def _update_val(self, obs, returns, v_loss_c, max_grad_norm):
        # Value
        value = self.v_net(obs)
        v_loss = v_loss_c * F.smooth_l1_loss(value.squeeze(1), returns).mean()

        self.optim_v.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), max_grad_norm)
        self.optim_v.step()

        return v_loss.item()


class Discriminator:
    def __init__(self, pseudo_rew_net, shaping_val_net, optim_discrim, device):
        self.pseudo_rew_net = pseudo_rew_net
        self.shaping_val_net = shaping_val_net
        self.optim_discrim = optim_discrim
        self.device = device

    def get_pseudo_reward(self, obs):
        obs = obs.long()
        with torch.no_grad():
            reward = self.pseudo_rew_net(obs)
        return reward.item()

    def update(self, agent_traj, expert_traj, batch_size, gamma, agent):
        agent_iterate = agent_traj.iterate(batch_size, epoch=1)
        expert_iterate = expert_traj.iterate(batch_size, epoch=1)
        loss_vals = []
        for agent_batch, expert_batch in zip(agent_iterate, expert_iterate):
            loss_vals.append(self._update_batch(agent_batch, expert_batch, gamma, agent))

        loss_vals = np.mean(loss_vals, axis=0)
        return loss_vals

    def save_model(self, filename, suffix):
        if suffix:
            filename += "_" + suffix
        filename += ".pkl"
        print("save model: {}".format(filename))
        save_data = {
            "state_dict_rew": self.pseudo_rew_net.state_dict(),
            "state_dict_val": self.shaping_val_net.state_dict(),
        }
        torch.save(save_data, filename)

    def load_model(self, filename, suffix):
        if suffix:
            filename += "_" + suffix
        filename += ".pkl"
        if not os.path.exists(filename):
            return None
        print("load model: {}".format(filename))
        load_data = torch.load(filename, map_location=self.device)
        self.pseudo_rew_net.load_state_dict(load_data["state_dict_rew"])
        self.shaping_val_net.load_state_dict(load_data["state_dict_val"])

    def train(self):
        self.pseudo_rew_net.train()
        self.shaping_val_net.train()

    def eval(self):
        self.pseudo_rew_net.eval()
        self.shaping_val_net.eval()

    def _update_batch(self, agent_batch, expert_batch, gamma, agent):
        agent_loss = self._calc_airl_loss(agent_batch, gamma, agent, True)
        expert_loss = self._calc_airl_loss(expert_batch, gamma, agent, False)
        discrim_loss = agent_loss + expert_loss

        self.optim_discrim.zero_grad()
        discrim_loss.backward()
        self.optim_discrim.step()

        return agent_loss.item(), expert_loss.item()

    def _calc_airl_loss(self, batch, gamma, agent, is_agent):
        obs = batch["obs"].long()
        actions = batch["actions"]
        next_obs = batch["next_obs"].long()
        dones = batch["dones"]

        value = self.shaping_val_net(obs).squeeze(1)
        next_value = self.shaping_val_net(next_obs).squeeze(1)
        reward = self.pseudo_rew_net(obs).squeeze(1)
        energies = reward + (1 - dones) * gamma * next_value - value
        llhs = agent.get_llhs(obs, actions)
        logits = energies - llhs

        if is_agent:
            target = torch.zeros(len(logits)).to(self.device)
        else:
            target = torch.ones(len(logits)).to(self.device)
        discrim_loss = 0.5 * F.binary_cross_entropy_with_logits(logits, target)
        return discrim_loss
