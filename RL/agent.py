import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

# import math


class Agent:
    def __init__(self, p_net, v_net, optim_p, optim_v, observation_space, device):
        super(Agent, self).__init__()

        self.in_features = observation_space.n
        self.p_net = p_net
        self.v_net = v_net
        self.optim_p = optim_p
        self.optim_v = optim_v
        self.device = device
        # self.train_count = 0

    def get_action(self, obs):
        obs = obs.long()
        with torch.no_grad():
            pi = self.p_net(obs)
            value = self.v_net(obs)
        c = Categorical(pi)
        action = c.sample()
        # Action, Value, llh(対数尤度(log likelihood))
        return action.squeeze().item(), value.item(), c.log_prob(action).item(), pi.detach().cpu().numpy()

    def update(self, traj, batch_size, epochs, clip_param, v_loss_c, ent_c, max_grad_norm):
        # self.calc_exp_val(self.train_count, self.start_ent_c, self.end_ent_c, self.tau_ent_c)

        iterate = traj.iterate(batch_size, epochs)
        loss_vals = []
        for batch in iterate:
            loss_vals.append(self._update_batch(batch, clip_param, v_loss_c, ent_c, max_grad_norm))

        # self.train_count += 1
        loss_vals = np.mean(loss_vals, axis=0)
        return loss_vals

    # def calc_exp_val(self, step, max_val, min_val, tau):
    #     return (max_val - min_val) * math.exp(-step / tau) + min_val

    def save_model(self, filename, info):
        print("save_model: {}".format(filename))
        save_data = {"state_dict_p": self.p_net.state_dict(), "state_dict_v": self.v_net.state_dict(), "info": info}
        torch.save(save_data, filename)

    def load_model(self, filename):
        print("save_model: {}".format(filename))
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

