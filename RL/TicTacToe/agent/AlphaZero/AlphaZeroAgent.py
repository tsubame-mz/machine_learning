from __future__ import annotations
from typing import Dict
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from agent import Agent


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


class AlphaZeroNetwork(nn.Module):
    def __init__(self):
        super(AlphaZeroNetwork, self).__init__()
        input_num = 10  # TODO
        hid_num = 32  # TODO
        out_Num = 9  # TODO

        self.common_layers = nn.Sequential(
            nn.Linear(input_num, hid_num), nn.ReLU(inplace=True), nn.Linear(hid_num, hid_num), nn.ReLU(inplace=True),
        )
        self.policy_layers = nn.Sequential(nn.Linear(hid_num, out_Num),)
        self.value_layers = nn.Sequential(nn.Linear(hid_num, 1), nn.Tanh(),)  # ターンプレイヤーが勝ち=1, 負け=-1

        self.common_layers.apply(weight_init)
        self.policy_layers.apply(weight_init)
        self.value_layers.apply(weight_init)

    def inference(self, x: torch.Tensor, mask: torch.Tensor = None):
        # print(x, mask)
        h = self.common_layers(x)
        policy = self.policy_layers(h)
        value = self.value_layers(h)
        if mask is not None:
            policy += mask.masked_fill(mask == 1, -np.inf)
        return F.softmax(policy, dim=0), value


class AlphaZeroNode:
    def __init__(self, prior, player):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0
        self.to_play = player
        self.children: Dict[int, AlphaZeroNode] = {}

    def expand(self, legal_actions, policy, child_player):
        for action in legal_actions:
            self.children[action] = AlphaZeroNode(policy[action].item(), child_player)

    def add_exploration_noise(self):
        root_dirichlet_alpha = 0.15  # TODO
        root_exploration_fraction = 0.25  # TODO
        # node.print_node()
        actions = self.children.keys()
        noise = np.random.gamma(root_dirichlet_alpha, 1, len(actions))
        frac = root_exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = (self.children[a].prior * (1 - frac)) + (n * frac)
        # node.print_node()

    def select_child(self):
        ucb = {action: self._ucb_score(child) for action, child in self.children.items()}
        max_ucb = max(ucb.items(), key=lambda x: x[1])
        action = max_ucb[0]
        # print("UCB:", ucb, max_ucb)
        return action, self.children[action]

    def select_action(self):
        visits = {action: child.visit_count for action, child in self.children.items()}
        # print("visits:", visits)
        max_visit = max(visits.items(), key=lambda x: x[1])
        action = max_visit[0]
        return action

    @property
    def expanded(self):
        return len(self.children) > 0

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def print_node(self, depth=0, action=None, print_depth=None):
        print("- Node -" + "-" * 72) if depth == 0 else None
        print("--" * depth, end="")
        print(" ", end="") if depth != 0 else None
        print(f"action:[{action}]: ", end="") if action is not None else None
        print(
            f"id[{id(self)}]/prior[{self.prior}]/visit_count[{self.visit_count}]/value_sum[{self.value_sum}]/to_play[{self.to_play}]"
        )
        if depth < print_depth:
            for action, child in self.children.items():
                child.print_node(depth + 1, action, print_depth)
        print("-" * 80) if depth == 0 else None

    def _ucb_score(self, child: AlphaZeroNode):
        pb_c_base = 19652  # TODO
        pb_c_init = 1.25  # TODO
        pb_c = np.log((self.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
        pb_c *= np.sqrt(self.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior
        value_score = child.value
        score = prior_score + value_score
        return score


class AlphaZeroAgent(Agent):
    def __init__(self):
        self.network = AlphaZeroNetwork()

    def get_action(self, env, return_root=False):
        self.network.eval()
        return self._run_mcts(env, return_root)

    def train(self, batch, optimizer):
        self.network.train()

        observations, targets = zip(*batch)
        observations = torch.from_numpy(np.array(observations)).float()
        target_values, target_policies = zip(*targets)
        target_values = torch.from_numpy(np.array(target_values)).unsqueeze(1).float()
        target_policies = torch.from_numpy(np.array(target_policies)).float()

        policy, value = self.network.inference(observations)
        p_loss = -(target_policies * policy.log()).mean()
        v_loss = F.mse_loss(value, target_values)

        optimizer.zero_grad()
        total_loss = p_loss + v_loss
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        optimizer.step()

        return p_loss.item(), v_loss.item()

    def save_model(self, filename):
        print(f"Save model: {filename}")
        save_data = {"state_dict": self.network.state_dict()}
        torch.save(save_data, filename)

    def load_model(self, filename):
        print(f"Load model: {filename}")
        if os.path.exists(filename):
            load_data = torch.load(filename)
            self.network.load_state_dict(load_data["state_dict"])
        else:
            print(f"{filename} not found")

    def _run_mcts(self, env, return_root=False):
        root = AlphaZeroNode(0, env.player)
        policy, _ = self._inference(env)
        root.expand(env.legal_actions, policy, env.opponent_player)
        root.add_exploration_noise()
        # root.print_node()

        num_simulations = 100  # TODO
        for i in range(num_simulations):
            # print(f"Simulation[{i+1}]")
            node = root
            scratch_env = copy.deepcopy(env)
            scratch_path = [node]

            while node.expanded:
                action, node = node.select_child()
                scratch_env.step(action)
                scratch_path.append(node)

            value = -self._evaluate(node, scratch_env)
            self._backup(scratch_path, value)
        action = root.select_action()
        # root.print_node()

        if return_root:
            return action, root
        return action

    def _inference(self, env):
        obs = torch.from_numpy(env.observation).float()
        mask = self._make_mask(env.legal_actions)
        # print(obs, mask)
        policy, value = self.network.inference(obs, mask)
        # print(policy, value)
        return policy.detach().numpy(), value.item()

    def _make_mask(self, legal_actions):
        mask = torch.ones(9)  # TODO
        mask[legal_actions] = 0.0
        return mask

    def _evaluate(self, node: AlphaZeroNode, env):
        if env.done:
            if env.winner == env.EMPTY:
                return 0  # 引き分け
            elif env.winner != node.to_play:
                return -1  # 負け
            else:
                return +1  # 勝ち

        policy, value = self._inference(env)
        node.expand(env.legal_actions, policy, env.opponent_player)
        return value

    def _backup(self, scratch_path, value):
        for node in reversed(scratch_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value
