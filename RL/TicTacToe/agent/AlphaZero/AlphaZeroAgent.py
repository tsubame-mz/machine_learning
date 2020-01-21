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


class AlphaZeroNode:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0
        self.to_play = None
        self.children: Dict[int, AlphaZeroNode] = {}

    @property
    def expanded(self):
        return len(self.children) > 0

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def print_node(self, depth=0, action=None):
        print("- Node -" + "-" * 72) if depth == 0 else None
        print("--" * depth, end="")
        print(" ", end="") if depth != 0 else None
        print(f"action:[{action}]: ", end="") if action is not None else None
        print(
            f"id[{id(self)}]/prior[{self.prior}]/visit_count[{self.visit_count}]/value_sum[{self.value_sum}]/to_play[{self.to_play}]"
        )
        for action, child in self.children.items():
            child.print_node(depth + 1, action)
        print("-" * 80) if depth == 0 else None


class AlphaZeroNetwork(nn.Module):
    def __init__(self):
        super(AlphaZeroNetwork, self).__init__()
        input_num = 10  # TODO
        hid_num = 16  # TODO
        out_Num = 9  # TODO
        self.policy_layers = nn.Sequential(
            nn.Linear(input_num, hid_num),
            nn.ReLU(inplace=True),
            nn.Linear(hid_num, hid_num),
            nn.ReLU(inplace=True),
            nn.Linear(hid_num, out_Num),
        )
        self.policy_layers.apply(weight_init)

        self.value_layers = nn.Sequential(
            nn.Linear(input_num, hid_num),
            nn.ReLU(inplace=True),
            nn.Linear(hid_num, hid_num),
            nn.ReLU(inplace=True),
            nn.Linear(hid_num, 1),
        )
        self.value_layers.apply(weight_init)

    def inference(self, x: torch.Tensor, mask: torch.Tensor = None):
        # print(x, mask)
        policy = self.policy_layers(x)
        value = self.value_layers(x)
        if mask is not None:
            policy += mask.masked_fill(mask == 1, -np.inf)
        return F.softmax(policy, dim=0), value


class AlphaZeroAgent(Agent):
    def __init__(self):
        self.network = AlphaZeroNetwork()

    def get_action(self, env):
        self.network.eval()
        return self._run_mcts(env)

    def train(self, batch, optimizer):
        self.network.train()
        p_loss = 0.0
        v_loss = 0.0
        # TODO: minibatch
        for board, player, (target_value, target_policy) in batch:
            obs = self._make_obs(board, player)
            policy, value = self.network.inference(obs)

            p_loss += -(torch.from_numpy(target_policy).float() * policy.log()).mean()
            v_loss += F.mse_loss(value, torch.Tensor([target_value]).float())
        p_loss /= len(batch)
        v_loss /= len(batch)

        optimizer.zero_grad()
        total_loss = p_loss + v_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
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

    def _run_mcts(self, env):
        root = AlphaZeroNode(0)
        self._evaluate(root, env)
        self._add_exploration_noise(root)

        num_simulations = 200  # TODO
        for i in range(num_simulations):
            # print(f"Simulation[{i+1}]")
            node = root
            scratch_env = copy.deepcopy(env)
            scratch_path = [node]

            while node.expanded:
                action, node = self._select_child(node)
                scratch_env.step(action)
                scratch_path.append(node)
            # scratch_env.render()
            # node.print_node()
            value = self._evaluate(node, scratch_env)
            self._backpropagate(scratch_path, value, scratch_env.player)
        # root.print_node()
        return self._select_action(env, root), root

    def _evaluate(self, node: AlphaZeroNode, env):
        obs = self._make_obs(env.board, env.player)
        mask = self._make_mask(env.legal_actions)
        policy, value = self.network.inference(obs, mask)
        # print(policy, value)

        # expand
        node.to_play = env.player
        for action in env.legal_actions:
            node.children[action] = AlphaZeroNode(policy[action].item())
        # node.print_node()

        return value.item()

    def _make_obs(self, board, player):
        obs = torch.from_numpy(np.concatenate([board, [player]])).float()
        return obs

    def _make_mask(self, legal_actions):
        mask = torch.ones(9)  # TODO
        mask[legal_actions] = 0.0
        return mask

    def _add_exploration_noise(self, node: AlphaZeroNode):
        root_dirichlet_alpha = 0.15  # TODO
        root_exploration_fraction = 0.25  # TODO
        # node.print_node()
        actions = node.children.keys()
        noise = np.random.gamma(root_dirichlet_alpha, 1, len(actions))
        frac = root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = (node.children[a].prior * (1 - frac)) + (n * frac)
        # node.print_node()

    def _select_child(self, node: AlphaZeroNode):
        ucb = {action: self._ucb_score(node, child) for action, child in node.children.items()}
        max_ucb = max(ucb.items(), key=lambda x: x[1])
        action = max_ucb[0]
        # print(ucb, max_ucb)
        return action, node.children[action]

    def _ucb_score(self, parent: AlphaZeroNode, child: AlphaZeroNode):
        pb_c_base = 19652  # TODO
        pb_c_init = 1.25  # TODO
        pb_c = np.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
        pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior
        value_score = child.value
        score = prior_score + value_score
        return score

    def _backpropagate(self, scratch_path, value, to_play):
        for node in reversed(scratch_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1

    def _select_action(self, env, root: AlphaZeroNode):
        visits = {action: child.visit_count for action, child in root.children.items()}
        # print(visits)
        max_visit = max(visits.items(), key=lambda x: x[1])
        action = max_visit[0]
        return action
