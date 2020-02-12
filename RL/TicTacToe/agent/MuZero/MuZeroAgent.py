from __future__ import annotations
from typing import List, Optional
import numpy as np
import torch
import os
import logging

from agent import Agent
from TicTacToe import TicTacToeEnv
from logger import setup_logger
from .utils import MinMaxStats
from .network import Network

logger = setup_logger(__name__, logging.INFO)


class Node:
    def __init__(self, id: int, player: int):
        self.id = id
        self.player = player
        self.state: Optional[torch.Tensor] = None
        self.edges: List[Edge] = []

    def expand(self, actions: List[int], policy: List[float], state: torch.Tensor, next_id: int = 1):
        self.state = state
        for i, action in enumerate(actions):
            child = Node(next_id + i, -self.player)
            edge = Edge(self, child, action, policy[action])
            self.edges.append(edge)

    def add_exploration_noise(self):
        root_dirichlet_alpha = 0.15  # TODO
        root_exploration_fraction = 0.25  # TODO
        noise = np.random.gamma(root_dirichlet_alpha, 1, len(self.edges))
        frac = root_exploration_fraction
        for edge, n in zip(self.edges, noise):
            edge.prior = (edge.prior * (1 - frac)) + (n * frac)

    def select_edge(self, min_max_stats: MinMaxStats) -> Edge:
        # UCBを計算
        # self.print_node()
        total_visit = sum([edge.visit_count for edge in self.edges])
        ucb_scores = [edge.ucb_score(total_visit, min_max_stats) for edge in self.edges]
        max_idx = np.argmax(ucb_scores)
        edge = self.edges[max_idx]
        logger.debug("USB score: " + str(ucb_scores))
        logger.debug(f"Max UCB: edge[{edge}]/score[{ucb_scores[max_idx]}]")
        return edge

    def select_action(self) -> int:
        visits = [edge.visit_count for edge in self.edges]
        max_idx = np.argmax(visits)
        action = self.edges[max_idx].action
        logger.debug(f"Max action[{action}]/visits[{visits[max_idx]}]")
        return action

    @property
    def expanded(self) -> bool:
        return len(self.edges) > 0

    def print_node(self, depth=0, limit_depth: int = None, edge_info: str = None):
        if limit_depth and depth > limit_depth:
            return
        if depth == 0:
            logger.debug("- Node -" + "-" * 72)
        prefix_str = " " * depth + " - "
        info_str = prefix_str + edge_info + " : " + str(self) if edge_info else str(self)
        logger.debug(info_str)
        for i, edge in enumerate(self.edges):
            edge.out_node.print_node(depth=depth + 1, limit_depth=limit_depth, edge_info=str(edge))
        if depth == 0:
            logger.debug("-" * 80)

    def __str__(self):
        return f"id[{self.id}]/player[{self.player}]/edges[{len(self.edges)}]"


class Edge:
    def __init__(self, in_node: Node, out_node: Node, action: int, prior: float):
        self.id = str(in_node.id) + ":" + str(out_node.id)
        self.player = in_node.player
        self.action = action
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.reward = 0.0
        self.in_node = in_node
        self.out_node = out_node

    def ucb_score(self, total_visit, min_max_stats: MinMaxStats):
        pb_c_base = 19652  # TODO
        pb_c_init = 1.25  # TODO
        pb_c = np.log((total_visit + pb_c_base + 1) / pb_c_base) + pb_c_init
        pb_c *= np.sqrt(total_visit) / (self.visit_count + 1)
        prior_score = pb_c * self.prior
        value_score = min_max_stats.normalize(self.value)
        score = prior_score + value_score
        return score

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def __str__(self):
        return f"id[{self.id}]/player[{self.player}]/action[{self.action}]/N[{self.visit_count}]/W[{self.value_sum:.2f}]/P[{self.prior:.2f}]/R[{self.reward:.2f}]"


class MuZeroAgent(Agent):
    def __init__(self):
        self.simulation_num = 100
        self.network = Network()
        self.node_num = 0

    def get_action(self, env: TicTacToeEnv, return_root=False):
        self.network.eval()
        return self._run_mcts(env, return_root)

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

    def train(self, batch, optimizer):
        self.network.train()
        [print("batch", b) for b in batch]

        # observations, targets = zip(*batch)
        # observations = torch.from_numpy(np.array(observations)).float()
        # target_values, target_policies = zip(*targets)
        # target_values = torch.from_numpy(np.array(target_values)).unsqueeze(1).float()
        # target_policies = torch.from_numpy(np.array(target_policies)).float()

        # policy, value = self.network.inference(observations)
        # p_loss = -(target_policies * policy.log()).mean()
        # v_loss = F.mse_loss(value, target_values)

        # optimizer.zero_grad()
        # total_loss = p_loss + v_loss
        # total_loss.backward()
        # # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        # optimizer.step()

        # return p_loss.item(), v_loss.item()
        return None

    def _run_mcts(self, env, return_root=False):
        logger.debug(f"*** Run MCTS ***")
        root = Node(0, env.player)
        state, policy = self._initial_inference(env)
        # print(state, policy, value)
        root.expand(env.legal_actions, policy, state)
        root.add_exploration_noise()
        self.node_num = len(root.edges) + 1
        root.print_node()

        min_max_stats = MinMaxStats()
        for i in range(self.simulation_num):
            logger.debug("#" * 80)
            logger.debug(f"### *** Simulation[{i+1}] *** ###")
            search_path = self._find_leaf(root, min_max_stats)
            value = self._evaluate(search_path[-1])
            self._backup(search_path, value, min_max_stats)

        logger.debug("#" * 80)
        logger.debug("### *** Simulation complete *** ###")
        # root.print_node()
        # root.print_node(limit_depth=1)
        # print(min_max_stats.maximum, min_max_stats.minimum)
        action = root.select_action()

        if return_root:
            return action, root
        return action

    def _initial_inference(self, env):
        obs = torch.from_numpy(env.observation).float()
        mask = self._make_mask(env.legal_actions)
        # print(obs, mask)
        state, policy, _ = self.network.initial_inference(obs, mask)
        # print(state, policy, value)
        return state.detach(), policy.detach().numpy()

    def _recurrent_inference(self, state, action):
        x = torch.cat([state, torch.eye(9)[action]], dim=0)
        # print(x)
        next_state, reward, policy, value = self.network.recurrent_inference(x)
        # print(next_state, reward, policy, value)
        return next_state.detach(), reward.item(), policy.detach().numpy(), value.item()

    def _make_mask(self, legal_actions):
        mask = torch.ones(9)  # TODO
        mask[legal_actions] = 0.0
        return mask

    def _find_leaf(self, node: Node, min_max_stats: MinMaxStats) -> List[Edge]:
        logger.debug(f"*** Find leaf ***")
        search_path = []

        edge = None
        while node.expanded:
            edge = node.select_edge(min_max_stats)
            search_path.append(edge)
            node = edge.out_node

        return search_path

    def _evaluate(self, edge: Edge):
        parent = edge.in_node
        # print(parent.state, edge.action)
        next_state, reward, policy, value = self._recurrent_inference(parent.state, edge.action)
        # print(next_state, reward, policy, value)

        logger.debug(f"Node expand")
        edge.reward = reward
        edge.out_node.expand(list(range(9)), policy, next_state, self.node_num)
        self.node_num += len(edge.out_node.edges)
        return -value  # 相手から見た価値なので反転させる

    def _backup(self, search_path: List[Edge], value: float, min_max_stats: MinMaxStats):
        logger.debug(f"*** Backup[value=[{value}]] ***")
        discount = 0.95  # TODO
        player = search_path[-1].player
        for edge in reversed(search_path):
            edge.visit_count += 1
            edge.value_sum += value if edge.player == player else -value
            min_max_stats.update(edge.value)
            value = edge.reward + discount * value
            logger.debug(edge)
