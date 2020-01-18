from __future__ import annotations
from typing import Dict
import copy
import numpy as np

from agent import Agent, RandomAgent


class MCTSNode:
    def __init__(self, parent: MCTSNode = None):
        self.parent = parent
        self.visit_count = 0  # W
        self.value_sum = 0  # Q
        self.children: Dict[int, MCTSNode] = {}

    def expand(self, actions):
        for action in actions:
            self.children[action] = MCTSNode(self)

    def select_child(self, ucb_sign):
        # print("select_child")
        for action, child in self.children.items():
            # 未探索の子を優先
            if child.visit_count == 0:
                return action, child
        # UCB
        # self.print_node()
        ucb = {action: self._ucb_score(child, ucb_sign) for action, child in self.children.items()}
        max_ucb = max(ucb.items(), key=lambda x: x[1])
        # print(ucb, max_ucb)
        action = max_ucb[0]
        return action, self.children[action]

    def backup(self, reward):
        self.visit_count += 1
        self.value_sum += reward
        if self.parent:
            self.parent.backup(reward)

    def _ucb_score(self, child: MCTSNode, ucb_sign):
        c = 1.0  # TODO
        exploitation_value = (ucb_sign * child.value_sum) / child.visit_count
        exploration_value = np.sqrt(2.0 * np.log(self.visit_count) / child.visit_count)
        ucb_score = exploitation_value + c * exploration_value
        return ucb_score

    @property
    def expanded(self) -> bool:
        return len(self.children) > 0

    def print_node(self, depth=0, action=None):
        print("- Node -" + "-" * 72) if depth == 0 else None
        print("--" * depth, end="")
        print(" ", end="") if depth != 0 else None
        print(f"action:[{action}]: ", end="") if action is not None else None
        print(f"id[{id(self)}]/visit_count[{self.visit_count}]/value_sum[{self.value_sum}]")
        for action, child in self.children.items():
            child.print_node(depth + 1, action)
        print("-" * 80) if depth == 0 else None


class MCTSAgent(Agent):
    def __init__(self):
        self.simulation_num = 500
        self.expand_th = 10
        self.random_agent = RandomAgent()

    def get_action(self, env):
        return self._uct_search(env)

    def _uct_search(self, env):
        root = MCTSNode()
        root.expand(env.legal_actions)
        # root.print_node()
        for i in range(self.simulation_num):
            # print(f"Simulation[{i+1}]")
            temp_env = copy.deepcopy(env)
            node = self._find_leaf(root, temp_env, env.player)
            reward = self._playout(temp_env, env.player)
            node.backup(reward)
        # root.print_node()
        visits = {action: child.visit_count for action, child in root.children.items()}
        # print(visits)
        max_visit = max(visits.items(), key=lambda x: x[1])
        action = max_visit[0]
        return action

    def _find_leaf(self, node: MCTSNode, env, root_player):
        while node.expanded:
            ucb_sign = -1.0 if env.player != root_player else 1.0
            action, node = node.select_child(ucb_sign)
            env.step(action)

        # expand
        if (node.visit_count >= self.expand_th) and (not env.done):
            node.expand(env.legal_actions)
            action = env.legal_actions[0]
            node = node.children[action]
            env.step(action)
        return node

    def _playout(self, env, player):
        while not env.done:
            action = self.random_agent.get_action(env)
            env.step(action)
        reward = env.winner * player
        # env.render()
        return reward
