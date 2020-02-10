from __future__ import annotations
from typing import List
import copy
import numpy as np
import logging

from agent import Agent, RandomAgent
from TicTacToe import TicTacToeEnv
from logger import setup_logger

logger = setup_logger(__name__, logging.INFO)


class Node:
    def __init__(self, id: int, player: int):
        self.id = id
        self.player = player
        self.edges: List[Edge] = []

    def expand(self, actions: List[int], next_id: int = 1):
        for i, action in enumerate(actions):
            child = Node(next_id + i, -self.player)
            edge = Edge(self, child, action)
            self.edges.append(edge)

    def select_edge(self) -> Edge:
        for edge in self.edges:
            # 未探索のものを優先
            if edge.visit_count == 0:
                logger.debug("Unexplored edge: " + str(edge))
                return edge

        # UCBを計算
        # self.print_node()
        total_visit = sum([edge.visit_count for edge in self.edges])
        ucb_scores = [edge.ucb_score(total_visit) for edge in self.edges]
        max_idx = np.argmax(ucb_scores)
        edge = self.edges[max_idx]
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
    def __init__(self, in_node: Node, out_node: Node, action: int):
        self.id = str(in_node.id) + ":" + str(out_node.id)
        self.player = in_node.player
        self.action = action
        self.visit_count = 0
        self.value_sum = 0.0
        self.in_node = in_node
        self.out_node = out_node

    def ucb_score(self, total_visit):
        c = 1.0  # TODO
        exploitation_value = self.value_sum / self.visit_count
        exploration_value = np.sqrt(2.0 * np.log(total_visit) / self.visit_count)
        ucb_score = exploitation_value + c * exploration_value
        return ucb_score

    def __str__(self):
        return f"id[{self.id}]/player[{self.player}]/action[{self.action}]/N[{self.visit_count}]/W[{self.value_sum}]"


class MCTSAgent(Agent):
    def __init__(self):
        self.simulation_num = 1000
        self.expand_th = 5
        self.random_agent = RandomAgent()
        self.node_num = 0

    def get_action(self, env: TicTacToeEnv, return_root=False):
        return self._run_mcts(env, return_root)

    def _run_mcts(self, env: TicTacToeEnv, return_root=False):
        logger.debug(f"*** Run MCTS ***")
        root = Node(0, env.player)
        root.expand(env.legal_actions)
        self.node_num = len(root.edges) + 1
        # root.print_node()

        for i in range(self.simulation_num):
            logger.debug("#" * 80)
            logger.debug(f"### *** Simulation[{i+1}] *** ###")
            temp_env = copy.deepcopy(env)
            search_path = self._find_leaf(root, temp_env)
            value = self._playout(temp_env, search_path[-1].player)
            self._backup(search_path, value)

        logger.debug("#" * 80)
        logger.debug("### *** Simulation complete *** ###")
        # root.print_node(limit_depth=1)
        action = root.select_action()

        if return_root:
            return action, root
        return action

    def _find_leaf(self, node: Node, env: TicTacToeEnv) -> List[Edge]:
        logger.debug(f"*** Find leaf ***")
        search_path = []

        edge = None
        while node.expanded:
            edge = node.select_edge()
            env.step(edge.action)
            search_path.append(edge)
            node = edge.out_node

        if edge and (edge.visit_count >= self.expand_th) and (not env.done):
            logger.debug(f"Node expand(th=[{self.expand_th}])")
            node.expand(env.legal_actions, self.node_num)
            self.node_num += len(node.edges)
            edge = node.select_edge()
            env.step(edge.action)
            search_path.append(edge)
            node = edge.out_node

        return search_path

    def _playout(self, env: TicTacToeEnv, player: int):
        while not env.done:
            action = self.random_agent.get_action(env)
            env.step(action)
        value = 0
        if env.winner != 0:
            if env.winner == player:
                value = +1
            else:
                value = -1
        # env.render()
        logger.debug(f"Playout result: winner[{env.winner}]/player[{player}] -> value[{value}]")
        return value

    def _backup(self, search_path: List[Edge], value: int):
        logger.debug(f"*** Backup ***")
        player = search_path[-1].player
        for edge in reversed(search_path):
            edge.visit_count += 1
            edge.value_sum += value if edge.player == player else -value
            logger.debug(edge)
