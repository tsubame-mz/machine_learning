from typing import List, Tuple
import numpy as np
import torch

from node import Node
from network import Network
from utils import MinMaxStats


class MCTS:
    def __init__(
        self,
        dirichlet_alpha: float,
        exploration_fraction: float,
        pb_c_base: float,
        pb_c_init: float,
        discount: float,
        num_simulations: int,
    ):
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.discount = discount
        self.num_simulations = num_simulations

    def run_mcts(self, obs: np.ndarray, network: Network) -> Node:
        # ルートノードを展開
        root = Node(0)
        state, policy, value = network.initial_inference(obs)
        root.expand_node(0, state.squeeze().detach().numpy(), 0, policy.squeeze().detach().numpy())
        root.add_exploration_noise(self.dirichlet_alpha, self.exploration_fraction)  # if train:

        min_max_stats = MinMaxStats(None)
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            while node.expanded:
                # 展開されていない子まで辿る
                action, node = self._select_child(node, min_max_stats)
                search_path.append(node)

            # 子ノードを展開
            parent = search_path[-2]
            next_state, reward, policy, value = network.recurrent_inference(
                torch.from_numpy(parent.hidden_state).unsqueeze(0), np.array([action])
            )
            node.expand_node(reward.item(), next_state.squeeze().detach().numpy(), 0, policy.squeeze().detach().numpy())

            # 探索結果をルートまで反映
            self._backpropagate(search_path, value.item(), 0, min_max_stats)

        return root

    def _select_child(self, node: Node, min_max_stats: MinMaxStats) -> Tuple[int, Node]:
        """
        UCBが最も高い子を選択する
        """
        ucb = [self._ucb_score(node, child, min_max_stats) for child in node.children]
        action = np.argmax(ucb)
        return action, node.children[action]

    def _ucb_score(self, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
        """
        UCBの計算
        """
        pb_c = np.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
        pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = min_max_stats.normalize(child.value)
        return prior_score + value_score

    def _backpropagate(self, search_path: List[Node], value: float, player: int, min_max_stats: MinMaxStats):
        for node in reversed(search_path):
            node.value_sum += value if node.player == player else -value
            node.visit_count += 1
            min_max_stats.update(node.value)
            value = node.reward + self.discount * value
