from typing import Optional, List
import numpy as np


class Node:
    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count: int = 0
        self.value_sum: float = 0
        self.reward: float = 0
        self.hidden_state: Optional[np.ndarray] = None
        self.player: int = 0
        self.children: List[Node] = []

    def expand_node(self, reward: float, hidden_state: np.ndarray, player: int, policy: np.ndarray):
        """
        子ノードを展開
        """
        self.reward = reward
        self.hidden_state = hidden_state
        self.player = player
        for p in policy:
            self.children.append(Node(p))

    def add_exploration_noise(self, dirichlet_alpha: float, exploration_fraction: float):
        """
        子ノードの優先度にディリクレノイズをかける
        """
        noise = np.random.dirichlet([dirichlet_alpha] * len(self.children))
        for n, child in zip(noise, self.children):
            child.prior = child.prior * (1 - exploration_fraction) + (n * exploration_fraction)

    @property
    def expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def print_node(self, depth=0):
        print("- Node -" + "-" * 72) if depth == 0 else None
        print("--" * depth, end="")
        print(" ", end="") if depth != 0 else None
        print(
            f"id[{id(self)}]/prior[{self.prior:.3f}]/count[{self.visit_count}]/sum[{self.value_sum:.3f}]/value[{self.value:.3f}]/reward[{self.reward:.3f}]/player[{self.player}]"
        )
        for child in self.children:
            child.print_node(depth + 1)
        print("-" * 80) if depth == 0 else None
