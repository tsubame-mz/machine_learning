from typing import Tuple
import numpy as np
import torch
from torch.distributions import Categorical


from node import Node
from network import Network
from mcts import MCTS


class Agent:
    def __init__(self, network: Network, mcts: MCTS):
        self.network = network
        self.mcts = mcts

    def get_action(self, obs: np.ndarray) -> Tuple[int, Node]:
        root = self.mcts.run_mcts(obs, self.network)
        # root.print_node()
        action = self._select_action(root)
        return action, root

    def load_model(self, filename, device):
        print(f"Load last model: {filename}")
        load_data = torch.load(filename, map_location=device)
        self.network.load_state_dict(load_data["state_dict"])

    def save_model(self, filename):
        print(f"Save last model: {filename}")
        save_data = {"state_dict": self.network.state_dict()}
        torch.save(save_data, filename)

    def _select_action(self, node: Node) -> int:
        return Categorical(logits=torch.Tensor([child.visit_count for child in node.children])).sample().item()  # type: ignore
