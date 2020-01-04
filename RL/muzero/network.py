import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class MLP(nn.Module):
    def __init__(self, in_units: int, hid_units: int, out_units: int):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_units, hid_units),
            Swish(),
            nn.Linear(hid_units, hid_units),
            Swish(),
            nn.Linear(hid_units, out_units),
        )
        self.layers.apply(weight_init)

    def forward(self, obs):
        return self.layers(obs)


class Representation(nn.Module):
    """
    observation -> hidden state
    """

    def __init__(self, obs_units: int, hid_units: int, state_units: int):
        super(Representation, self).__init__()
        self.layers = MLP(obs_units, hid_units, state_units)

    def forward(self, x):
        return self.layers(x)


class Prediction(nn.Module):
    """
    hidden state -> policy + value
    """

    def __init__(self, state_units: int, hid_units: int, act_units: int):
        super(Prediction, self).__init__()

        # Policy : 各行動の確率(logit)を出力
        self.policy_layers = nn.Sequential(MLP(state_units, hid_units, act_units), nn.Softmax(dim=-1))

        # Value : 状態の価値を出力
        self.value_layers = MLP(state_units, hid_units, 1)

    def forward(self, x):
        return self.policy_layers(x), self.value_layers(x)


class Dynamics(nn.Module):
    """
    hidden state + action -> next hidden state + reward
    """

    def __init__(self, state_units: int, act_units: int, hid_units: int):
        super(Dynamics, self).__init__()
        in_units = state_units + act_units
        self.state_layers = MLP(in_units, hid_units, state_units)
        self.reward_layers = MLP(in_units, hid_units, 1)

    def forward(self, x):
        return self.state_layers(x), self.reward_layers(x)


class Network(nn.Module):
    def __init__(self, obs_units: int, act_units: int, state_units: int, hid_units: int):
        super(Network, self).__init__()
        self.representation = Representation(obs_units, hid_units, state_units)
        self.prediction = Prediction(state_units, hid_units, act_units)
        self.dynamics = Dynamics(state_units, act_units, hid_units)

    def initial_inference(self, obs: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Representation + Prediction
        """
        state = self.representation(self._conditioned_observation(obs))
        policy, value = self.prediction(state)
        return state, policy, value

    def recurrent_inference(
        self, state: torch.Tensor, action: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dynamics + Prediction
        """
        next_state, reward = self.dynamics(self._conditioned_state(state, action))
        policy, value = self.prediction(next_state)
        return next_state, reward, policy, value

    def _conditioned_observation(self, obs: np.array) -> torch.Tensor:
        obs_t = obs.T
        taxi_row = torch.eye(5)[obs_t[0]]
        taxi_col = torch.eye(5)[obs_t[1]]
        pass_idx = torch.eye(5)[obs_t[2]]
        dest_idx = torch.eye(4)[obs_t[3]]
        return torch.cat([taxi_row, taxi_col, pass_idx, dest_idx], dim=1)

    def _conditioned_state(self, state: torch.Tensor, action: np.ndarray) -> torch.Tensor:
        return torch.cat([state, torch.eye(6)[action]], dim=1)
