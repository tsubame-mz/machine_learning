from typing import Dict
import gym
import numpy as np

from agent import Agent


class RandomAgent(Agent):  # type: ignore
    def get_action(self, env: gym.Env, obs: Dict):
        return np.random.choice(obs["legal_actions"])
