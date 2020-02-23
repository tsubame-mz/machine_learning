from typing import Dict
from abc import ABC, abstractmethod
import gym


class Agent(ABC):
    @abstractmethod
    def get_action(self, env: gym.Env, obs: Dict):
        raise NotImplementedError
