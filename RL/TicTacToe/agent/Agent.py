from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def get_action(self, env):
        raise NotImplementedError
