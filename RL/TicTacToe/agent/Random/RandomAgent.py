import numpy as np

from agent.Agent import Agent


class RandomAgent(Agent):
    def get_action(self, env):
        return np.random.choice(env.legal_actions)
