import gym
from gym.utils import seeding
import numpy as np


"""
+------+
|@....@|
+-+.---+
|@|....|
|.|.+-.|
|...|@.|
+---+--+
"""


class SimpleMaze(gym.Env):
    LOCATIONS = ((1, 1), (3, 1), (1, 6), (5, 5))  # row, col
    MAP = np.asarray(
        [
            "11111111",
            "10000001",
            "11101111",
            "10100001",
            "10101101",
            "10001001",
            "11111111",
        ],
        dtype="c",
    )

    def __init__(self):
        self.observation_space = gym.spaces.Discrete(9)
        self.action_space = gym.spaces.Discrete(4)
        self.seed()

    def reset(self):
        locations = self.np_random.permutation(self.LOCATIONS)
        self.player_pos = locations[0]
        self.goal_pos = locations[1]
        return 0

    def step(self, action):
        return 0, -1, False, {}

    def render(self, mode="human"):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)

