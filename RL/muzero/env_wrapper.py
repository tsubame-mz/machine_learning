import gym
import numpy as np


class TaxiObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(TaxiObservationWrapper, self).__init__(env)
        self.observation_space = gym.spaces.MultiDiscrete([5, 5, 5, 4])

    def observation(self, observation):
        return self._decode(observation)

    def _decode(self, obs):
        dest_idx = obs % 4
        obs = obs // 4
        pass_idx = obs % 5
        obs = obs // 5
        taxi_col = obs % 5
        obs = obs // 5
        taxi_row = obs
        return np.array([taxi_row, taxi_col, pass_idx, dest_idx])
