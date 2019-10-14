import numpy as np
import gym
from gym.core import Wrapper
from gym.core import RewardWrapper, ObservationWrapper


class EnvMonitor(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._reset_state()

    def reset(self, **kwargs):
        self._reset_state()
        return self.env.reset(**kwargs)

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        self._update(ob, rew, done, info)
        return (ob, rew, done, info)

    def _reset_state(self):
        self.rewards = []

    def _update(self, ob, rew, done, info):
        self.rewards.append(rew)
        if done:
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {"r": round(eprew, 6), "l": eplen}
            if isinstance(info, dict):
                info["episode"] = epinfo
