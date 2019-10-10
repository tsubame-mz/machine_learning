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


class CartPoleRewardWrapper(RewardWrapper):
    def reset(self, **kwargs):
        self.step_cnt = 0
        self.done = False
        return super().reset(**kwargs)

    def step(self, action):
        ob, rew, self.done, info = self.env.step(action)
        self.step_cnt += 1
        return ob, self.reward(rew), self.done, info

    def reward(self, reward):
        reward = 0
        if self.done:
            if self.step_cnt >= 195:
                # 195ステップ以上立てていたらOK
                reward = +1
            else:
                # 途中で転んでいたらNG
                reward = -1
        return reward


class OnehotObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.observation_space, gym.spaces.Discrete)
        self.in_features = self.observation_space.n

    def reset(self, **kwargs):
        return super().reset(**kwargs)

    def observation(self, obs):
        return self.one_hot(obs)

    def one_hot(self, index):
        return np.eye(self.in_features)[index]
