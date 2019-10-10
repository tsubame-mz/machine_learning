import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import pickle
import os


def sarsa(q_value, state, action, reward, next_state, next_action, eta, gamma):
    q_value[state, action] = (1 - eta) * q_value[state, action] + eta * (
        reward + gamma * q_value[next_state, next_action]
    )
    return q_value


def get_action(q_value, state):
    logits = F.softmax(torch.from_numpy(q_value[state, :]), dim=0)
    action = Categorical(logits=logits).sample().item()
    return action


def one_epi(env, q_value, eta, gamma):
    state = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = get_action(q_value, state)
        next_state, reward, done, info = env.step(action)
        next_action = get_action(q_value, next_state)
        # print(state, action, reward, next_state, next_action)
        q_value = sarsa(q_value, state, action, reward, next_state, next_action, eta, gamma)

        episode_reward += reward
        state = next_state
    return q_value, episode_reward


def main():
    env = gym.make("Taxi-v2")
    ob_num = env.observation_space.n
    ac_num = env.action_space.n

    eta = 0.1  # 学習率
    gamma = 0.9  # 報酬割引率
    n_episode = 1000
    log_interval = 10

    q_value = np.zeros((ob_num, ac_num))
    filename = "./data/taxi_q_value.pkl"
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            print("load q_value")
            q_value = pickle.load(f)

    try:
        for episode in range(n_episode):
            q_value, episode_reward = one_epi(env, q_value, eta, gamma)
            if ((episode + 1) % log_interval) == 0:
                print("Episode[{:06d}/{:06d}], Reward[{:+04d}]".format(episode + 1, n_episode, episode_reward))
    except KeyboardInterrupt:
        print("Keyboard interrupt")

    for ob in range(ob_num):
        print("{:03d}: ".format(ob), end="")
        for ac in range(ac_num):
            print("{:+8.4f} ".format(q_value[ob, ac]), end="")
        print()

    with open(filename, "wb") as f:
        print("save q_value")
        pickle.dump(q_value, f)


if __name__ == "__main__":
    main()
