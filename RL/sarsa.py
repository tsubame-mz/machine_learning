import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import pickle
import os
import math
import time

# from datetime import datetime


def calc_temperature(step, max_t, min_t, tau):
    return (max_t - min_t) * math.exp(-step / tau) + min_t


def sarsa(q_value, ob, action, reward, next_ob, next_action, eta, gamma):
    q_value[ob, action] = (1 - eta) * q_value[ob, action] + eta * (reward + gamma * q_value[next_ob, next_action])
    return q_value


def get_action(q_value, ob, step, max_t, min_t, tau):
    t = calc_temperature(step, max_t, min_t, tau)
    pi = F.softmax(torch.from_numpy(q_value[ob, :]) / t, dim=0)
    action = Categorical(pi).sample().item()
    return action


def get_max_action(q_value, ob):
    return np.argmax(q_value[ob])


def one_epi(env, q_value, eta, gamma, start_step, max_t, min_t, tau):
    done = False
    episode_reward = 0
    rewards = []
    step = start_step

    ob = env.reset()
    action = get_action(q_value, ob, step, max_t, min_t, tau)
    while not done:
        next_ob, reward, done, info = env.step(action)
        next_action = get_action(q_value, next_ob, step, max_t, min_t, tau)
        # print(state, action, reward, next_ob, next_action)
        q_value = sarsa(q_value, ob, action, reward, next_ob, next_action, eta, gamma)

        rewards.append(reward)
        episode_reward += reward
        ob = next_ob
        action = next_action
        step += 1
    return q_value, episode_reward, len(rewards)


def one_epi_test(env, q_value):
    done = False
    episode_reward = 0
    obs = []
    actions = []
    rewards = []
    dones = []

    ob = env.reset()
    while not done:
        action = get_max_action(q_value, ob)
        next_ob, reward, done, info = env.step(action)

        obs.append(ob)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        episode_reward += reward
        ob = next_ob
    return (
        episode_reward,
        len(obs),
        dict(
            obs=np.array(obs, dtype=float),
            actions=np.array(actions, dtype=float),
            rewards=np.array(rewards, dtype=float),
            dones=np.array(dones, dtype=float),
        ),
    )


def train(env, n_episode, q_value, eta, gamma, max_t, min_t, tau, log_interval):
    try:
        start_step = 0
        start_time = time.perf_counter()
        total_time = 0
        for episode in range(n_episode):
            q_value, episode_reward, epi_len = one_epi(env, q_value, eta, gamma, start_step, max_t, min_t, tau)
            start_step += epi_len
            if ((episode + 1) % log_interval) == 0:
                elapsed_time = time.perf_counter() - start_time
                total_time += elapsed_time
                start_time = time.perf_counter()
                print(
                    "[Train]Episode[{:06d}/{:06d}], Reward[{:+04d}], Step[{:3d}], Time[{:9.6f}s]".format(
                        episode + 1, n_episode, episode_reward, epi_len, elapsed_time
                    )
                )
        print("[Train]Total time[{:9.6f}s](ave[{:9.6f}s])".format(total_time, total_time / n_episode))
    except KeyboardInterrupt:
        print("Keyboard interrupt")

    return q_value


def test(env, n_test_episode, q_value):
    try:
        epis = []
        start_time = time.perf_counter()
        total_time = 0
        for episode in range(n_test_episode):
            episode_reward, epi_len, epi = one_epi_test(env, q_value)

            if (episode_reward > 0) and (epi_len < 50):
                epis.append(epi)

            elapsed_time = time.perf_counter() - start_time
            total_time += elapsed_time
            start_time = time.perf_counter()
            print(
                "[Test]Episode[{:06d}/{:06d}], Reward[{:+04d}], Step[{:3d}], Time[{:9.6f}s]".format(
                    episode + 1, n_test_episode, episode_reward, epi_len, elapsed_time
                )
            )
        print("[Test]Total time[{:9.6f}s](ave[{:9.6f}s])".format(total_time, total_time / n_test_episode))
    except KeyboardInterrupt:
        print("Keyboard interrupt")

    return epis


def main():
    is_train = True
    is_test = True

    env = gym.make("Taxi-v3")
    ob_num = env.observation_space.n
    ac_num = env.action_space.n

    eta = 0.1  # 学習率
    gamma = 0.9  # 報酬割引率
    n_episode = 30000
    n_test_episode = 1000
    log_interval = 100

    # temperature
    max_t = 100
    min_t = 0.01
    tau = (n_episode / 50) * 200

    q_value = np.zeros((ob_num, ac_num))
    filename = "./data/taxi_q_value.pkl"
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            print("load q_value")
            q_value = pickle.load(f)

    if is_train:
        q_value = train(env, n_episode, q_value, eta, gamma, max_t, min_t, tau, log_interval)

        with open(filename, "wb") as f:
            print("save q_value")
            pickle.dump(q_value, f)

        for ob in range(ob_num):
            print("{:03d}: ".format(ob), end="")
            for ac in range(ac_num):
                print("{:+8.4f} ".format(q_value[ob, ac]), end="")
            print()

    if is_test:
        # now_str = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
        # expert_filename = "./data/expert_data/taxi_expert_" + now_str + ".pkl"
        expert_filename = "./data/expert_data/taxi_expert.pkl"

        expert_epis = test(env, n_test_episode, q_value)

        if len(expert_epis) > (n_test_episode * 0.95):
            with open(expert_filename, "wb") as f:
                print("Expert episode size[{}]".format(len(expert_epis)))
                pickle.dump(expert_epis, f)


if __name__ == "__main__":
    main()
