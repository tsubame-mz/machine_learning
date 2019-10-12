import torch
import numpy as np
import gym
from collections import deque
import argparse
import os

from utils import _windows_enable_ANSI
from env_wrapper import CartPoleRewardWrapper, OnehotObservationWrapper, EnvMonitor
from model import PVNet
from memory import PPOBuffer, RolloutStorage
from agent import PPOAgent


def main():
    parser = argparse.ArgumentParser()
    # 環境関係
    # parser.add_argument("--env", type=str, default="CartPole-v0")
    # parser.add_argument("--env", type=str, default="FrozenLake-v0")
    parser.add_argument("--env", type=str, default="Taxi-v2")
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--use_seed", type=int, default=False)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--resume", type=bool, default=True)
    parser.add_argument("--test_epochs", type=int, default=10)
    parser.add_argument("--render_env", type=bool, default=True)
    # ネットワーク関係
    parser.add_argument("--hid_num", type=int, default=128)
    parser.add_argument("--droprate", type=float, default=0.2)
    # メモリ関係
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--adv_eps", type=float, default=1e-8)
    # 学習関係
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--train_epochs", type=int, default=1000)
    parser.add_argument("--sample_episodes", type=int, default=10)
    parser.add_argument("--opt_epochs", type=int, default=3)
    parser.add_argument("--use_minibatch_seq", type=bool, default=False)
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--minibatch_size_seq", type=int, default=4)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--v_loss_c", type=float, default=0.9)
    parser.add_argument("--ent_c", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    # ログ関係
    parser.add_argument("--data", type=str, default="data")
    args = parser.parse_args(args=[])

    # サポート対象のGPUがあれば使う
    if args.use_gpu:
        print("Check GPU available")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print("Use device: {}".format(device))

    # モデル保存用フォルダ生成
    if not os.path.exists(args.data):
        print("Create directory: {0}".format(args.data))
        os.mkdir(args.data)
    if not os.path.exists(os.path.join(args.data, "models")):
        print("Create directory: {0}".format(os.path.join(args.data, "models")))
        os.mkdir(os.path.join(args.data, "models"))

    # 乱数固定化
    env_seed = None
    if args.use_seed:
        print("Use random seed: {0}".format(args.seed))
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        env_seed = args.seed

    env = gym.make(args.env)
    env.seed(env_seed)
    if args.env == "CartPole-v0":
        env = CartPoleRewardWrapper(env)
    if args.env in ["FrozenLake-v0", "Taxi-v2"]:
        env = OnehotObservationWrapper(env)
    env = EnvMonitor(env)

    model = PVNet(env.observation_space, env.action_space, args)
    print(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    agent = PPOAgent(model, optimizer, device, args)
    local_buf = PPOBuffer(args)
    rollouts = RolloutStorage()

    # モデルの途中状態を読込む
    max_rew = -1e6
    last_model_filename = os.path.join(args.data, "models", "model_last_" + args.env + ".pkl")
    best_model_filename = os.path.join(args.data, "models", "model_best_" + args.env + ".pkl")
    if args.resume:
        print("Load model params")
        if os.path.exists(last_model_filename):
            # load last model
            load_data = torch.load(last_model_filename, map_location=device)
            model.load_state_dict(load_data["state_dict"])
            max_rew = load_data["max_rew"]
            print("Max reward: {0}".format(max_rew))
        else:
            print("Model file not found")

    print("Train start")
    try:
        episode_rewards = deque(maxlen=args.sample_episodes)
        for epoch in range(args.train_epochs):
            model.eval()  # 評価モード
            for episode in range(args.sample_episodes):
                obs = env.reset()
                done = False
                while not done:
                    action, value, log_pi = agent.get_action(obs)
                    next_obs, reward, done, info = env.step(action)
                    local_buf.add(obs, action, reward, value, log_pi)
                    obs = next_obs

                    if "episode" in info.keys():
                        eprew = info["episode"]["r"]
                        episode_rewards.append(eprew)

                local_buf.finish_path()
                rollouts.append(local_buf)
                local_buf.initialize()

            model.train()  # 学習モード
            pi_loss, v_loss, entropy, total_loss = agent.train(rollouts)
            rollouts.initialize()

            mean_rew = np.mean(episode_rewards)
            print("Epoch[{:3d}], ".format(epoch + 1), end="")
            print(
                "Loss(P/V/E/T)[{:+.6f}/{:+.6f}/{:+.6f}/{:+.6f}], ".format(pi_loss, v_loss, entropy, total_loss), end=""
            )
            reward_format = "Reward(Mean/Median/Min/Max)[{:+3.3f}/{:+3.3f}/{:+3.3f}/{:+3.3f}]"
            print(
                reward_format.format(
                    mean_rew, np.median(episode_rewards), np.min(episode_rewards), np.max(episode_rewards)
                ),
                end="",
            )
            print()

            if mean_rew > max_rew:
                print("Reward Update: Old[{:+3.3f}] -> New[{:+3.3f}]".format(max_rew, mean_rew))
                max_rew = mean_rew
                # save best model
                print("Save best model")
                save_data = {"state_dict": model.state_dict(), "max_rew": max_rew}
                torch.save(save_data, best_model_filename)

            if (args.env in ["CartPole-v0", "FrozenLake-v0"]) and (mean_rew > 0.95):
                break
    except KeyboardInterrupt:
        print("Keyboard interrupt")
    print("Train complete")

    # save last model
    print("Save last model")
    save_data = {"state_dict": model.state_dict(), "max_rew": max_rew}
    torch.save(save_data, last_model_filename)

    # test
    test_reward = []
    print("Test start")
    try:
        model.eval()  # 評価モード
        for epoch in range(args.test_epochs):
            obs = env.reset()
            done = False
            episode_reward = 0
            step = 0
            if args.render_env:
                print("-- env[{0:02d}:{1:03d}] ".format(epoch + 1, step) + "-" * 30)
                env.render()
            while not done:
                action, value, log_pi = agent.get_action(obs)
                next_obs, reward, done, info = env.step(action)
                episode_reward += reward
                obs = next_obs

                step += 1
                if args.render_env:
                    print("-- env[{0:02d}:{1:03d}] ".format(epoch + 1, step) + "-" * 30)
                    env.render()
                    print(info)

            episode_reward = np.mean(episode_reward)
            test_reward.append(episode_reward)
            print("Epoch[{:3d}], ".format(epoch + 1), end="")
            print("Reward[{0}]".format(episode_reward), end="")
            print()
        print("Mean reward[{0}]".format(np.mean(test_reward)))
    except KeyboardInterrupt:
        print("Keyboard interrupt")
    print("Test complete")


if __name__ == "__main__":
    # _windows_enable_ANSI()
    main()
