import torch
import numpy as np
import gym
from collections import deque
import argparse
import os

# from utils import _windows_enable_ANSI
from env_wrapper import EnvMonitor
from model import PVNet
from memory import PPOBuffer, RolloutStorage
from agent import PPOAgent


def main():
    parser = argparse.ArgumentParser()
    # 環境関係
    parser.add_argument("--env", type=str, default="Taxi-v2")
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--use_seed", type=int, default=False)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--resume", type=bool, default=True)
    parser.add_argument("--test_epochs", type=int, default=10)
    parser.add_argument("--render_env", type=bool, default=True)
    # ネットワーク関係
    parser.add_argument("--hid_num", type=int, default=128)
    # メモリ関係
    parser.add_argument("--gamma", type=float, default=0.99)  # 0.8～0.99
    parser.add_argument("--lam", type=float, default=0.95)  # 0.9～1.0
    parser.add_argument("--adv_eps", type=float, default=1e-6)
    # 学習関係
    parser.add_argument("--lr", type=float, default=1e-4)  # 0.003〜5e-6(0.000005)
    parser.add_argument("--train_epochs", type=int, default=10000)
    parser.add_argument("--sample_episodes", type=int, default=10)
    parser.add_argument("--opt_epochs", type=int, default=3)  # 3～30
    parser.add_argument("--use_minibatch_seq", type=bool, default=False)
    parser.add_argument("--minibatch_size", type=int, default=64)  # 4～4096
    parser.add_argument("--minibatch_size_seq", type=int, default=4)
    parser.add_argument("--clip_ratio", type=float, default=0.2)  # 0.1～0.3
    parser.add_argument("--v_loss_c", type=float, default=0.9)  # 0.5～1.0
    parser.add_argument("--start_ent_c", type=float, default=10.0)  # 10.0～0.1
    parser.add_argument("--end_ent_c", type=float, default=0.01)  # 0～0.01
    parser.add_argument("--tau_ent_c", type=float, default=1000.0)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    # ログ関係
    parser.add_argument("--data", type=str, default="data")
    # parser.add_argument("--data", type=str, default="/content/drive/My Drive/data")
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
    env = EnvMonitor(env)

    pv_net = PVNet(env.observation_space, env.action_space, args)
    print(pv_net)
    pv_net.to(device)
    optimizer = torch.optim.Adam(pv_net.parameters(), lr=args.lr)
    agent = PPOAgent(pv_net, optimizer, env.observation_space, device, args)
    local_buf = PPOBuffer(args)
    rollouts = RolloutStorage(args)

    # モデルの途中状態を読込む
    max_rew = -1e6
    model_filename_base = "model_" + args.env + "_PPO_H" + str(args.hid_num)
    last_model_filename = os.path.join(args.data, "models", model_filename_base + "_last.pkl")
    best_model_filename = os.path.join(args.data, "models", model_filename_base + "_best.pkl")
    checkpoint_model_filename = os.path.join(args.data, "models", model_filename_base + "_checkpoint.pkl")
    if args.resume:
        print("Load model params")
        load_filename = None
        if os.path.exists(last_model_filename):
            # load last model
            print("Load last model")
            load_filename = last_model_filename
        elif os.path.exists(checkpoint_model_filename):
            # load checkpoint model
            print("Load checkpoint model")
            load_filename = checkpoint_model_filename
        elif os.path.exists(best_model_filename):
            # load best model
            print("Load best model")
            load_filename = best_model_filename

        if load_filename:
            load_data = torch.load(load_filename, map_location=device)
            pv_net.load_state_dict(load_data["state_dict_pv"])
            max_rew = load_data["max_rew"]
            print("Max reward: {0}".format(max_rew))
        else:
            print("Model file not found")

    print("Train start")
    try:
        episode_rewards = deque(maxlen=args.sample_episodes)
        for epoch in range(args.train_epochs):
            # 評価モード
            pv_net.eval()
            for episode in range(args.sample_episodes):
                obs = env.reset()
                done = False
                while not done:
                    action, value, log_pi, _ = agent.get_action(obs)
                    next_obs, reward, done, info = env.step(action)
                    local_buf.add(obs, action, reward, value, log_pi)
                    obs = next_obs

                    if "episode" in info.keys():
                        eprew = info["episode"]["r"]
                        episode_rewards.append(eprew)

                local_buf.finish_path()
                rollouts.append(local_buf)
                local_buf.initialize()

            # 学習モード
            pv_net.train()
            rollouts.finish_path()
            pi_loss, v_loss, entropy = agent.train(rollouts)
            rollouts.initialize()

            mean_rew = np.mean(episode_rewards)
            print("Epoch[{:3d}], ".format(epoch + 1), end="")
            print("Loss(P/V/E)[{:+.6f}/{:+.6f}/{:+.6f}], ".format(pi_loss, v_loss, entropy), end="")
            reward_format = "Reward(Mean/Min/Max)[{:+3.3f}/{:+3.3f}/{:+3.3f}]"
            print(reward_format.format(mean_rew, np.min(episode_rewards), np.max(episode_rewards)), end="")
            print()

            if mean_rew > max_rew:
                print("Reward Update: Old[{:+3.3f}] -> New[{:+3.3f}]".format(max_rew, mean_rew))
                max_rew = mean_rew
                # save best model
                print("Save best model")
                save_data = {"state_dict_pv": pv_net.state_dict(), "max_rew": max_rew}
                torch.save(save_data, best_model_filename)

            if ((epoch + 1) % 100) == 0:
                # save checkpoint model
                print("Save checkpoint model")
                save_data = {"state_dict_pv": pv_net.state_dict(), "max_rew": max_rew}
                torch.save(save_data, checkpoint_model_filename)

            # if (args.env in ["CartPole-v0", "FrozenLake-v0"]) and (mean_rew > 0.95):
            #     break
    except KeyboardInterrupt:
        print("Keyboard interrupt")
    print("Train complete")

    # save last model
    print("Save last model")
    save_data = {"state_dict_pv": pv_net.state_dict(), "max_rew": max_rew}
    torch.save(save_data, last_model_filename)

    # test
    test_reward = []
    print("Test start")
    try:
        # 評価モード
        pv_net.eval()
        for epoch in range(args.test_epochs):
            obs = env.reset()
            done = False
            episode_reward = 0
            step = 0
            if args.render_env:
                print("-- env[{0:02d}:{1:03d}] ".format(epoch + 1, step) + "-" * 30)
                env.render()
            while not done:
                action, value, log_pi, pi = agent.get_action(obs)
                next_obs, reward, done, info = env.step(action)
                episode_reward += reward
                obs = next_obs

                step += 1
                if args.render_env:
                    print("-- env[{0:02d}:{1:03d}] ".format(epoch + 1, step) + "-" * 30)
                    env.render()
                    print(info, (pi * 1000).astype(int), "{:.6f}".format(value))

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
