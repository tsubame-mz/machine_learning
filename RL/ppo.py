import torch
import numpy as np
import gym
import argparse
import os
from adabound import AdaBound

# from utils import _windows_enable_ANSI
from model import PNet, VNet
from trajectory import Buffer, Trajectory
from agent import Agent


def train(
    env,
    agent,
    n_sample_episodes,
    n_train_epochs,
    gamma,
    lam,
    eps,
    batch_size,
    n_opt_epochs,
    clip_param,
    v_loss_c,
    ent_c,
    max_grad_norm,
    max_rew,
    last_model_filename,
    best_model_filename,
    checkpoint_model_filename,
    device,
):
    print("Train start")
    try:
        for epoch in range(n_train_epochs):
            # 評価モード
            agent.eval()
            traj = Trajectory()
            episode_rewards = []
            for episode in range(n_sample_episodes):
                ob = env.reset()
                done = False
                buf = Buffer()
                while not done:
                    ob = torch.from_numpy(np.array(ob)).to(device=device)  # テンソル化
                    action, value, llh, _ = agent.get_action(ob)
                    next_ob, reward, done, info = env.step(action)
                    buf.append(ob, action, reward, value, llh)

                    ob = next_ob

                buf.finish_path(gamma, lam)
                traj.append(buf)
                episode_rewards.append(buf.data_map["rewards"].sum())

            # 学習モード
            agent.train()
            traj.finish_path(eps, device)
            pol_loss, v_loss, entropy = agent.update(
                traj, batch_size, n_opt_epochs, clip_param, v_loss_c, ent_c, max_grad_norm
            )

            mean_rew = np.mean(episode_rewards)
            print("Epoch[{:3d}], ".format(epoch + 1), end="")
            print("Loss(P/V/E)[{:+.6f}/{:+.6f}/{:+.6f}], ".format(pol_loss, v_loss, entropy), end="")
            reward_format = "Reward(Mean/Min/Max)[{:+3.3f}/{:+3.3f}/{:+3.3f}]"
            print(reward_format.format(mean_rew, np.min(episode_rewards), np.max(episode_rewards)), end="")
            print()

            if (((epoch + 1) % 10) == 0) and (mean_rew > max_rew):
                print("Reward Update: Old[{:+3.3f}] -> New[{:+3.3f}]".format(max_rew, mean_rew))
                max_rew = mean_rew
                print("Save best model")
                save_info = dict(max_rew=max_rew)
                agent.save_model(best_model_filename, save_info)

            if ((epoch + 1) % 100) == 0:
                print("Save checkpoint model")
                save_info = dict(max_rew=max_rew)
                agent.save_model(checkpoint_model_filename, save_info)

    except KeyboardInterrupt:
        print("Keyboard interrupt")
    print("Train complete")

    print("Save last model")
    save_info = dict(max_rew=max_rew)
    agent.save_model(last_model_filename, save_info)


def test(env, agent, n_test_epochs, render_env, device):
    print("Test start")
    try:
        # 評価モード
        agent.eval()
        test_reward = []
        for epoch in range(n_test_epochs):
            ob = env.reset()
            done = False
            step = 0
            episode_reward = 0
            if render_env:
                print("-- env[{0:02d}:{1:03d}] ".format(epoch + 1, step) + "-" * 30)
                env.render()

            while not done:
                ob = torch.from_numpy(np.array(ob)).to(device=device)
                action, value, _, pi = agent.get_action(ob)
                next_ob, reward, done, info = env.step(action)
                episode_reward += reward
                ob = next_ob

                step += 1
                if render_env:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Taxi-v3")
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--hid_num", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--ent_c", type=float, default=0.2)
    parser.add_argument("--v_loss_c", type=float, default=0.9)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)

    parser.add_argument("--n_sample_episodes", type=int, default=10)
    parser.add_argument("--n_train_epochs", type=int, default=10000)
    parser.add_argument("--n_opt_epochs", type=int, default=3)
    parser.add_argument("--n_test_epochs", type=int, default=10)
    parser.add_argument("--render_env", type=bool, default=True)

    parser.add_argument("--data_dir", type=str, default="data")
    # parser.add_argument("--data", type=str, default="/content/drive/My Drive/data")
    parser.add_argument("--resume", type=bool, default=True)

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

    # # モデル保存用フォルダ生成
    if not os.path.exists(args.data_dir):
        print("Create directory: {0}".format(args.data_dir))
        os.mkdir(args.data_dir)
    if not os.path.exists(os.path.join(args.data_dir, "models")):
        print("Create directory: {0}".format(os.path.join(args.data_dir, "models")))
        os.mkdir(os.path.join(args.data_dir, "models"))

    env = gym.make(args.env)

    p_net = PNet(env.observation_space, env.action_space, args.hid_num)
    v_net = VNet(env.observation_space, args.hid_num)
    print(p_net)
    print(v_net)
    p_net.to(device)
    v_net.to(device)
    optim_p = AdaBound(p_net.parameters(), lr=args.lr, final_lr=0.01)
    optim_v = AdaBound(v_net.parameters(), lr=args.lr, final_lr=0.01)
    agent = Agent(p_net, v_net, optim_p, optim_v, env.observation_space, device)

    # モデルの途中状態を読込む
    max_rew = -1e6
    model_filename_base = os.path.join(args.data_dir, "models", "model_" + args.env + "_PPO_H" + str(args.hid_num))
    last_model_filename = model_filename_base + "_last.pkl"
    best_model_filename = model_filename_base + "_best.pkl"
    checkpoint_model_filename = model_filename_base + "_checkpoint.pkl"
    if args.resume:
        print("Load model params")
        if os.path.exists(last_model_filename):
            print("Load last model")
            load_info = agent.load_model(last_model_filename)
            max_rew = load_info["max_rew"]
            print("Max reward: {0}".format(max_rew))
        else:
            print("Model file not found")

    train(
        env,
        agent,
        args.n_sample_episodes,
        args.n_train_epochs,
        args.gamma,
        args.lam,
        args.eps,
        args.batch_size,
        args.n_opt_epochs,
        args.clip_param,
        args.v_loss_c,
        args.ent_c,
        args.max_grad_norm,
        max_rew,
        last_model_filename,
        best_model_filename,
        checkpoint_model_filename,
        device,
    )

    test(env, agent, args.n_test_epochs, args.render_env, device)


if __name__ == "__main__":
    # _windows_enable_ANSI()
    main()
