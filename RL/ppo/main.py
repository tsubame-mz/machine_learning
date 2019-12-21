import torch
import numpy as np
import gym
import os
import pickle
from absl import app
from absl import flags

# from utils import _windows_enable_ANSI
from model import PNet, VNet
import ralamb
from trajectory import Buffer, Trajectory
from agent import Agent, Discriminator


FLAGS = flags.FLAGS
flags.DEFINE_string("env", "Taxi-v3", "gym env name")
flags.DEFINE_bool("use_gpu", True, "using GPU")
flags.DEFINE_integer("hid_num", 16, "hidden layer unit size")
flags.DEFINE_float("lr", 1e-4, "learning rate")
flags.DEFINE_float("final_lr", 1e-2, "final learning rate for AdaBound")
flags.DEFINE_float("weight_decay", 1e-6, "weight decay")
flags.DEFINE_float("gamma", 0.99, "reward discount factor")
flags.DEFINE_float("lam", 0.95, "GAE lambda")
flags.DEFINE_float("eps", 1e-6, "small value for preventing 0 division in Advantage")
flags.DEFINE_integer("batch_size", 16, "batch size for each training")
flags.DEFINE_float("clip_param", 0.2, "PPO clipping liklihood ratio")
flags.DEFINE_float("ent_c", 0.2, "entropy coefficient")
flags.DEFINE_float("v_loss_c", 0.9, "value loss coefficient")
flags.DEFINE_float("max_grad_norm", 0.5, "maximum gradient norm")
flags.DEFINE_bool("use_discrim", False, "using AIRL Discriminator")

flags.DEFINE_integer("n_sample_episodes", 10, "sampling episode in each training epoch")
flags.DEFINE_integer("n_train_epochs", 10000, "training epoch")
flags.DEFINE_integer("n_opt_epochs", 3, "update epoch in each iteration")
flags.DEFINE_integer("n_test_epochs", 10, "test epoch")
flags.DEFINE_bool("render_env", True, "env render in test epoch")

flags.DEFINE_string("data_dir", "./data", "data save directory")
# flags.DEFINE_string("data_dir", "/content/drive/My Drive/data", "data save directory")    # Google Drive
flags.DEFINE_bool("resume", True, "load model parameters")


def create_directory(dir_name):
    if not os.path.exists(dir_name):
        print("Create directory: {0}".format(dir_name))
        os.mkdir(dir_name)


def get_device(use_gpu):
    # サポート対象のGPUがあれば使う
    if use_gpu:
        print("Check GPU available")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    return device


def train(env, agent, max_rew, model_filename_base, device, discrim, discrim_filename_base, expert_traj):
    print("Train start")
    try:
        for epoch in range(FLAGS.n_train_epochs):
            # 評価モード
            agent.eval()
            traj = Trajectory()
            episode_rewards = []
            if discrim is not None:
                real_episode_rewards = []
            for episode in range(FLAGS.n_sample_episodes):
                ob = env.reset()
                done = False
                buf = Buffer()
                while not done:
                    ob_tsr = torch.from_numpy(np.array(ob)).to(device=device)  # テンソル化
                    action, value, llh, _ = agent.get_action(ob_tsr)
                    next_ob, reward, done, info = env.step(action)
                    buf_data = dict(obs=ob, actions=action, rewards=reward, values=value, llhs=llh)

                    if discrim is not None:
                        pseudo_rew = discrim.get_pseudo_reward(ob_tsr)
                        buf_data["real_rewards"] = buf_data["rewards"]
                        buf_data["rewards"] = pseudo_rew
                        buf_data["next_obs"] = next_ob
                        buf_data["dones"] = done

                    buf.append(buf_data)
                    ob = next_ob

                buf.finish_path(FLAGS.gamma, FLAGS.lam)
                traj.append(buf.data_map)
                episode_rewards.append(buf.data_map["rewards"].sum())

                if discrim is not None:
                    real_episode_rewards.append(buf.data_map["real_rewards"].sum())

            # 学習モード
            agent.train()
            traj.finish_path(FLAGS.eps, device)
            pol_loss, v_loss, entropy = agent.update(
                traj,
                FLAGS.batch_size,
                FLAGS.n_opt_epochs,
                FLAGS.clip_param,
                FLAGS.v_loss_c,
                FLAGS.ent_c,
                FLAGS.max_grad_norm,
            )
            if discrim is not None:
                discrim.train()
                discrim_a_loss, discrim_e_loss = discrim.update(traj, expert_traj, FLAGS.batch_size, FLAGS.gamma, agent)

            mean_rew = np.mean(episode_rewards)
            print("Epoch[{:5d}/{:5d}], ".format(epoch + 1, FLAGS.n_train_epochs), end="")
            if discrim is not None:
                print(
                    "Loss(P/V/E/Da/De)[{:+.6f}/{:+.6f}/{:+.6f}/{:+.6f}/{:+.6f}], ".format(
                        pol_loss, v_loss, entropy, discrim_a_loss, discrim_e_loss
                    ),
                    end="",
                )
            else:
                print("Loss(P/V/E)[{:+.6f}/{:+.6f}/{:+.6f}], ".format(pol_loss, v_loss, entropy), end="")
            reward_format = "Reward(Mean/Min/Max)[{:+3.3f}/{:+3d}/{:+3d}]"
            print(reward_format.format(mean_rew, int(np.min(episode_rewards)), int(np.max(episode_rewards))), end="")
            if discrim is not None:
                mean_real_rew = np.mean(real_episode_rewards)
                reward_format = ", Reward(Real)(Mean/Min/Max)[{:+3.3f}/{:+3.3f}/{:+3.3f}]"
                print(
                    reward_format.format(mean_real_rew, np.min(real_episode_rewards), np.max(real_episode_rewards)),
                    end="",
                )
            print()

            if (((epoch + 1) % 10) == 0) and (mean_rew > max_rew):
                print("Reward Update: Old[{:+3.3f}] -> New[{:+3.3f}]".format(max_rew, mean_rew))
                max_rew = mean_rew
                print("Save best model")
                save_info = dict(max_rew=max_rew)
                agent.save_model(model_filename_base, save_info, "best")

                if discrim is not None:
                    discrim.save_model(discrim_filename_base, "best")

            if ((epoch + 1) % 100) == 0:
                print("Save checkpoint model")
                save_info = dict(max_rew=max_rew)
                agent.save_model(model_filename_base, save_info, "checkpoint")

                if discrim is not None:
                    discrim.save_model(discrim_filename_base, "checkpoint")

    except KeyboardInterrupt:
        print("Keyboard interrupt")
    print("Train complete")

    print("Save last model")
    save_info = dict(max_rew=max_rew)
    agent.save_model(model_filename_base, save_info, "last")

    if discrim is not None:
        discrim.save_model(discrim_filename_base, "last")


def test(env, agent, device):
    print("Test start")
    try:
        # 評価モード
        agent.eval()
        test_reward = []
        for epoch in range(FLAGS.n_test_epochs):
            ob = env.reset()
            done = False
            step = 0
            episode_reward = 0
            if FLAGS.render_env:
                print("-- env[{0:02d}:{1:03d}] ".format(epoch + 1, step) + "-" * 30)
                env.render()

            while not done:
                ob_tsr = torch.from_numpy(np.array(ob)).to(device=device)
                action, value, _, pi = agent.get_action(ob_tsr)
                next_ob, reward, done, info = env.step(action)
                episode_reward += reward
                ob = next_ob

                step += 1
                if FLAGS.render_env:
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


def main(_):
    device = get_device(FLAGS.use_gpu)
    print("Use device: {}".format(device))

    # モデル保存用フォルダ生成
    data_dir = FLAGS.data_dir
    create_directory(data_dir)
    create_directory(os.path.join(data_dir, "models"))

    env = gym.make(FLAGS.env)

    p_net = PNet(env.observation_space, env.action_space, FLAGS.hid_num)
    v_net = VNet(env.observation_space, FLAGS.hid_num)
    print(p_net)
    print(v_net)
    p_net.to(device)
    v_net.to(device)
    optim_p = ralamb.Ralamb(p_net.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    optim_v = ralamb.Ralamb(v_net.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    agent = Agent(p_net, v_net, optim_p, optim_v, device)

    if FLAGS.use_discrim:
        expert_filename = os.path.join(FLAGS.data_dir, "expert_data", "taxi_expert.pkl")
        print("Load expert data: ", expert_filename)
        with open(expert_filename, "rb") as f:
            expert_traj = Trajectory()
            expert_epis = pickle.load(f)
            for epi in expert_epis:
                epi["next_obs"] = np.append(epi["obs"][1:], epi["obs"][0])
                expert_traj.append(epi)
            expert_traj.to_tensor(device)

        pseudo_rew_net = VNet(env.observation_space, FLAGS.hid_num)
        shaping_val_net = VNet(env.observation_space, FLAGS.hid_num)
        print(pseudo_rew_net)
        print(shaping_val_net)
        pseudo_rew_net.to(device)
        shaping_val_net.to(device)
        optim_discrim = ralamb.Ralamb(
            list(pseudo_rew_net.parameters()) + list(shaping_val_net.parameters()),
            lr=FLAGS.lr,
            weight_decay=FLAGS.weight_decay,
        )
        discrim = Discriminator(pseudo_rew_net, shaping_val_net, optim_discrim, device)
    else:
        discrim = None
        expert_traj = None

    # モデルの途中状態を読込む
    max_rew = -1e6
    model_filename_base = os.path.join(FLAGS.data_dir, "models", "model_" + FLAGS.env + "_PPO_H" + str(FLAGS.hid_num))
    discrim_filename_base = None
    if FLAGS.resume:
        print("Load last model")
        load_info = agent.load_model(model_filename_base, "last")
        if load_info:
            max_rew = load_info["max_rew"]
            print("Max reward: {0}".format(max_rew))
        else:
            print("Model file not found")

        if FLAGS.use_discrim:
            discrim_filename_base = os.path.join(
                FLAGS.data_dir, "models", "discrim_" + FLAGS.env + "_AIRL_H" + str(FLAGS.hid_num)
            )
            discrim.load_model(discrim_filename_base, "last")

    train(env, agent, max_rew, model_filename_base, device, discrim, discrim_filename_base, expert_traj)
    test(env, agent, device)


if __name__ == "__main__":
    # _windows_enable_ANSI()
    app.run(main)
