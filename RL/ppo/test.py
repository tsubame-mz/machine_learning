import torch
import numpy as np
import gym
import os
from absl import app
from absl import flags

from model import PNet, VNet
from agent import Agent

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "Taxi-v3", "gym env name")
flags.DEFINE_integer("hid_num", 16, "hidden layer unit size")
flags.DEFINE_integer("n_test_epochs", 1, "test epoch")
flags.DEFINE_bool("render_env", True, "env render in test epoch")
flags.DEFINE_string("data_dir", "./data", "data save directory")


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
    device = "cpu"
    print("Use device: {}".format(device))

    env = gym.make(FLAGS.env)

    p_net = PNet(env.observation_space, env.action_space, FLAGS.hid_num)
    v_net = VNet(env.observation_space, FLAGS.hid_num)
    p_net.to(device)
    v_net.to(device)
    agent = Agent(p_net, v_net, None, None, device)

    # モデルの途中状態を読込む
    max_rew = -1e6
    model_filename_base = os.path.join(FLAGS.data_dir, "models", "model_" + FLAGS.env + "_PPO_H" + str(FLAGS.hid_num))
    print("Load best model: {}".format(model_filename_base))
    load_info = agent.load_model(model_filename_base, "best")
    if load_info:
        max_rew = load_info["max_rew"]
        print("Max reward: {0}".format(max_rew))
    else:
        print("Model file not found")
        exit(0)

    test(env, agent, device)


if __name__ == "__main__":
    # _windows_enable_ANSI()
    app.run(main)
