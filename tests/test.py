import torch
import torch.nn as nn
import gym


def main():
    test_rnn()
    test_lstm()
    test_gru()
    test_gym()


def test_rnn():
    # RNN
    print("RNN")
    rnn_cell = nn.RNNCell(3, 4)
    ix = torch.randn(1, 3)
    hx = torch.zeros(1, 4)
    print("rnn_cell: ", rnn_cell)
    print("ix: ", ix)
    print("hx: ", hx)
    hx = rnn_cell(ix, hx)
    print("hx(output): ", hx)


def test_lstm():
    # LSTM
    print()
    print("LSTM")
    lstm_cell = nn.LSTMCell(3, 4)
    ix = torch.randn(1, 3)
    hx = torch.zeros(1, 4)
    cx = torch.zeros(1, 4)
    print("lstm_cell", lstm_cell)
    print("ix: ", ix)
    print("hx: ", hx)
    print("cx: ", cx)
    hx, cx = lstm_cell(ix, (hx, cx))
    print("hx(output): ", hx)
    print("cx(output): ", cx)


def test_gru():
    # GRU
    print()
    print("GRU")
    gru_cell = nn.GRUCell(3, 4)
    ix = torch.randn(1, 3)
    hx = torch.zeros(1, 4)
    print("gru_cell", gru_cell)
    print("ix: ", ix)
    print("hx: ", hx)
    hx = gru_cell(ix, hx)
    print("hx: ", hx)


def test_gym():
    print()
    print("Gym")
    env = gym.make("CartPole-v1")
    # env = gym.make("Acrobot-v1")
    print("observation space: ", env.observation_space)
    print("action space: ", env.action_space)
    obs = env.reset()
    done = False
    step = 0
    while not done:
        env.render()
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        print(step + 1, obs, action, next_obs, reward, done, info)
        obs = next_obs
        step += 1


if __name__ == "__main__":
    main()
