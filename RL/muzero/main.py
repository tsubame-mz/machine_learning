import os
import gym
from gym.wrappers import RecordEpisodeStatistics
import numpy as np
import torch

from env_wrapper import TaxiObservationWrapper
from network import Network
from utils import get_device
from ralamb import Ralamb
from mcts import MCTS
from agent import Agent
from trainer import Trainer


def main():
    seed = 1
    env_name = "Taxi-v3"
    state_units = 16
    hid_units = 8
    dirichlet_alpha = 0.1
    exploration_fraction = 0.25
    pb_c_base = 19652
    pb_c_init = 1.25
    discount = 0.99
    num_simulations = 50
    window_size = 100
    nb_self_play = 5
    num_unroll_steps = 5
    td_steps = 20
    batch_size = 64
    lr = 1e-4
    nb_train_update = 20
    nb_train_epochs = 10000
    max_grad_norm = 0.5
    filename = "model_last.pth"
    ent_c = 0.2

    device = get_device(True)

    env = gym.make(env_name)
    env = RecordEpisodeStatistics(env)
    env = TaxiObservationWrapper(env)

    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format})

    network = Network(env.observation_space.nvec.sum(), env.action_space.n, state_units, hid_units)
    mcts = MCTS(dirichlet_alpha, exploration_fraction, pb_c_base, pb_c_init, discount, num_simulations)
    agent = Agent(network, mcts)
    trainer = Trainer()
    optimizer = Ralamb(network.parameters(), lr=lr)

    if os.path.exists(filename):
        agent.load_model(filename, device)

    print("Train start")
    try:
        trainer.train(
            env,
            agent,
            network,
            optimizer,
            window_size,
            nb_self_play,
            num_unroll_steps,
            td_steps,
            discount,
            batch_size,
            nb_train_update,
            nb_train_epochs,
            max_grad_norm,
            filename,
            ent_c,
        )
    except KeyboardInterrupt:
        print("Keyboard interrupt")
    print("Train complete")

    agent.save_model(filename)


if __name__ == "__main__":
    main()
