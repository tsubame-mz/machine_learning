import os
import gym
from gym.wrappers import RecordEpisodeStatistics

from network import Network
from env_wrapper import TaxiObservationWrapper
from utils import get_device
from mcts import MCTS
from agent import Agent
from trainer import Trainer


def main():
    env_name = "Taxi-v3"
    state_units = 16
    hid_units = 8
    dirichlet_alpha = 0.25
    exploration_fraction = 0.25
    pb_c_base = 19652
    pb_c_init = 1.25
    discount = 0.99
    num_simulations = 50
    filename = "model_last.pth"
    is_train = False

    device = get_device(True)

    env = gym.make(env_name)
    env = RecordEpisodeStatistics(env)
    env = TaxiObservationWrapper(env)

    network = Network(env.observation_space.nvec.sum(), env.action_space.n, state_units, hid_units)
    mcts = MCTS(dirichlet_alpha, exploration_fraction, pb_c_base, pb_c_init, discount, num_simulations, is_train)
    agent = Agent(network, mcts)
    trainer = Trainer()

    if os.path.exists(filename):
        agent.load_model(filename, device)

    trainer.validate(env, agent, network)


if __name__ == "__main__":
    main()
