import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import gym_tictactoe  # NOQA
import radam  # NOQA

from .config import AlphaZeroConfig
from .network import AlphaZeroNetwork
from .replay import ReplayBuffer
from .self_play import SelfPlay
from .trainer import Trainer
from .AlphaZeroAgent import AlphaZeroAgent


class AlphaZero:
    def __init__(self):
        self.config = AlphaZeroConfig()
        np.random.seed(self.config.seed)
        torch.random.manual_seed(self.config.seed)

        self.env = gym.make("TicTacToe-v0")
        self.env.seed(self.config.seed)

        network = AlphaZeroNetwork(self.config)
        # print(network)
        network.to(self.config.device)
        optimizer = radam.RAdam(network.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        writer = SummaryWriter("./logs/AlphaZero")

        self.agent = AlphaZeroAgent(self.config, network)
        replay = ReplayBuffer(self.config)
        self.self_play = SelfPlay(self.config, self.env, self.agent, replay)
        self.trainer = Trainer(self.config, network, optimizer, replay, writer)

        self.model_file_path = "./pretrained/" + self.config.model_file
        self.agent.load_model(self.model_file_path)

    def train(self):
        try:
            for i in range(self.config.max_training_step):
                self.self_play.run()
                self.trainer.run(i)

                if (i > 0) and (i % self.config.validate_interval) == 0:
                    self.validate()
                    self.agent.save_model(self.model_file_path)
        except KeyboardInterrupt:
            print("Keyboard interrupt")
        print("Train complete")
        self.validate(True)
        self.agent.save_model(self.model_file_path)

    def validate(self, is_render=False):
        obs = self.env.reset()
        done = False
        while not done:
            if is_render:
                self.env.render()
            action, root = self.agent.get_action(self.env, obs, True)
            if is_render:
                root.print_node(limit_depth=1)
            obs, _, done, _ = self.env.step(action)
        self.env.render()
        print(f"Winner[{obs['winner']}]")
