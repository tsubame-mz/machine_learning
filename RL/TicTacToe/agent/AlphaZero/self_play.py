import copy

from .config import AlphaZeroConfig
from .AlphaZeroAgent import AlphaZeroAgent
from .replay import GameBuffer, ReplayBuffer


class SelfPlay:
    def __init__(self, config: AlphaZeroConfig, env, agent: AlphaZeroAgent, replay: ReplayBuffer):
        self.config = config
        self.env = env
        self.agent = agent
        self.replay = replay

    def run(self):
        for i in range(self.config.self_play_num):
            game = self._play_game()
            self.replay.append(game)

    def _play_game(self):
        obs = self.env.reset()
        done = False
        game = GameBuffer(copy.deepcopy(obs["board"]), obs["to_play"], self.config.action_space)
        while not done:
            # env.render()
            action, root = self.agent.get_action(self.env, obs, True)
            obs, _, done, _ = self.env.step(action)
            game.append(copy.deepcopy(obs["board"]), obs["to_play"], action)
            game.store_search_statistics(root)
        # env.render()
        game.set_winner(obs["winner"], self.config.discount, self.config.terminate_value)
        # game.print_buffer()
        return game
