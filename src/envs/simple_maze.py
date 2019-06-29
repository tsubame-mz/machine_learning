import sys
import gym
from gym.utils import seeding
import numpy as np


"""
+------+
|@....@|
+-+..+-+
|@|..|@|
|.|....|
|.|.+-.|
|...|@.|
+---+--+
@ : Start or Goal
"""


class SimpleMaze(gym.Env):
    MAX_ROWS = 8
    MAX_COLS = 8
    LOCATIONS = [(1, 1), (3, 1), (1, 6), (6, 5), (3, 6)]  # row, col
    NUM_LOCATIONS = len(LOCATIONS)
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # ↑↓←→
    NUM_ACTIONS = len(ACTIONS)
    ACTIONS_STR = ["↑", "↓", "←", "→"]
    # fmt: off
    MAP = [
        "11111111",
        "10000001",
        "11100111",
        "10100101",
        "10100001",
        "10101101",
        "10001001",
        "11111111"
    ]
    # fmt: on

    def __init__(self):
        self.observation_space = gym.spaces.Discrete(self.MAX_ROWS * self.MAX_COLS * self.NUM_LOCATIONS)
        self.action_space = gym.spaces.Discrete(self.NUM_ACTIONS)
        self.seed()
        self.reset()

    def reset(self):
        self.map = np.asarray(self.MAP, dtype="c")

        random_loc_idx = self.np_random.permutation(range(len(self.LOCATIONS)))
        self.player_pos = self.LOCATIONS[random_loc_idx[0]]
        self.goal_idx = random_loc_idx[1]
        self.last_action = None
        return self.state

    def step(self, action):
        action_dir = self.ACTIONS[action]
        next_player_row = self.player_pos[0] + action_dir[0]
        next_player_col = self.player_pos[1] + action_dir[1]

        done = False
        reward = 0

        if self.map[next_player_row][next_player_col] == b"0":
            self.player_pos = (next_player_row, next_player_col)
            if self.player_pos == self.goal_pos:
                # ゴール
                done = True
                reward = 20
            else:
                reward = -1
        else:
            # 壁
            reward = -5
        self.last_action = action
        return self.state, reward, done, {}

    def render(self, mode="human"):
        out_map = self.map.copy().tolist()
        out_map = [[cell.decode("utf-8") for cell in rows] for rows in out_map]
        out_map[self.goal_pos[0]][self.goal_pos[1]] = "G"
        out_map[self.player_pos[0]][self.player_pos[1]] = "P"
        out_map = "\n".join(["".join(rows) for rows in out_map]).replace("1", "#").replace("0", ".")
        sys.stdout.write(out_map)
        sys.stdout.write("\n")
        if self.last_action is not None:
            sys.stdout.write(" " * 4 + "(" + self.ACTIONS_STR[self.last_action] + " )\n")

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)

    @property
    def state(self):
        player_row = self.player_pos[0]
        player_col = self.player_pos[1]
        state = (((player_row * self.MAX_COLS) + player_col) * len(self.LOCATIONS)) + self.goal_idx
        assert state < self.observation_space.n
        return state

    @property
    def goal_pos(self):
        return self.LOCATIONS[self.goal_idx]
