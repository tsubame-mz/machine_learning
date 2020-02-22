from typing import List, Tuple
from abc import ABC, abstractmethod
import numpy as np


class Environment(ABC):
    @abstractmethod
    def step(self, action: int) -> Tuple:
        pass

    @abstractmethod
    def reset(self) -> np.ndarray:
        pass

    @abstractmethod
    def to_play(self) -> int:
        pass

    @abstractmethod
    def legal_actions(self) -> List[int]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def render(self) -> None:
        pass


class Game:
    def __init__(self, env: Environment):
        self.env = env

    def reset(self) -> np.ndarray:
        return self.env.reset()

    def step(self, action: int) -> Tuple:
        obs, reward, done = self.env.step(action)
        return obs, reward, done

    def to_play(self) -> int:
        return self.env.to_play()

    def legal_actions(self) -> List[int]:
        return self.env.legal_actions()

    def close(self) -> None:
        self.env.close()

    def render(self) -> None:
        self.env.render()


class TicTacToeEnv(Environment):
    BLACK = 0
    WHITE = 1
    EMPTY = 2
    PLAYER_NUM = 2
    NUM_CELLS = 9
    TOKEN_LIST = {BLACK: "o", WHITE: "x", EMPTY: "-"}
    LINE_MASKS = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self) -> np.ndarray:
        self.board = np.full(self.NUM_CELLS, self.EMPTY).astype(int)
        self.player = self.BLACK
        self.done = False
        return self.get_obs()

    def step(self, action: int) -> Tuple:
        reward = 0.0
        if self.done:
            pass
        elif action not in self.legal_actions():
            reward = -1.0
            self.done = True
        else:
            self.board[action] = self.player
            self.done = self.judge()
            reward = +1.0 if self.done else 0.0
            self.player = (self.player + 1) % self.PLAYER_NUM

        return self.get_obs(), reward, self.done

    def to_play(self) -> int:
        return self.player

    def legal_actions(self) -> List[int]:
        return np.where(self.board == self.EMPTY)[0].astype(np.int)

    def close(self) -> None:
        pass

    def render(self) -> None:
        print("+" + "-" * 3 + "+")
        for y in range(3):
            print("|", end="")
            for x in range(3):
                idx = y * 3 + x
                print(self.TOKEN_LIST[self.board[idx]], end="")
            print("|")
        print("+" + "-" * 3 + "+")
        print(f"Player[{self.TOKEN_LIST[self.player]}({self.player})], Done[{self.done}]")

    def get_obs(self) -> np.ndarray:
        return np.array(
            [
                np.where(self.board == self.BLACK, 1, 0).reshape((3, 3)),
                np.where(self.board == self.WHITE, 1, 0).reshape((3, 3)),
                np.full((3, 3), self.player),
            ]
        )

    def judge(self) -> bool:
        for mask in self.LINE_MASKS:
            line = self.board[mask]
            hit = np.all(np.where(line == self.player, True, False))
            if hit:
                return True
        return False


if __name__ == "__main__":
    env = TicTacToeEnv()
    obs = env.reset()
    done = False

    while not done:
        env.render()
        print(obs)
        print(env.legal_actions())
        action = np.random.choice(env.legal_actions())
        next_obs, reward, done = env.step(action)
        print(f"Action[{action}], Reward[{reward}]")
        obs = next_obs
    env.render()
    print(obs)
    print(env.legal_actions())
