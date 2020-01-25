import numpy as np


class TicTacToeEnv:
    EMPTY = 0
    BLACK = 1
    WHITE = -1
    NUM_PLAYER = 3
    NUM_CELLS = 9
    TOKENS = np.array([".", "o", "x"])
    LINE_MASKS = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]

    def __init__(self):
        super(TicTacToeEnv, self).__init__()
        self.reset()

    def reset(self):
        self._board = np.array([self.EMPTY] * self.NUM_CELLS, dtype=np.int)
        self._player = self.BLACK
        self._done = False
        self._winner = self.EMPTY

    def step(self, action):
        if self._done:
            pass
        elif action not in self.legal_actions:
            self._winner = self.opponent_player
            self._done = True
        else:
            self._board[action] = self._player
            self._winner, self._done = self._judge()
            self._player = self.opponent_player

        if len(self.legal_actions) == 0:
            self._done = True

    def render(self):
        token_list = self.TOKENS[self._board]
        print("+" + "-" * 3 + "+")
        for y in range(3):
            print("|", end="")
            for x in range(3):
                idx = y * 3 + x
                print(token_list[idx], end="")
            print("|")
        print("+" + "-" * 3 + "+")
        print(f"Player[{self._player}], Done[{self._done}], Winner[{self._winner}]")

    @property
    def player(self):
        return self._player

    @property
    def opponent_player(self):
        return -self._player

    @property
    def done(self):
        return self._done

    @property
    def board(self):
        return self._board

    @property
    def winner(self):
        return self._winner

    @property
    def legal_actions(self):
        if self._done:
            return []
        return np.where(self._board == 0)[0].astype(np.int)

    @property
    def observation(self):
        return np.concatenate([self._board, [self._player]])

    def _judge(self):
        winner = self.EMPTY
        done = False

        for mask in self.LINE_MASKS:
            line = self._board[mask]
            hit = np.all(np.where(line == self._player, True, False))
            if hit:
                winner = self._player
                done = True
        return winner, done


if __name__ == "__main__":
    env = TicTacToeEnv()
    env.reset()

    # actions = [4, 1, 3, 5, 6, 0, 2]  # black
    # actions = [0, 1, 2, 4, 3, 7]  # white
    # actions = [0, 1, 2, 4, 3, 5, 7, 6, 8]  # draw
    # actions = [0, 0]  # illegal(white)
    # actions = [0, 1, 1]  # illegal(black)
    # actions = list(reversed(actions))
    while not env.done:
        print("-" * 80)
        env.render()
        obs = env.board
        action = np.random.choice(env.legal_actions)
        # action = actions.pop()
        print("obs: ", obs)
        print("action: ", action)
        env.step(action)
        print("next obs: ", env.board)
        print("winner: ", env.winner)
        print("done: ", env.done)
    print("-" * 80)
    env.render()
