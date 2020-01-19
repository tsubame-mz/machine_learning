from TicTacToe import TicTacToeEnv
from agent import AlphaZeroAgent


def main():
    env = TicTacToeEnv()
    agent = AlphaZeroAgent()

    env.reset()
    env.step(1)
    env.step(0)
    env.step(4)
    action = agent.get_action(env)
    print(action)


if __name__ == "__main__":
    main()
