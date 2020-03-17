import gym

import gym_tictactoe  # NOQA
from agent import AlphaZeroAgent
from agent.AlphaZero import AlphaZeroConfig, AlphaZeroNetwork


def test_agent_mcts_action(env, actions, agent, except_action):
    obs = env.reset()
    for action in actions:
        obs, reward, done, _ = env.step(action)
    env.render()

    action, root = agent.get_action(env, obs, True)

    root.print_node()
    print(action)

    total_visit = sum([edge.visit_count for edge in root.edges])
    edge_map = {edge.action: edge.visit_count for edge in root.edges}
    child_visit = [edge_map[action] / total_visit if action in edge_map else 0 for action in range(9)]
    print(child_visit)
    assert action == except_action


def test_mcts_b_win(env, agent):
    # 勝てる手がある
    actions = [0, 1, 3, 4]
    except_action = 6
    test_agent_mcts_action(env, actions, agent, except_action)


def test_mcts_w_win(env, agent):
    # 勝てる手がある
    actions = [0, 4, 6, 3, 1]
    except_action = 5
    test_agent_mcts_action(env, actions, agent, except_action)


def test_mcts_b_lose(env, agent):
    # その場所以外負ける手がある
    actions = [0, 1, 8, 4]
    except_action = 7
    test_agent_mcts_action(env, actions, agent, except_action)


def test_mcts_w_lose(env, agent):
    # その場所以外負ける手がある
    actions = [6, 3, 7]
    except_action = 8
    test_agent_mcts_action(env, actions, agent, except_action)


if __name__ == "__main__":
    env = gym.make("TicTacToe-v0")

    config = AlphaZeroConfig()
    network = AlphaZeroNetwork(config)
    agent = AlphaZeroAgent(config, network)
    agent.load_model("./pretrained/alphazero_model.pth")

    test_mcts_b_win(env, agent)
    test_mcts_w_win(env, agent)
    test_mcts_b_lose(env, agent)
    test_mcts_w_lose(env, agent)
