from __future__ import annotations
import os
from typing import List, Optional, Tuple
import gym
from gym.wrappers import RecordEpisodeStatistics
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from env_wrapper import TaxiObservationWrapper
from network import Network
from utils import MinMaxStats
from ralamb import Ralamb


class Node:
    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count: int = 0
        self.value_sum: float = 0
        self.reward: float = 0
        self.hidden_state: Optional[np.ndarray] = None
        self.player: int = 0
        self.children: List[Node] = []

    def expand_node(self, reward: float, hidden_state: np.ndarray, player: int, policy: np.ndarray):
        """
        子ノードを展開
        """
        self.reward = reward
        self.hidden_state = hidden_state
        self.player = player
        for p in policy:
            self.children.append(Node(p))

    def add_exploration_noise(self, dirichlet_alpha: float, exploration_fraction: float):
        """
        子ノードの優先度にディリクレノイズをかける
        """
        noise = np.random.dirichlet([dirichlet_alpha] * len(self.children))
        for n, child in zip(noise, self.children):
            child.prior = child.prior * (1 - exploration_fraction) + (n * exploration_fraction)

    @property
    def expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def print_node(self, depth=0):
        print("- Node -" + "-" * 72) if depth == 0 else None
        print("--" * depth, end="")
        print(" ", end="") if depth != 0 else None
        print(
            f"id[{id(self)}]/prior[{self.prior:.3f}]/count[{self.visit_count}]/sum[{self.value_sum:.3f}]/value[{self.value:.3f}]/reward[{self.reward:.3f}]/player[{self.player}]"
        )
        for child in self.children:
            child.print_node(depth + 1)
        print("-" * 80) if depth == 0 else None


class MCTS:
    def __init__(
        self,
        dirichlet_alpha: float,
        exploration_fraction: float,
        pb_c_base: float,
        pb_c_init: float,
        discount: float,
        num_simulations: int,
    ):
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.discount = discount
        self.num_simulations = num_simulations

    def run_mcts(self, obs: np.ndarray, network: Network) -> Node:
        # ルートノードを展開
        root = Node(0)
        state, policy, value = network.initial_inference(obs)
        root.expand_node(0, state.squeeze().detach().numpy(), 0, policy.squeeze().detach().numpy())
        root.add_exploration_noise(self.dirichlet_alpha, self.exploration_fraction)  # if train:

        min_max_stats = MinMaxStats(None)
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            while node.expanded:
                # 展開されていない子まで辿る
                action, node = self._select_child(node, min_max_stats)
                search_path.append(node)

            # 子ノードを展開
            parent = search_path[-2]
            next_state, reward, policy, value = network.recurrent_inference(
                torch.from_numpy(parent.hidden_state).unsqueeze(0), np.array([action])
            )
            node.expand_node(reward.item(), next_state.squeeze().detach().numpy(), 0, policy.squeeze().detach().numpy())

            # 探索結果をルートまで反映
            self._backpropagate(search_path, value.item(), 0, min_max_stats)

        return root

    def _select_child(self, node: Node, min_max_stats: MinMaxStats) -> Tuple[int, Node]:
        """
        UCBが最も高い子を選択する
        """
        ucb = [self._ucb_score(node, child, min_max_stats) for child in node.children]
        action = np.argmax(ucb)
        return action, node.children[action]

    def _ucb_score(self, parent: Node, child: Node, min_max_stats: MinMaxStats) -> float:
        """
        UCBの計算
        """
        pb_c = np.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
        pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = min_max_stats.normalize(child.value)
        return prior_score + value_score

    def _backpropagate(self, search_path: List[Node], value: float, player: int, min_max_stats: MinMaxStats):
        for node in reversed(search_path):
            node.value_sum += value if node.player == player else -value
            node.visit_count += 1
            min_max_stats.update(node.value)
            value = node.reward + self.discount * value


class Agent:
    def __init__(self, network: Network, mcts: MCTS):
        self.network = network
        self.mcts = mcts

    def get_action(self, obs: np.ndarray) -> Tuple[int, Node]:
        root = self.mcts.run_mcts(obs, self.network)
        # root.print_node()
        action = self._select_action(root)
        return action, root

    def _select_action(self, node: Node) -> int:
        return Categorical(logits=torch.Tensor([child.visit_count for child in node.children])).sample().item()  # type: ignore


class GameBuffer:
    def __init__(self):
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.policy: List[List[float]] = []

    def append(self, obs: np.ndarray, action: int, reward: float, values: float, policy: List[float]):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(values)
        self.policy.append(policy)

    def print_buffer(self):
        for i, (obs, action, reward, value, policy) in enumerate(
            zip(self.observations, self.actions, self.rewards, self.values, self.policy)
        ):
            print(f"[{i+1:3d}]: obs[{obs}]/action[{action}]/reward[{reward:.3f}]/value[{value:.3f}], policy[{policy}]")

    def sample_target(self, num_unroll_steps: int, td_steps: int, discount: float):
        pos = np.random.randint(0, len(self.rewards) - num_unroll_steps)
        return (
            self.observations[pos],
            self.actions[pos : pos + num_unroll_steps],
            self._make_target(pos, num_unroll_steps, td_steps, discount),
        )

    def _make_target(
        self, pos: int, num_unroll_steps: int, td_steps: int, discount: float
    ) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        target_values = []
        target_rewards = []
        target_policies = []
        for current_pos in range(pos, pos + num_unroll_steps + 1):
            bootstrap_index = current_pos + td_steps
            if bootstrap_index < len(self.values):
                value = self.values[bootstrap_index] * (discount ** td_steps)
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_pos:bootstrap_index]):
                value += reward * (discount ** i)

            if current_pos < len(self.rewards):
                # (割引報酬和(Value), 報酬(Reward), 方策(Policy))
                target_values.append([value])
                target_rewards.append([self.rewards[current_pos]])
                target_policies.append(self.policy[current_pos])
            else:
                # 終了状態
                target_values.append([0])
                target_rewards.append([0])
                target_policies.append([])
        return target_values, target_rewards, target_policies


class ReplayBuffer:
    def __init__(self, window_size: int, batch_size: int):
        self.window_size = window_size
        self.games: List[GameBuffer] = []
        self.batch_size = batch_size

    def append(self, game: GameBuffer):
        if len(self.games) > self.window_size:
            self.games.pop(0)
        self.games.append(game)

    def sample_batch(self, num_unroll_steps: int, td_steps: int, discount: float):
        batch = []
        games = np.random.choice(self.games, self.batch_size)
        batch = [game.sample_target(num_unroll_steps, td_steps, discount) for game in games]
        return batch


class Trainer:
    def __init__(self):
        pass

    def train(
        self,
        env: gym.Env,
        agent: Agent,
        network: Network,
        optimizer,
        window_size: int,
        nb_self_play: int,
        num_unroll_steps: int,
        td_steps: int,
        discount: float,
        batch_size: int,
        nb_train_update: int,
        nb_train_epochs: int,
    ):
        replay_buffer = ReplayBuffer(window_size, batch_size)

        for epoch in range(nb_train_epochs):
            network.eval()
            rewards = []
            for _ in range(nb_self_play):
                game_buffer = self._play_one_game(env, agent)
                # game_buffer.print_buffer()
                replay_buffer.append(game_buffer)
                rewards.append(np.sum(game_buffer.rewards))

            network.train()
            losses = []
            for _ in range(nb_train_update):
                batch = replay_buffer.sample_batch(num_unroll_steps, td_steps, discount)
                losses.append(self._update_weights(network, optimizer, batch))
            v_loss, r_loss, p_loss = np.mean(losses, axis=0)
            print(
                f"Epoch[{epoch+1}]: Reward[{np.mean(rewards)}], Loss: V[{v_loss:.6f}]/R[{r_loss:.6f}]/P[{p_loss:.6f}]"
            )

    def _play_one_game(self, env: gym.Env, agent: Agent) -> GameBuffer:
        buffer = GameBuffer()
        obs = env.reset()
        done = False
        while not done:
            obs = np.array([obs])
            action, root = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)

            visit_sum = np.sum([child.visit_count for child in root.children])
            child_visits = [child.visit_count / visit_sum for child in root.children]
            buffer.append(obs, action, reward, root.value, child_visits)

            obs = next_obs
        return buffer

    def _update_weights(self, network, optimizer, batch):
        v_loss = 0.0
        r_loss = 0.0
        p_loss = 0.0
        batch_size = len(batch)
        for obs, actions, targets in batch:
            target_values, target_rewards, target_policies = targets
            target_values = torch.Tensor(target_values)
            target_rewards = torch.Tensor(target_rewards)
            target_policies = torch.Tensor(target_policies)

            state, policy, value = network.initial_inference(obs)

            v_loss += F.mse_loss(value, target_values[0].unsqueeze(0))
            p_loss += -(target_policies[0] * policy.log()).mean()

            gradient_scale = 1 / len(actions)
            for i, action in enumerate(actions):
                state, reward, policy, value = network.recurrent_inference(state, np.array([action]))
                v_loss += gradient_scale * F.mse_loss(value, target_values[i + 1].unsqueeze(0))
                r_loss += gradient_scale * F.mse_loss(reward, target_rewards[i + 1].unsqueeze(0))
                p_loss += gradient_scale * -(target_policies[i + 1] * policy.log()).mean()

        v_loss = v_loss / batch_size
        r_loss = r_loss / batch_size
        p_loss = p_loss / batch_size

        optimizer.zero_grad()
        total_loss = v_loss + r_loss + p_loss
        total_loss.backward()
        optimizer.step()

        return v_loss.item(), r_loss.item(), p_loss.item()


def get_device(use_gpu):
    # サポート対象のGPUがあれば使う
    if use_gpu:
        print("Check GPU available")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Use device[{device}]")
    return device


def main():
    seed = 1
    env_name = "Taxi-v3"
    state_units = 16
    hid_units = 8
    dirichlet_alpha = 0.25
    exploration_fraction = 0.25
    pb_c_base = 19652
    pb_c_init = 1.25
    discount = 0.99
    num_simulations = 10
    window_size = 10
    nb_self_play = 5
    num_unroll_steps = 5
    td_steps = 20
    batch_size = 8
    lr = 1e-6
    nb_train_update = 10
    nb_train_epochs = 1000
    filename = "model_last.pth"

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
        print(f"Load last model: {filename}")
        load_data = torch.load(filename, map_location=device)
        network.load_state_dict(load_data["state_dict"])

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
        )
    except KeyboardInterrupt:
        print("Keyboard interrupt")
    print("Train complete")

    print(f"Save last model: {filename}")
    save_data = {"state_dict": network.state_dict()}
    torch.save(save_data, filename)


if __name__ == "__main__":
    main()
