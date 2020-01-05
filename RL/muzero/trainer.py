import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from network import Network
from agent import Agent
from replay import GameBuffer, ReplayBuffer


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
        max_grad_norm: float,
        filename: str,
        ent_c: float,
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
                losses.append(self._update_weights(network, optimizer, batch, max_grad_norm, ent_c))
            v_loss, r_loss, p_loss, entropy = np.mean(losses, axis=0)
            print(
                f"Epoch[{epoch+1}]: Reward[{np.mean(rewards)}], Loss: V[{v_loss:.6f}]/R[{r_loss:.6f}]/P[{p_loss:.6f}]/E[{entropy:.6f}]"
            )

            if (epoch + 1) % 10 == 0:
                agent.save_model(filename)

    def validate(self, env: gym.Env, agent: Agent, network: Network):
        network.eval()
        rewards = []
        for _ in range(1):
            game_buffer = self._play_one_game(env, agent)
            game_buffer.print_buffer()
            rewards.append(np.sum(game_buffer.rewards))
        print(f"Episode reward[{np.mean(rewards)}]")

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

    def _update_weights(self, network, optimizer, batch, max_grad_norm, ent_c):
        v_loss = 0.0
        r_loss = 0.0
        p_loss = 0.0
        entropy = 0.0
        batch_size = len(batch)
        for obs, actions, targets in batch:
            target_values, target_rewards, target_policies = targets
            target_values = torch.Tensor(target_values)
            target_rewards = torch.Tensor(target_rewards)
            target_policies = torch.Tensor(target_policies)

            state, policy, value = network.initial_inference(obs)
            c = Categorical(policy)

            v_loss += F.mse_loss(value, target_values[0].unsqueeze(0))
            p_loss += -(target_policies[0] * policy.log()).mean()
            entropy += -ent_c * c.entropy().mean()

            gradient_scale = 1 / len(actions)
            for i, action in enumerate(actions):
                state, reward, policy, value = network.recurrent_inference(state, np.array([action]))
                v_loss += gradient_scale * F.mse_loss(value, target_values[i + 1].unsqueeze(0))
                r_loss += gradient_scale * F.mse_loss(reward, target_rewards[i + 1].unsqueeze(0))
                p_loss += gradient_scale * (-(target_policies[i + 1] * policy.log()).mean())
                entropy += gradient_scale * (-ent_c * c.entropy().mean())

        v_loss = v_loss / batch_size
        r_loss = r_loss / batch_size
        p_loss = p_loss / batch_size
        entropy = entropy / batch_size

        optimizer.zero_grad()
        total_loss = v_loss + r_loss + p_loss + entropy
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm)
        optimizer.step()

        return v_loss.item(), r_loss.item(), p_loss.item(), entropy.item()
