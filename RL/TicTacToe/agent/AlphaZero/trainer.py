import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

from logger import setup_logger

from .config import AlphaZeroConfig
from .network import AlphaZeroNetwork
from .replay import ReplayBuffer

logger = setup_logger(__name__, logging.INFO)


class Trainer:
    def __init__(
        self,
        config: AlphaZeroConfig,
        network: AlphaZeroNetwork,
        optimizer: Optimizer,
        replay: ReplayBuffer,
        writer: SummaryWriter,
    ):
        self.config = config
        self.network = network
        self.optimizer = optimizer
        self.replay = replay
        self.writer = writer

    def run(self, i: int):
        p_loss, v_loss = self._train()

        win_b_rate, win_w_rate, draw_rate = self._calc_win_rate()
        logger.info(
            f"{i}: Loss:P[{p_loss:.6f}]/V[{v_loss:.6f}], Win:B[{win_b_rate:.6f}]/W[{win_w_rate:.6f}], Draw[{draw_rate:.6f}]"
        )
        self.writer.add_scalar("AlphaZero/p_loss", p_loss, i)
        self.writer.add_scalar("AlphaZero/v_loss", v_loss, i)
        self.writer.add_scalar("AlphaZero/win_b_rate", win_b_rate, i)
        self.writer.add_scalar("AlphaZero/win_w_rate", win_w_rate, i)
        self.writer.add_scalar("AlphaZero/draw_rate", draw_rate, i)

    def _train(self):
        self.network.train()

        batch = self.replay.sample_batch()

        observations, targets = zip(*batch)
        observations = torch.from_numpy(np.array(observations)).float()
        target_values, target_policies = zip(*targets)
        target_values = torch.from_numpy(np.array(target_values)).unsqueeze(1).float()
        target_policies = torch.from_numpy(np.array(target_policies)).float()

        target_values = self._scalar_to_support(target_values)

        policy_logits, value_logits = self.network.inference(observations)
        # print(policy_logits, value_logits)

        p_loss = (-(target_policies * F.log_softmax(policy_logits, dim=1))).sum(dim=1).mean()
        v_loss = (-(target_values * F.log_softmax(value_logits, dim=1))).sum(dim=1).mean()

        self.optimizer.zero_grad()
        total_loss = p_loss + v_loss
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        return p_loss.item(), v_loss.item()

    def _scalar_to_support(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Reduce scaling
        eps = self.config.support_eps
        scaled_x = x.sign() * ((x.abs() + 1).sqrt() - 1) + eps * x
        scaled_x.clamp_(self.config.min_v, self.config.max_v)

        b = (scaled_x - self.config.min_v) / (self.config.delta_z)  # どのインデックスになるか
        lower_index, upper_index = b.floor().long(), b.ceil().long()  # インデックスを整数値に変換
        # l = u = bの場合インデックスをずらす
        lower_index[(upper_index > 0) * (lower_index == upper_index)] -= 1  # lを1減らす
        upper_index[(lower_index < (self.config.atoms - 1)) * (lower_index == upper_index)] += 1  # uを1増やす
        lower_probs = upper_index - b
        upper_probs = b - lower_index

        logits = torch.zeros(batch_size, self.config.atoms)
        logits.scatter_(dim=1, index=lower_index, src=lower_probs)
        logits.scatter_(dim=1, index=upper_index, src=upper_probs)
        return logits

    def _calc_win_rate(self):
        win_b_cnt = 0
        win_w_cnt = 0
        draw_cnt = 0

        replay_temp = self.replay.buffer[-self.config.calc_rate_size :]
        for game in replay_temp:
            if game.winner is None:
                draw_cnt += 1
            elif game.winner == 0:
                win_b_cnt += 1
            else:
                win_w_cnt += 1

        replay_size = len(replay_temp)
        win_b_rate = win_b_cnt / replay_size
        win_w_rate = win_w_cnt / replay_size
        draw_rate = draw_cnt / replay_size

        return win_b_rate, win_w_rate, draw_rate
