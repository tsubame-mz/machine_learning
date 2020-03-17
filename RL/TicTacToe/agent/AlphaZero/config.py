import torch


class AlphaZeroConfig:
    def __init__(self):
        # 乱数のシード
        self.seed = 0

        # Game
        self.obs_space = (3, 3, 3)
        self.action_space = 9
        self.terminate_value = 10

        # MCTS
        self.simulation_num = 100
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Network
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")  # GT740じゃもうダメだ
        self.num_channels = 8
        self.fc_hid_num = 16
        self.fc_output_num = 9
        self.model_file = "alphazero_model.pth"

        # Value support
        self.min_v = -2.5
        self.max_v = +2.5
        self.support_size = 25
        self.support_eps = 0.001
        self.atoms = self.support_size * 2 + 1
        self.delta_z = (self.max_v - self.min_v) / (self.atoms - 1)
        self.support_base = torch.linspace(self.min_v, self.max_v, self.atoms)
        # print(self.support_base)

        # SelfPlay
        self.self_play_num = 10
        self.discount = 0.95

        # Replay
        self.replay_buffer_size = 1000
        self.batch_size = 128

        # Training
        self.lr = 1e-2
        self.weight_decay = 1e-6
        self.max_training_step = 10000
        self.validate_interval = 100

        # Validate
        self.calc_rate_size = 100  # 直近のReplayから勝率を計算
