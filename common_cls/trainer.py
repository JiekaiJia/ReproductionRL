from datetime import datetime
import os
from typing import Dict

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.tensorboard import SummaryWriter

from common_cls.logger import Logger


class Trainer:
    def __init__(self, env: gym.Env, cfg: Dict, algorithm_name: str):
        self.logger = Logger(__name__)
        self.logger.add_stream_handler()
        timestep = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        logfile = f"{algorithm_name}-{timestep}"
        log_dir = os.path.join(os.getcwd(), logfile)
        self.tb_writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        self.seed = cfg.get("seed")

        self.env = env
        self.cfg = cfg

        if isinstance(env.observation_space, Box):
            self.obs_dim = np.prod(env.observation_space.shape)
            self.logger.info(f"The observation space {env.observation_space} is continuous "
                             f"and the observation dimension is {self.obs_dim}.")
        elif isinstance(env.observation_space, Discrete):
            self.obs_dim = env.observation_space.n
            self.logger.info(f"The observation space {env.observation_space} is discrete "
                             f"and the observation dimension is {self.obs_dim}.")

        if isinstance(env.action_space, Box):
            self.action_dim = 2 * np.prod(env.action_space.shape)
            self.action_dist_cls = torch.distributions.Normal
            self.logger.info(f"The action space is {env.action_space} continuous "
                             f"and the action dimension is {self.action_dim}.")
        elif isinstance(env.action_space, Discrete):
            self.action_dim = env.action_space.n
            self.action_dist_cls = torch.distributions.Categorical
            self.logger.info(f"The action space is {env.action_space} discrete "
                             f"and the action dimension is {self.action_dim}.")

    def train(self):
        return NotImplementedError

    def evaluate(self):
        return NotImplementedError

    def sample_action(
            self,
            net_output: torch.Tensor,
            is_deterministic: bool = False
            ) -> NDArray:

        if len(net_output.shape) == 1:
            net_output = net_output.unsqueeze(0)

        # continuous action case
        if self.action_dist_cls == torch.distributions.Normal:
            mu, log_std = torch.chunk(net_output, 2, 1)
            log_std = torch.clamp(log_std, -2, 5)
            std = torch.exp(log_std)
            action_dist = self.action_dist_cls(loc=mu, scale=std)
            if is_deterministic:
                action = mu
            else:
                action = action_dist.rsample()
            # convert tensor to ndarray.
            action = action.detach().cpu().numpy()
        # discrete action case
        elif self.action_dist_cls == torch.distributions.Categorical:
            action_dist = self.action_dist_cls(logits=net_output)
            if is_deterministic:
                action = torch.max(action_dist.probs, dim=1)
            else:
                action = action_dist.sample()
            # convert tensor to ndarray.
            action = action.detach().cpu().numpy()

        return action
