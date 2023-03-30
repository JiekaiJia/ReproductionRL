import json
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from utils import compute_returns
from common_cls.trainer import Trainer


# network
class ActorNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        nn.Module.__init__(self)

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        x = self.linear3(x)

        return x


class REINFORCE(Trainer):
    def __init__(self, env: gym.Env, config: dict):
        Trainer.__init__(self, env, config, self.__class__.__name__)

        self.actor_model = ActorNet(input_dim=self.obs_dim, hidden_dim=config["hidden_dim"], output_dim=self.action_dim)
        self.actor_model.to(config["device"])

        self.optimizer = Adam(params=self.actor_model.parameters(), lr=config["lr"])

    def train(self):
        env = self.env
        env.reset(seed=self.seed)
        actor_model = self.actor_model
        cfg = self.cfg
        actor_model.eval()
        obs_list, new_obs_list, r_list, a_list = [], [], [], []
        stat_dict = {"episode_reward": [], "loss": [], "episode_length": []}
        for i in range(cfg["num_epochs"]):
            terminated, truncated = False, False
            obs_list.clear()
            new_obs_list.clear()
            r_list.clear()
            a_list.clear()
            obs, _ = env.reset()
            obs_tensor = torch.from_numpy(obs).to(cfg["device"])
            obs_tensor = obs_tensor.reshape([1, -1])
            episode_reward = 0
            steps = 0
            # collect one episode experience
            with torch.no_grad():
                while not (terminated or truncated):
                    logits = actor_model(obs_tensor)
                    action = self.sample_action(logits)
                    obs, r, terminated, truncated, _ = env.step(int(action))
                    episode_reward += r

                    new_obs_tensor = torch.from_numpy(obs).to(cfg["device"]).reshape([1, -1])
                    r_tensor = torch.from_numpy(np.asarray(r)).to(cfg["device"])
                    a_tensor = torch.from_numpy(action).to(cfg["device"]).reshape([1, -1])

                    obs_list.append(obs_tensor)
                    new_obs_list.append(new_obs_tensor)
                    r_list.append(r_tensor)
                    a_list.append(a_tensor)

                    obs_tensor = new_obs_tensor
                    steps += 1
                stat_dict["episode_reward"].append(episode_reward)
                stat_dict["episode_length"].append(steps)
                ret_list = compute_returns(r_list, cfg["gamma"])
                ret_list = [ret.reshape([1, -1]) for ret in ret_list]
            # update network
            actor_model.train()
            obs_batch = torch.concat(obs_list, dim=0)
            a_batch = torch.concat(a_list, dim=0)
            ret_batch = torch.concat(ret_list, dim=0)
            net_output = actor_model(obs_batch)
            if self.action_dist_cls == torch.distributions.Categorical:
                action_dist = self.action_dist_cls(logits=net_output)
                log_probs = action_dist.log_prob(a_batch.squeeze(1)).unsqueeze(1)
            elif self.action_dist_cls == torch.distributions.Normal:
                mu, log_std = torch.chunk(net_output, 2, 1)
                log_std = torch.clamp(log_std, -2, 5)
                std = torch.exp(log_std)
                action_dist = self.action_dist_cls(loc=mu, scale=std)
                log_probs = action_dist.log_prob(a_batch)

            loss = - torch.mean(ret_batch * log_probs, dim=0)
            stat_dict["loss"].append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.tb_writer.add_scalar("loss", loss.item(), i+1)
            self.tb_writer.add_scalar("episode_reward", episode_reward, i+1)

            if (i+1) % cfg["report_interval"] == 0:
                self.logger.info(f"----------epoch {i + 1}-------------")
                self.logger.info(f"episode reward: {episode_reward}, loss: {loss.item()}, episode_length: {steps}")

            if (i+1) % cfg["checkpoint_interval"] == 0:
                torch.save(actor_model.state_dict(), os.path.join(self.log_dir, f"checkpoint-{i+1}.pt"))
                torch.save(actor_model, os.path.join(self.log_dir, f"model-{i + 1}.pt"))

        with open(os.path.join(self.log_dir, "stat_results.json"), "w") as f:
            f.write(json.dumps(stat_dict))

    def evaluate(self):
        env = self.env
        cfg = self.cfg
        actor_model = self.actor_model
        while True:
            terminated, truncated = False, False
            obs, _ = env.reset()
            obs_tensor = torch.from_numpy(obs).to(cfg["device"]).reshape([1, -1])
            with torch.no_grad():
                while not (terminated or truncated):
                    logits = actor_model(obs_tensor)
                    action = self.sample_action(logits)
                    obs, r, terminated, truncated, _ = env.step(int(action))
                    obs_tensor = torch.from_numpy(obs).to(cfg["device"]).reshape([1, -1])
                    # if cfg["render"]:
                    #     env.render()

    def restore_from_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.actor_model.load_state_dict(state_dict)


config = {
    "seed": 2023,
    "num_epochs": 10,
    "lr": 1e-4,
    "gamma": 0.99,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "hidden_dim": 256,
    "report_interval": 100,
    "checkpoint_interval": 200,
}


if __name__ == "__main__":
    # exp_env = gym.make("Pendulum-v1")
    exp_env = gym.make('CartPole-v1')

    trainer = REINFORCE(config=config, env=exp_env)
    trainer.train()

    # exp_env = gym.make('CartPole-v1', render_mode="human")
    # trainer = REINFORCE(config=config, env=exp_env)
    # trainer.evaluate()




