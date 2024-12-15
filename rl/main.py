import argparse
import datetime
import os
import pprint
import yaml
import numpy as np
import torch
# from atari_network import DQN
# from atari_wrapper import make_atari_env
from copy import deepcopy
import gymnasium as gym
from environments.nav2d.nav2d_sb3 import Navigate2D
from models import nets, gen_model_nets
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import DQN
from omegaconf import DictConfig, OmegaConf
import hydra
from stable_baselines3.dqn.policies import DQNPolicy
from rl.policies.policies import BisimCNN, linear_schedule
from wandb.integration.sb3 import WandbCallback
import wandb


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    
    env = Navigate2D(cfg.env.num_obstacles, grid_size=cfg.env.grid_size, 
                                static_goal=True,
                                obstacle_diameter=cfg.env.obstacle_diameter,)

    # policy = DQNPolicy(env.observation_space, env.action_space, lr_schedule=lr_schedule, features_extractor_class=BisimCNN, features_extractor_kwargs=cfg.encoder_args,)
    # wandb.init(project="nav2d", config=cfg)

    run = wandb.init(
        project="sb3",
        config=cfg,
        entity="maxrudolph"
    )
    policy_kwargs = dict(features_extractor_class=BisimCNN, features_extractor_kwargs=cfg.encoder_args,)
    model = DQN(DQNPolicy, env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=1000000, log_interval=4, callback=WandbCallback())
    model.save("nav2d")
    

if __name__ == "__main__":
    main()
