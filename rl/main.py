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
from rl.policies.policies import BisimCNN
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
from stable_baselines3.common.env_util import make_vec_env
from rl.utils import WandbEvalCallback

import copy
import wandb


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):

    
    env = Navigate2D(cfg.env.num_obstacles, grid_size=cfg.env.grid_size, 
                                static_goal=True,
                                obstacle_diameter=cfg.env.obstacle_diameter,)
    
    eval_env = copy.copy(env)
    env_kwarg_dict = dict(num_obstacles=cfg.env.num_obstacles, grid_size=cfg.env.grid_size, static_goal=True, obstacle_diameter=cfg.env.obstacle_diameter)

    # train_env = make_vec_env(Navigate2D, env_kwargs=env_kwarg_dict, n_envs=4, seed=0, vec_env_cls=SubprocVecEnv)
    # eval_env = make_vec_env(Navigate2D, env_kwargs=env_kwarg_dict, n_envs=4, seed=0, vec_env_cls=SubprocVecEnv)
    train_env = env
    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.name if len(cfg.name) > 0 else None, 
                    dir=cfg.logdir, entity=cfg.wandb_entity,
                    config=OmegaConf.to_container(cfg, resolve=True))
        
        eval_callback = WandbEvalCallback(eval_env, best_model_save_path=cfg.logdir,
                                        log_path=cfg.logdir, eval_freq=cfg.eval_freq,
                                        n_eval_episodes=50, deterministic=True,
                                        render=False, render_freq=cfg.render_freq)
    else:
        eval_callback = EvalCallback(eval_env, best_model_save_path=cfg.logdir,
                                    log_path=cfg.logdir, eval_freq=cfg.eval_freq,
                                    n_eval_episodes=5, deterministic=True,
                                    render=False)

    policy_kwargs = dict(features_extractor_class=BisimCNN, features_extractor_kwargs=cfg.encoder_args,)
    model = DQN(DQNPolicy, train_env, policy_kwargs=policy_kwargs, verbose=1, **cfg.rl)
    model.learn(total_timesteps=cfg.train_steps, log_interval=4, callback=eval_callback)
    

if __name__ == "__main__":
    main()
