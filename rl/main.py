import argparse
import datetime
import os
import sys
sys.path.append('/nfs/homes/bisim/rrai/action-bisimulation')
import pprint
import yaml
import numpy as np
import torch
# from atari_network import DQN
# from atari_wrapper import make_atari_env
from copy import deepcopy
from tianshou.data import Collector, VectorReplayBuffer, HERVectorReplayBuffer
from tianshou.policy import DQNPolicy
from tianshou.policy.modelbased.icm import ICMPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import WandbLogger
from tianshou.utils.net.discrete import IntrinsicCuriosityModule
import tianshou as ts
import gymnasium as gym
from environments.nav2d.nav2d import Navigate2D 
from models import nets, gen_model_nets
from torch.utils.tensorboard import SummaryWriter

from omegaconf import DictConfig, OmegaConf
import hydra

import wandb #for sweeping

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print("inside main")

    if wandb.run is not None:
        sweep_config = wandb.config
        cfg.env.grid_size = sweep_config.get("grid_size", cfg.env.grid_size)
        cfg.env.num_obstacles = sweep_config.get("num_obstacles", cfg.env.num_obstacles)
        cfg.env.obstacle_diameter = sweep_config.get("obstacle_diameter", cfg.env.obstacle_diameter)

    print(f"Grid Size: {cfg.env.grid_size}")
    print(f"Number of Obstacles: {cfg.env.num_obstacles}")
    print(f"Obstacle Diameter: {cfg.env.obstacle_diameter}")
    
    # grid_size = int(cfg.env.grid_size)
    # num_obstacles = int(cfg.env.num_obstacles)
    # obstacle_diameter = int(cfg.env.obstacle_diameter)
    # print(f"Grid Size: {grid_size}")
    # print(f"Number of Obstacles: {num_obstacles}")
    # print(f"Obstacle Diameter: {obstacle_diameter}")

    env_lambda = lambda: Navigate2D(cfg.env.num_obstacles, grid_size=cfg.env.grid_size, 
                                static_goal=True,
                                obstacle_diameter=cfg.env.obstacle_diameter,)

    env = env_lambda()

    train_envs = ts.env.DummyVectorEnv([env_lambda for _ in range(cfg.training_num)])
    test_envs = ts.env.DummyVectorEnv([env_lambda  for _ in range(10)])
    
    cfg.state_shape = env.observation_space.shape or env.observation_space.n
    cfg.action_shape = env.action_space.shape or env.action_space.n
    
    # should be N_FRAMES x H x W
    print("Observations shape:", cfg.state_shape)
    print("Actions shape:", cfg.action_shape)
    
    # seed
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # define model

    # net = nets.DQNHER(cfg.state_shape, cfg.action_shape, args=cfg, atoms=1,split_obs=cfg.use_her, device=cfg.device, encoder_path=cfg.pretrained_encoder_path).to(cfg.device)
    net = gen_model_nets.GenDQNFull(cfg.state_shape, cfg.action_shape, cfg=cfg,).to(cfg.device)

    if cfg.freeze_encoder:
        optim = torch.optim.Adam(net.dqn.parameters(), lr=cfg.lr)
    else:
        optim = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    
    # optim = torch.optim.Adam( net.parameters(), lr=cfg.lr)
    # define policy
    policy = DQNPolicy(
        net,
        optim,
        cfg.gamma,
        cfg.n_step,
        target_update_freq=cfg.target_update_freq
    )

    if cfg.resume_path:
        policy.load_state_dict(torch.load(cfg.resume_path, map_location=cfg.device))
        print("Loaded agent from: ", cfg.resume_path)
        
        
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    if not cfg.use_her:
        buffer = VectorReplayBuffer(
            cfg.buffer_size,
            buffer_num=len(train_envs),
            ignore_obs_next=True,

        )
    else:
        def compute_reward_fn(ag, dg, goal_shape=None):
            ag = ag.reshape(goal_shape)
            dg = dg.reshape(goal_shape)
            
            rew = -np.ones((ag.shape[0], ag.shape[1]))
            for i in range(goal_shape[1]):
                ag_loc = np.where(ag[:,i,...] == 1)
                dg_loc = np.where(dg[:,i,...] == 1)
                # print(dg_loc)
                vec_x = ag_loc[2] == dg_loc[2]
                vec_y = ag_loc[3] == dg_loc[3]
                agdg = (vec_x & vec_y)
                # print(agdg)
                rew[agdg,i] = 0

            rew = -np.ones((ag.shape[0]* ag.shape[1],))
            # rew[agdg] = 0
   
            return rew
        
        buffer = HERVectorReplayBuffer(
            cfg.buffer_size,
            len(train_envs),
            compute_reward_fn=compute_reward_fn,
            horizon=env.max_timesteps,
            future_k=2,
        )
    # collector
    
    def preprocess_her_fn(**kwargs):
        in_reset = kwargs.keys() == {"obs", "info", "env_id"}
        if in_reset:
            return kwargs
        buffer = kwargs
        her_buffer = deepcopy(kwargs)
        final_obs = her_buffer["obs_next"][-1]
        fake_goal_grid = final_obs[1, :, :]
        fake_goal_pos = np.where(fake_goal_grid == np.max(fake_goal_grid))
        
        
        her_buffer["obs_next"][:, 2,:,:] = fake_goal_grid

        # Finds where the agent reaches the new goal the first time
        reach_goal_idx = np.where([np.all(obs[1,:,:] == obs[2,:,:]) for obs in her_buffer["obs_next"][:]])[0][0]
        
        # keep only experience up to the point where the agent reaches the new goal
        for ele in her_buffer:
            ele = ele[:reach_goal_idx+1]
            
        # update reward and termination
        her_buffer["rew"][-1] = 0
        her_buffer["done"][-1] = True
        
        # merge the original and HER transitions
        for key in buffer:
            if key == 'policy':
                info_keys = buffer[key].keys()
                for info_key in info_keys:
                    buffer[key][info_key] = np.concatenate([buffer[key][info_key], her_buffer[key][info_key]], axis=0)
            else:
                buffer[key] = np.concatenate([buffer[key], her_buffer[key]], axis=0)
        print(buffer["obs_next"].shape)
        for key in buffer:
            if key == 'policy':
                info_keys = buffer[key].keys()
                for info_key in info_keys:
                    buffer[key][info_key] = buffer[key][info_key][-10:]
            else:
                buffer[key] = buffer[key][-10:]
                
        return buffer
    
    
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    cfg.algo_name = "dqn_icm" if cfg.icm_lr_scale > 0 else "dqn"
    log_name = os.path.join(cfg.task, cfg.algo_name, str(cfg.seed), cfg.name + (("_" + now) if cfg.date else ""))
    log_path = os.path.join(cfg.logdir, log_name)

    
    # small test
    # wandb.init(
    #     project="nav2d",
    #     entity="rhearai-university-of-texas-at-austin", 
    #     config={
    #         "grid_size": 15,
    #         "num_obstacles": 10,
    #         "obstacle_diameter": 2,
    #         "epoch": 5
    #     }
    # )

    
    # logger
    if cfg.logger == "wandb":
        logger = WandbLogger(save_interval=1,name='test' ,  project=cfg.wandb_project, monitor_gym=False, entity=cfg.wandb_entity)
    
    # sweep variables from the sweep.yaml file, just making sure they're set correctly to be dynamic
    # cfg.env.grid_size = wandb.config.grid_size
    # cfg.env.num_obstacles = wandb.config.num_obstacles
    # cfg.env.obstacle_diameter = wandb.config.obstacle_diameter
    # cfg.epoch = wandb.config.get("epoch", cfg.epoch)


    writer = SummaryWriter(log_path)
    writer.add_text("cfg", str(cfg))

    logger.load(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        elif "Pong" in cfg.task:
            return mean_rewards >= 20
        else:
            return False

    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 1M steps
        linear_decay_time = 1e5
        if env_step <= linear_decay_time:
            eps = cfg.eps_train - env_step / linear_decay_time * \
                (cfg.eps_train - cfg.eps_train_final)
        else:
            eps = cfg.eps_train_final

        policy.set_eps(eps)
        if env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})

    def test_fn(epoch, env_step):
        policy.set_eps(cfg.eps_test)

    def dont_save_fn(epoch, env_step, gradient_step):
        return "not saved"
    
    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save({"model": policy.state_dict()}, ckpt_path)
        return ckpt_path

    # watch agent's performance
    def watch():
        print("Setup test envs ...")
        policy.eval()
        policy.set_eps(cfg.eps_test)
        test_envs.seed(cfg.seed)
        if cfg.save_buffer_name:
            print(f"Generate buffer with size {cfg.buffer_size}")
            buffer = VectorReplayBuffer(
                cfg.buffer_size,
                buffer_num=len(test_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=cfg.frames_stack
            )
            collector = Collector(policy, test_envs, buffer, exploration_noise=True)
            result = collector.collect(n_step=cfg.buffer_size)
            print(f"Save buffer into {cfg.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(cfg.save_buffer_name)
        else:
            print("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(
                n_episode=cfg.test_num, render=cfg.render
            )
        rew = result["rews"].mean()
        print(f"Mean reward (over {result['n/ep']} episodes): {rew}")

    if cfg.watch:
        watch()
        exit(0)
    cfg.state_shape = list(cfg.state_shape)
    with open(os.path.join(log_path, "cfg.yaml"), "w") as f:
        yaml.dump({}, f,)

    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=cfg.batch_size * cfg.training_num)
    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        cfg.epoch,
        cfg.step_per_epoch,
        cfg.step_per_collect,
        cfg.episode_per_test, # aka test_num
        cfg.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=cfg.update_per_step,
        test_in_train=False,
        resume_from_log=cfg.resume_id.lower() != 'none',
        save_checkpoint_fn=save_checkpoint_fn if cfg.save_models else dont_save_fn,
    )

    pprint.pprint(result)
    watch()


if __name__ == "__main__":
    print("got to main")
    main()