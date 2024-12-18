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


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
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
    net = gen_model_nets.GenDQNFull(cfg.state_shape, cfg.action_shape, cfg=cfg,).to(cfg.device)

    if cfg.freeze_encoder:
        optim = torch.optim.Adam(net.dqn.parameters(), lr=cfg.lr)
    else:
        optim = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    # define policy
    policy = DQNPolicy(
        net,
        optim,
        cfg.gamma,
        cfg.n_step,
        target_update_freq=cfg.target_update_freq
    )

    import pdb; pdb.set_trace()

    # if cfg.resume_path:
    #     policy.load_state_dict(torch.load(cfg.resume_path, map_location=cfg.device))
    #     print("Loaded agent from: ", cfg.resume_path)
        
        
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    if not cfg.use_her:
        buffer = VectorReplayBuffer(
            cfg.buffer_size,
            buffer_num=len(train_envs),
            ignore_obs_next=True,

        )

    
    
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    cfg.algo_name = "dqn_icm" if cfg.icm_lr_scale > 0 else "dqn"
    log_name = os.path.join(cfg.task, cfg.algo_name, str(cfg.seed), cfg.name + (("_" + now) if cfg.date else ""))
    log_path = os.path.join(cfg.logdir, log_name)

    # logger
    if cfg.logger == "wandb":
        logger = WandbLogger(save_interval=1,name='test' ,  project=cfg.wandb_project, monitor_gym=False, entity=cfg.wandb_entity)
        
    writer = SummaryWriter(log_path)
    writer.add_text("cfg", str(cfg))

    logger.load(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

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
    main()