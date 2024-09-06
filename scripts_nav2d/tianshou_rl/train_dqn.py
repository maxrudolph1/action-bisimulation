import argparse
import datetime
import os
import pprint
import yaml
import numpy as np
import torch
# from atari_network import DQN
# from atari_wrapper import make_atari_env
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from tianshou.data import Collector, VectorReplayBuffer, HERVectorReplayBuffer
from tianshou.policy import DQNPolicy
from tianshou.policy.modelbased.icm import ICMPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.discrete import IntrinsicCuriosityModule
import tianshou as ts
import gymnasium as gym
import nav2d
from nav2d_representation import nets
# from nav2d_representation.nav2d.nav2d import Navigate2D



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Nav2D-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--eps-test", type=float, default=0.005)
    parser.add_argument("--eps-train", type=float, default=0.9)
    parser.add_argument("--eps-train-final", type=float, default=0.2)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=10000)
    parser.add_argument("--step-per-collect", type=int, default=100)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument("--date", default=False, action="store_true")
    parser.add_argument("--pretrained-encoder-path", type=str, default=None)
    
    # custom args
    parser.add_argument("--episode-per-test", type=int, default=100)
    parser.add_argument("--use-her", default=False, action="store_true")
    parser.add_argument("--dense-reward", default=False, action="store_true")
    parser.add_argument("--name",default="")
    # arg parse for max_timesteps, grid_size, num_obstacles, static_goal
    parser.add_argument("--max-timesteps", type=int, default=50)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--num-obstacles", type=int, default=0)
    parser.add_argument("--obstacle-size", type=int, default=1)
    parser.add_argument("--static-goal", default=False, action="store_true")
    parser.add_argument("--freeze-encoder", default=False, action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--frames-stack", type=int, default=1)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument("--save-models", default=False, action="store_true")

    # network arguments
    parser.add_argument('--encode-hidden-layers', type=int, nargs='+', default=[64,64,64,96,96,128])
    parser.add_argument('--encode-num-pooling', type=int, default=2)
    parser.add_argument('--encode-activation', default="relu")
    parser.add_argument('--encode-layer-norm', action="store_true")
    parser.add_argument('--DQN-hidden-layers', type=int, nargs='+', default=[256])
    parser.add_argument('--DQN-activation', default="relu")
    parser.add_argument('--DQN-layer-norm', action="store_true")
    parser.add_argument("--use-gen-nets", action="store_true")
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="atari.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only"
    )
    parser.add_argument("--save-buffer-name", type=str, default=None)
    parser.add_argument(
        "--icm-lr-scale",
        type=float,
        default=0.,
        help="use intrinsic curiosity module with this lr scale"
    )
    parser.add_argument(
        "--icm-reward-scale",
        type=float,
        default=0.01,
        help="scaling factor for intrinsic curiosity reward"
    )
    parser.add_argument(
        "--icm-forward-loss-weight",
        type=float,
        default=0.2,
        help="weight for the forward model loss in ICM"
    )

    return parser.parse_args()


def train_dqn(args=get_args()):
    env_lambda = lambda: gym.make("Nav2D-v0", grid_size=args.grid_size, 
                                              num_obstacles=args.num_obstacles, 
                                              static_goal=args.static_goal,
                                              her_obs=args.use_her, 
                                            #   sparse_reward=not args.dense_reward,
                                              max_timesteps=args.max_timesteps,)

    env = env_lambda()

    train_envs = ts.env.DummyVectorEnv([env_lambda for _ in range(args.training_num)])
    test_envs = ts.env.DummyVectorEnv([env_lambda  for _ in range(10)])
    
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # define model

    net = nets.DQNHER(args.state_shape, args.action_shape, args=args, atoms=1,split_obs=args.use_her, device=args.device, encoder_path=args.pretrained_encoder_path).to(args.device)

    if args.freeze_encoder:
        optim = torch.optim.Adam(net.dqn.parameters(), lr=args.lr)
    else:
        optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    
    # optim = torch.optim.Adam( net.parameters(), lr=args.lr)
    # define policy
    policy = DQNPolicy(
        net,
        optim,
        args.gamma,
        args.n_step,
        target_update_freq=args.target_update_freq
    )

    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)
        
        
    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    if not args.use_her:
        buffer = VectorReplayBuffer(
            args.buffer_size,
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
            args.buffer_size,
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
    args.algo_name = "dqn_icm" if args.icm_lr_scale > 0 else "dqn"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), args.name + (("_" + now) if args.date else ""))
    log_path = os.path.join(args.logdir, log_name)

    # logger
    if args.logger == "wandb":
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        elif "Pong" in args.task:
            return mean_rewards >= 20
        else:
            return False

    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 1M steps
        linear_decay_time = 1e5
        if env_step <= linear_decay_time:
            eps = args.eps_train - env_step / linear_decay_time * \
                (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final

        policy.set_eps(eps)
        if env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

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
        policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)
        if args.save_buffer_name:
            print(f"Generate buffer with size {args.buffer_size}")
            buffer = VectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(test_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=args.frames_stack
            )
            collector = Collector(policy, test_envs, buffer, exploration_noise=True)
            result = collector.collect(n_step=args.buffer_size)
            print(f"Save buffer into {args.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
            print("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(
                n_episode=args.test_num, render=args.render
            )
        rew = result["rews"].mean()
        print(f"Mean reward (over {result['n/ep']} episodes): {rew}")

    if args.watch:
        watch()
        exit(0)
    args.state_shape = list(args.state_shape)
    with open(os.path.join(log_path, "args.yaml"), "w") as f:
        yaml.dump(args, f, sort_keys=False, indent=4)

    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.episode_per_test, # aka test_num
        args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        save_best_fn=save_best_fn,
        logger=logger,
        update_per_step=args.update_per_step,
        test_in_train=False,
        resume_from_log=args.resume_id is not None,
        save_checkpoint_fn=save_checkpoint_fn if args.save_models else dont_save_fn,
    )

    pprint.pprint(result)
    watch()


if __name__ == "__main__":
    train_dqn(get_args())
