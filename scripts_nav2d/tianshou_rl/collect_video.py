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
import cv2
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="log")
    # parser.add_argument("--task", type=str, default="Nav2D-v0")
    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--scale-obs", type=int, default=0)
    # parser.add_argument("--eps-test", type=float, default=0.005)
    # parser.add_argument("--eps-train", type=float, default=1.)
    # parser.add_argument("--eps-train-final", type=float, default=0.05)
    # parser.add_argument("--buffer-size", type=int, default=100000)
    # parser.add_argument("--lr", type=float, default=0.00001)
    # parser.add_argument("--gamma", type=float, default=0.99)
    # parser.add_argument("--n-step", type=int, default=3)
    # parser.add_argument("--target-update-freq", type=int, default=500)
    # parser.add_argument("--epoch", type=int, default=100)
    # parser.add_argument("--step-per-epoch", type=int, default=10000)
    # parser.add_argument("--step-per-collect", type=int, default=10)
    # parser.add_argument("--update-per-step", type=float, default=0.1)
    # parser.add_argument("--batch-size", type=int, default=32)
    # parser.add_argument("--training-num", type=int, default=10)
    # parser.add_argument("--test-num", type=int, default=10)
    # parser.add_argument("--logdir", type=str, default="log")
    # parser.add_argument("--render", type=float, default=0.)
    # parser.add_argument(
    #     "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    # )
    # parser.add_argument("--frames-stack", type=int, default=1)
    # parser.add_argument("--resume-path", type=str, default=None)
    # parser.add_argument("--resume-id", type=str, default=None)
    # parser.add_argument(
    #     "--logger",
    #     type=str,
    #     default="tensorboard",
    #     choices=["tensorboard", "wandb"],
    # )
    # parser.add_argument("--wandb-project", type=str, default="atari.benchmark")
    # parser.add_argument(
    #     "--watch",
    #     default=False,
    #     action="store_true",
    #     help="watch the play of pre-trained policy only"
    # )
    # parser.add_argument("--save-buffer-name", type=str, default=None)
    # parser.add_argument(
    #     "--icm-lr-scale",
    #     type=float,
    #     default=0.,
    #     help="use intrinsic curiosity module with this lr scale"
    # )
    # parser.add_argument(
    #     "--icm-reward-scale",
    #     type=float,
    #     default=0.01,
    #     help="scaling factor for intrinsic curiosity reward"
    # )
    # parser.add_argument(
    #     "--icm-forward-loss-weight",
    #     type=float,
    #     default=0.2,
    #     help="weight for the forward model loss in ICM"
    # )
    # parser.add_argument(
    #     "--use-her",
    #     default=False,
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "--sparse-reward",
    #     default=True,
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "--name",
    #     default="",
    # )
    return parser.parse_args()


def watch_dqn(args=get_args()):
    
    train_args = read_yaml_file(os.path.join(args.path, "args.yaml"))
    policy_path = os.path.join(args.path, "policy.pth")
    env_lambda = lambda: gym.make("Nav2D-v0", grid_size=train_args['grid_size'],
                                              num_obstacles=train_args['num_obstacles'], 
                                              her_obs=train_args['use_her'], 
                                              sparse_reward=train_args['sparse_reward'],
                                              max_timesteps=train_args['max_timesteps'],
                                              static_goal=train_args['static_goal'],)

    env = env_lambda()

    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    
    # should be N_FRAMES x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # define model
    net = nets.DQNHER(args.state_shape, args.action_shape,atoms=1,split_obs=train_args['use_her'], device=train_args['device']).to(device=train_args['device'])

    optim = torch.optim.Adam(net.parameters(), lr=train_args['lr'])
    # define policy
    policy = DQNPolicy(
        net,
        optim,
        train_args['gamma'],
        train_args['n_step'],
        target_update_freq=1,
    )


    policy.load_state_dict(torch.load(policy_path, map_location=train_args['device']))
    print("Loaded agent from: ", policy_path)
        
    net.load_state_dict(policy.model.state_dict())
    obs, _ = env.reset()

    # watch agent's performance

    print("Setup test envs ...")
    policy.eval()
    policy.set_eps(train_args['eps_test'])
    env.seed(args.seed)
    n_episodes = 5
    frames = []
    for i in range(n_episodes):
        done = False
        
        while not done:
            act = net(obs[np.newaxis,...])[0].argmax().item()
            obs, rew,  done , _, _ = env.step(act)
            frames.append(obs)
 
        env.reset()
        frames.extend([np.zeros_like(obs)] * 10)
    write_video_from_array_list(frames, "test.mp4", 10)
    



def write_video_from_array_list(array_list, output_file, fps):
    # Retrieve the shape of the first array in the list
    height, width = array_list[0].shape[1:]

    size = 512
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Choose the video codec
    out = cv2.VideoWriter(output_file, fourcc, fps, (size, size))

    # Scale and convert the arrays to uint8 type
    scaled_array_list = [np.transpose((255 * (array + 1) / 2).astype(np.uint8), (1,2,0))for array in array_list]
    # scaled_array_list = [np.array(Image.fromarray(array.astype('uint8')).resize((256,256))) for array in scaled_array_list]
    scaled_array_list = [
        cv2.resize(array, (size, size), interpolation=cv2.INTER_AREA) for array in scaled_array_list
    ]
    # Iterate over the scaled arrays and write each frame to the video
    for array in scaled_array_list:
        # Transpose the array to match the expected format
        frame = array
        out.write(frame)

    # Release the VideoWriter object and close the video file
    out.release()
    

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:

        yaml_data = yaml.safe_load(file)
    return yaml_data



if __name__ == "__main__":
    watch_dqn(get_args())
