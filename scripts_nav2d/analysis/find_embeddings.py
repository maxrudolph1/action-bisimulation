import json
import os
import shutil
import sys
from argparse import ArgumentParser
from collections import deque
# from d4rl.pointmaze.maze_model import LARGE_OPEN
import time
import h5py
from tqdm import tqdm
from matplotlib import cm
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

from nav2d_representation.models import SingleStep, MultiStep
from nav2d_representation.nav2d.nav2d import Navigate2D

np.set_printoptions(threshold=10000)
import torch.nn.functional as F
import random
from nav2d_representation.utils import ENV_DICT
from nav2d_representation.nav2d.utils import perturb_heatmap, return_encoded_vector
import time


def change_obs(obs, pos):
    obs[1, pos[0], pos[1]] = 1
    return obs

def train(args):

    log_path = args.logdir

    dataset = h5py.File(args.dataset, "r")
    dataset_keys = []
    dataset.visit(
        lambda key: dataset_keys.append(key)
        if isinstance(dataset[key], h5py.Dataset)
        else None
    )

    attrs = dict(dataset.attrs)
    #attrs['env_config'] = 'configs/simple_barrier.yaml'
    env_name = attrs.pop("env")
    env = ENV_DICT[env_name](**attrs)
    action_dim = env.action_space.n
    obs = env.reset()
    obs_shape = obs.shape
    #obs[1,:,:] = -1
    # print(obs[0,:,:] + 1)
    # print("Observation shape: ", obs_shape)
    # print("Action dim: ", action_dim)

    ss_model = torch.load(args.logdir + '/single_step_final.pt').cuda()
    ms_model = torch.load(args.logdir + '/multi_step_final.pt').cuda()

    # print(ss_model.encoder)

    global_step = 0
    h,w = obs[0,:,:].shape
    # print(obs + 1)
    #obs[0, :, :] = np.random.randint(-1, 1, (h,w))

    data_list = []
    total = h * w
    count = 0

    pattern = args.name.split('_')

    outer_obss = []
    inner_obss = []
    serial=False
    if ('random' in pattern[0]) and ('random' in pattern[1]):
    # random inner and outer
        num_inner_samples = 400
        num_outer_samples = 400 

        for i in range(num_inner_samples):
            io = np.random.randint(0,2, (5,5))
            io[2,2] = 0
            inner_obss.append(io)

        n_obs = 15
        d_obs = 2
        size = 20
        for i in range(num_outer_samples):
            temp_obs = np.zeros((size, size), dtype=np.float32)
            for i in range(n_obs):
                center = np.random.randint(0, 20, 2)
                minX = np.maximum(center[0] - d_obs, 0)
                minY = np.maximum(center[1] - d_obs, 0)
                maxX = np.minimum(center[0] + d_obs, size)
                maxY = np.minimum(center[1] + d_obs, size)
                temp_obs[minX:maxX, minY:maxY] = 1.0
            outer_obss.append(temp_obs)
        serial=True
        pattern[0] = 'rnull'
        pattern[1] = 'rnull'
    if 'line' in pattern[0]:
        lines = [0,1,3,4]
        inner_obss = []
        for k in range(len(lines)):
            for j in range(k, len(lines)):
                io = np.zeros((5,5))
                io[lines[k],:] = 1
                io[lines[j], :] = 1
                inner_obss.append(io)
        inner_obss.extend([ob.T for ob in inner_obss])
    elif 'block2' in pattern[0]:
        inner_obss = []
        patts = np.zeros((4, 5,5))
        patts[0, 0:2,0:2] = 1
        patts[1, 3:,3:] = 1 
        patts[2, 0:2,3:] = 1
        patts[3, 3:,0:2] = 1
        
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    for d in range(2):
                        io = patts[0] * a + patts[1] * b + patts[2] * c + patts[3] * d
                        inner_obss.append(io)
    elif 'random' in pattern[0]:
        num_inner_samples = 400
        inner_obss = []
        for i in range(num_inner_samples):
            io = np.random.randint(0,2, (5,5))
            io[2,2] = 0
            inner_obss.append(io)
    elif 'empty' in pattern[0]:
        num_innter_samples = 1
        inner_obss = [np.zeros((5,5)) for i in range(num_innter_samples)]
    elif 'none' in pattern[0]:
        num_innter_samples = 1
        inner_obss = [np.zeros((1,1)) for i in range(num_innter_samples)]


    

    if 'static' in pattern[1]:
        n_obs = 15
        d_obs = 2
        size = 20
        temp_obs = np.zeros((size, size), dtype=np.float32)
        for i in range(n_obs):
            center = np.random.randint(0, 20, 2)
            minX = np.maximum(center[0] - d_obs, 0)
            minY = np.maximum(center[1] - d_obs, 0)
            maxX = np.minimum(center[0] + d_obs, size)
            maxY = np.minimum(center[1] + d_obs, size)
            temp_obs[minX:maxX, minY:maxY] = 1.0
        outer_obss.append(temp_obs)
    elif 'random' in pattern[1]:
        num_outer_samples = 400
        n_obs = 15
        d_obs = 2
        size = 20
        for i in range(num_outer_samples):
            temp_obs = np.zeros((size, size), dtype=np.float32)
            for i in range(n_obs):
                center = np.random.randint(0, 20, 2)
                minX = np.maximum(center[0] - d_obs, 0)
                minY = np.maximum(center[1] - d_obs, 0)
                maxX = np.minimum(center[0] + d_obs, size)
                maxY = np.minimum(center[1] + d_obs, size)
                temp_obs[minX:maxX, minY:maxY] = 1.0

            outer_obss.append(temp_obs)
    elif 'single' in pattern[1]:
        tot = h * w
        size = h
        idx = np.unravel_index(np.arange(tot), (h,w))
        n_obs = 10
        d_obs = 2

        base_obs = np.zeros((size, size), dtype=np.float32)
        for i in range(n_obs):
            center = np.random.randint(0, 20, 2)
            minX = np.maximum(center[0] - d_obs, 0)
            minY = np.maximum(center[1] - d_obs, 0)
            maxX = np.minimum(center[0] + d_obs, size)
            maxY = np.minimum(center[1] + d_obs, size)
            base_obs[minX:maxX, minY:maxY] = 1.0

        base_obs[7:13, 7:13] = 0
             
        print(base_obs)
        for i in range(tot):

            temp_obs = np.copy(base_obs)
            temp_obs[idx[0][i], idx[1][i]] = 1
        
            outer_obss.append(temp_obs)

        serial=True
        inner_obss = [np.zeros((1,1)) for _ in range(len(outer_obss))]

    
    num_inners = len(inner_obss)
    num_outers= len(outer_obss)

    if serial:
        for i in tqdm(range(num_outers)):


            center_obs = (9,9) #np.random.randint(2, 17, (2,))
            obs = np.zeros((3, size, size), dtype=np.float32)
            obs[0, :, :] = outer_obss[i]
        
            inner_obs_h, inner_obs_w = inner_obss[i].shape
            inner_obs_h_half = inner_obs_h // 2
            inner_obs_w_half = inner_obs_w // 2
            obs[0, :, :] = outer_obss[i]
            
            obs[0, (center_obs[0]-inner_obs_h_half):(center_obs[0]+inner_obs_h_half+1),
                    (center_obs[1]-inner_obs_w_half):(center_obs[1]+inner_obs_w_half+1)] = inner_obss[i]
            obs[1,:,:] = 0
            obs[1,center_obs[0], center_obs[1]] = 1

            model_obs = env._get_obs(obs)
            ss_enc = return_encoded_vector(obs, ss_model.encoder)
            ms_enc = return_encoded_vector(obs, ms_model.encoder)
            # print(obs[0,:,:])
            data_list.append({'obs': obs, 'ss_enc': ss_enc, 'ms_enc': ms_enc, 'pos': np.array(center_obs),'inner_obs': inner_obss[i]})
    else:

        for i in tqdm(range(num_outers)):
            for inner_obs in inner_obss:

                center_obs = (9,9) #np.random.randint(2, 17, (2,))
                obs = np.zeros((3, h, w), dtype=np.float32)
                obs[0, :, :] = outer_obss[i]

                inner_obs_h, inner_obs_w = inner_obs.shape
                inner_obs_h_half = inner_obs_h // 2
                inner_obs_w_half = inner_obs_w // 2

                obs[0, (center_obs[0]-inner_obs_h_half):(center_obs[0]+inner_obs_h_half+1),
                       (center_obs[1]-inner_obs_w_half):(center_obs[1]+inner_obs_w_half+1)] = inner_obs
                obs[1,:,:] = 0
                obs[1,center_obs[0], center_obs[1]] = 1

                model_obs = env._get_obs(obs)
                
                ss_enc = return_encoded_vector(model_obs, ss_model.encoder)
                ms_enc = return_encoded_vector(model_obs, ms_model.encoder)
                
                data_list.append({'obs': obs, 'ss_enc': ss_enc, 'ms_enc': ms_enc, 'pos': np.array(center_obs),'inner_obs': inner_obs})


    with h5py.File(
        f"{args.logdir}/embeddings/{args.name}.hdf5", "w"
    ) as f:
        f["obs"] = np.array([x['obs'] for x in data_list])
        f["ss_enc"] = np.array([x['ss_enc'] for x in data_list])
        f["ms_enc"] = np.array([x['ms_enc'] for x in data_list])
        f["pos"] = np.array([x['pos'] for x in data_list])
        f["inner_obs"] = np.array([x['inner_obs'] for x in data_list])
        f["num_samples"] = np.array([num_inners * num_outers for x in data_list])
        # f["num_locations"] = np.array([num_locations for x in data_list])
        # f.attrs["row_stride"] =  block.shape[0]
        # f.attrs["col_stride"] = block.shape[1]

    





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--name", type=str, default="")
    # parser.add_argument("--single_step_path", type=str, default=None)


    args = parser.parse_args()

    train(args)
