import json
import os
import shutil
import sys
from argparse import ArgumentParser
from collections import deque
# from d4rl.pointmaze.maze_model import LARGE_OPEN

import h5py
import tqdm
from matplotlib import cm
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from matplotlib import pyplot as plt

from nav2d_representation.models import SingleStep, MultiStep
from nav2d_representation.nav2d.nav2d import Navigate2D

np.set_printoptions(threshold=10000)
import torch.nn.functional as F
import random
from nav2d_representation.utils import ENV_DICT
from nav2d_representation.nav2d.utils import perturb_heatmap, return_encoded_vector

def main(args):
    tensorboard = SummaryWriter(args.datadir)
    dataset = h5py.File(args.datadir+"/embeddings.hdf5", "r")
    dataset_keys = []
    dataset.visit(
        lambda key: dataset_keys.append(key)
        if isinstance(dataset[key], h5py.Dataset)
        else None
    )


    num_embeddings = len(dataset['obs'])
    true_dist = [] #np.zeros((num_embeddings, num_embeddings))
    pattern_dist = []# np.zeros((num_embeddings, num_embeddings))
    embed_dist = [] #np.zeros((num_embeddings, num_embeddings))
    count = 0
    models = ['ms_enc', 'ss_enc']
    norm_orders = [1]
    tesselation = [(0,0), (1,1), (2,2),(3,3), (4,4)]
    # rs = dataset.attrs['row_stride']
    # cs = dataset.attrs['col_stride']
    global_step = 0
    
    for i in range(num_embeddings):
        distance_img_single = np.zeros((20,20)) - 1
        distance_img_multi = np.zeros((20,20))
        for j in range(num_embeddings):
            single_step_dist = distance_measure(dataset['ss_enc'][i], dataset['ss_enc'][j])
            multi_step_dist =  distance_measure(dataset['ms_enc'][i], dataset['ms_enc'][j])

            distance_img_single[dataset['pos'][j][0], dataset['pos'][j][1]] = single_step_dist 
            distance_img_multi[dataset['pos'][j][0], dataset['pos'][j][1]] = multi_step_dist 


        filler = np.zeros(distance_img_single.shape)
        filler[distance_img_single==-1] = 1


        distance_img_single = normalize(distance_img_single)
        distance_img_multi = normalize(distance_img_multi)

        

        single_img = np.stack((filler,filler,distance_img_single))
        single_img = single_img[np.newaxis, :,:,:]
        single_img[0, 0, dataset['pos'][i][0], dataset['pos'][i][1]] = 1

        multi_img = np.stack((filler,filler,distance_img_multi))
        multi_img = multi_img[np.newaxis, :,:,:]
        multi_img[0, 0, dataset['pos'][i][0], dataset['pos'][i][1]] = 1



        tensorboard.add_images(
                "single_step", single_img, global_step
            )
        tensorboard.add_images(
                "multi_step", multi_img, global_step
            )

        tensorboard.add_images(
                "joint", np.concatenate((multi_img, single_img),axis=3), global_step
        )
        global_step += 1
        print(global_step)
        """
        for norm_order in norm_orders:
            true_dist = [] #np.zeros((num_embeddings, num_embeddings)) 
            pattern_dist = []# np.zeros((num_embeddings, num_embeddings))
            embed_dist = [] #np.zeros((num_embeddings, num_embeddings))
            label = []
            
            for i in range(num_embeddings):
                for j in range(num_embeddings):
                    # true_dist[i,j] = np.linalg.norm(dataset['pos'][i] - dataset['pos'][j], ord=norm_order)
                    # pattern_dist[i,j] = find_pattern_distance(dataset['pos'][i], dataset['pos'][j], 
                    #                                         dataset.attrs['row_stride'],dataset.attrs['col_stride'], order=norm_order)
                    # embed_dist[i,j] = distance_measure(dataset[model][i], dataset[model][j])
                    pos1 = dataset['pos'][i]
                    pos2 = dataset['pos'][j]

                    pattern_pos1 = fpl(pos1, rs, cs)
                    pattern_pos2 = fpl(pos2, rs, cs)

                    pos1_tessel = (pos1[0] // rs, pos1[1] // cs)
                    pos2_tessel = (pos2[0] // rs, pos2[1] // cs)

                    for k in range(len(tesselation)):
                        if pos1_tessel == tesselation[0] and pos2_tessel == tesselation[k]:
                            true_dist.append(np.linalg.norm(pos1 - pos2, ord=norm_order))
                            pattern_dist.append(find_pattern_distance(pos1, pos2, rs, cs, order=norm_order))
                            embed_dist.append(distance_measure(dataset[model][i], dataset[model][j]))
                            label.append(i / len(tesselation))

                    # elif pos1_tessel == tesselation[0] and pos2_tessel == tesselation[2]:
                    #     true_dist.append(np.linalg.norm(pos1 - pos2, ord=norm_order))
                    #     pattern_dist.append(find_pattern_distance(pos1, pos2, rs, cs, order=norm_order))
                    #     embed_dist.append(distance_measure(dataset[model][i], dataset[model][j]))
                    #     label.append(0.5)
                    # elif pos1_tessel == tesselation[0] and pos2_tessel == tesselation[3]:
                    #     true_dist.append(np.linalg.norm(pos1 - pos2, ord=norm_order))
                    #     pattern_dist.append(find_pattern_distance(pos1, pos2, rs, cs, order=norm_order))
                    #     embed_dist.append(distance_measure(dataset[model][i], dataset[model][j]))
                    #     label.append(0.8)

            """
            # print('gere')
            # print(len(true_dist))
            # true_dist = np.array(true_dist)
            # pattern_dist = np.array(pattern_dist)
            # embed_dist = np.array(embed_dist)
            # label = np.array(label)
            # plt.figure()
            # plt.subplot(1,2,1)
            # plt.scatter(true_dist.flatten(), embed_dist.flatten(), c=label.flatten())
            # plt.xlabel('True Distance between agents')
            # plt.subplot(1,2,2)
            # plt.scatter(pattern_dist.flatten(), embed_dist.flatten(),  c=label.flatten())
            # plt.xlabel('Pattern distance between agents')
            # plt.savefig(args.datadir+'/' + model + '_distmetric_' + str(norm_order) + '_distance_results.png')

def distance_measure(embed1, embed2, order=2):
    return np.linalg.norm(embed1-embed2, ord=order)

def fpl(pos, rs, cs): # find pattern location
    return np.array([pos[0] % rs, pos[1] % cs])


def normalize(vec):
    return (vec - np.min(vec)) / (np.max(vec) - np.min(vec))

def find_pattern_distance(pos1, pos2, row_stride, col_stride, order=2):
    diff = pos1 - pos2
    diff[0] = diff[0] % row_stride
    diff[1] = diff[1] % col_stride
    return np.linalg.norm(diff,ord=order)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--datadir", type=str, required=True)
    args = parser.parse_args()

    main(args)