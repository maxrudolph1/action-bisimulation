import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from nav2d_representation.utils import ENV_DICT, grad_heatmap
from nav2d_representation.nav2d.nav2d import Navigate2D
from nav2d_representation.models import SingleStep, MultiStep
from nav2d_representation.nav2d.utils import perturb_heatmap, return_encoded_vector
from nav2d_representation.utils import ENV_DICT

import h5py



# models = ['/nfs/data/mrudolph/nav2d_models/default_model_gamma_0_99',
#             '/nfs/data/mrudolph/nav2d_models/default_model_gamma_0_9',
#             '/nfs/data/mrudolph/nav2d_models/default_model_gamma_0_5',
#             '/nfs/data/mrudolph/nav2d_models/default_model_gamma_0_7',
#             '/nfs/data/mrudolph/nav2d_models/default_model_gamma_0_5_wd']

models = ['/home/mrudolph/documents/actbisim/scripts_nav2d/good_ms_models/gamma_0.99_gt_forward_dc_2']
            
            
for model_dir in models:

    ss_model = torch.load(model_dir + '/single_step_final.pt').cuda()
    ms_model = torch.load(model_dir + '/multi_step_final.pt').cuda()


    dataset_path = '/home/mrudolph/documents/actbisim/scripts_nav2d/nav2d_dataset_s0_e0.5_size1000_test.hdf5'
    dataset_path = '/home/mrudolph/documents/actbisim/scripts_nav2d/datasets/nav2d_dataset_s0_e0.5_size1000_tiny_k_steps_10.hdf5'
    dataset = h5py.File(dataset_path, "r")
    dataset_keys = []
    dataset.visit(
        lambda key: dataset_keys.append(key)
        if isinstance(dataset[key], h5py.Dataset)
        else None
    )
    
    obs = dataset['obs'][0]

    maps = grad_heatmap(obs, ms_model.encoder)

    # print(type(obs))
    # print(ms_model.encoder)
    # imshow the heatmaps and save them in a file
    fig, ax = plt.subplots(2, 2)
    # add some more space in between subplots
    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.5)
    ax[0, 1].imshow(maps[0])
    ax[0, 1].set_title('Agent Heatmap')
    
    ax[0, 0].imshow(maps[1])
    ax[0, 0].set_title('Obstacle Heatmap')
    
    ax[1, 0].imshow(obs[0])
    ax[1, 0].set_title('Obstacles')
    
    
    ax[1, 1].imshow(obs[1])
    ax[1, 1].set_title('Agent')
    
    plt.suptitle(f'Gamma: {model_dir[-5:]}')
    print(ms_model.encoder)
    # set_set_title each subplot with a,b,c,d
    plt.savefig(f'model_heatmaps/heatmap_{model_dir[-5:]}.png')