import json
import os
import shutil
import sys
from argparse import ArgumentParser
from collections import deque
import time 
import h5py
import tqdm
from matplotlib import cm
import torch
import numpy as np
from nav2d_representation import utils

from nav2d_representation.models import *
from nav2d_representation.info_nce import KNCEStep

from omegaconf import DictConfig, OmegaConf
import hydra

import torch.nn.functional as F
import random
from nav2d_representation.utils import ENV_DICT
from environments.nav2d.utils import perturb_heatmap
import datetime
from nav2d_representation.models import SingleStep, MultiStep


from omegaconf import DictConfig, OmegaConf
import hydra

MODEL_DICT = {'single_step': SingleStep, 'multi_step': MultiStep}

def load_dataset(dataset_path):
    dataset = h5py.File(dataset_path, "r")
    dataset_keys = []
    dataset.visit(lambda key: dataset_keys.append(key) if isinstance(dataset[key], h5py.Dataset) else None)
    
    mem_dataset = {}
    for key in dataset_keys:
        mem_dataset[key] = dataset[key][:]
    dataset = mem_dataset
    
    act_shape = 7
    obs_shape = dataset["obs"][0].shape
    return dataset, dataset_keys, obs_shape, act_shape

def create_models(cfg: DictConfig):
    model_names = list(cfg.keys())
    models = {}
    for model_name in model_names:
        model_cfg = cfg[model_name]
        model = MODEL_DICT[model_cfg.model](**model_cfg.attrs)
        models[model_name] = model
    
def create_env(cfg: DictConfig):
    env = ENV_DICT[cfg.environment](**cfg.environment.attrs)
    return env

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    log_path = cfg.logdir + ("_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        
    if cfg.wandb: 
        import wandb
        wandb.init(entity='maxrudolph', project="nav2d", config=cfg)
        
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    dataset, dataset_keys, obs_shape, act_shape = load_dataset(cfg.dataset)
    models = create_models(cfg.algos)


if __name__ == "__main__":
    main()
    
def train(args):

    attrs = dict(dataset.attrs)

    env_name = attrs.pop("env")
    
    env = ENV_DICT[env_name](**attrs)
    action_dim = env.action_space.n
    obs_shape = dataset["obs"][0].shape
    
    if env_name == "pointmass":
        obs_shape = (obs_shape[0] * obs_shape[1], obs_shape[2], obs_shape[3])
    print("Observation shape: ", obs_shape)
    print("Action dim: ", action_dim)
    
    global_step = 0
    
    for epoch in range(args.n_epochs):
        sample_ind_all = np.random.permutation(len(dataset["obs"]))
        sample_ind_next = np.random.permutation(len(dataset["obs"]))
        steps_per_epoch = -(len(sample_ind_all) // -args.batch_size)
        for i in tqdm.tqdm(range(steps_per_epoch), desc=f"Epoch #{epoch}"):
            start = i * args.batch_size
            end = min(len(sample_ind_all), (i + 1) * args.batch_size)
            sample_ind = np.sort(sample_ind_all[start:end])
            samples = {key: dataset[key][sample_ind] for key in dataset_keys}
            sample_ind = np.sort(sample_ind_next[start:end])
            samples['permuted'] = {key: dataset[key][sample_ind] for key in dataset_keys}
            
            if args.train_obs_forward_model:
                obs_forward_losses, obs_next_pred = obs_forward_model.train_step(samples)
                tensorboard.add_scalars("obs_forward", obs_forward_losses, global_step)
            
            if args.train_single_step and epoch < args.ss_stop_epochs:
                ss_losses = ss_model.train_step(samples)
                tensorboard.add_scalars("ss", ss_losses, global_step)
                
            if (epoch >= args.ss_warmup_epochs) and (args.train_multi_step):
                ms_losses = ms_model.train_step(samples)
                for key in ms_losses.keys():
                    tensorboard.add_scalars(key, ms_losses[key], global_step)

            if global_step % 100000 == 0:
                torch.save(
                    ss_model, os.path.join(log_path, f"single_step_{global_step}.pt")
                )
                torch.save(
                    ms_model, os.path.join(log_path, f"multi_step_{global_step}.pt")
                )


            if global_step % 1000 == 0:
                if args.train_single_step:
                    ss_heatmaps = perturb_heatmap(samples["obs"][-1], ss_model.encoder)
                    tensorboard.add_images(
                        "single_step", np.stack(ss_heatmaps), global_step
                    )
                if args.train_multi_step:
                    ms_heatmaps = perturb_heatmap(samples["obs"][-1], ms_model.encoder)
                    tensorboard.add_images(
                        "multi_step", np.stack(ms_heatmaps), global_step
                    )
            
                if args.train_obs_forward_model:
                    obs_next_pred = obs_next_pred[[-1], ...].cpu().detach().numpy()
                    obs_next_pred = obs_next_pred - np.min(obs_next_pred)
                    obs_next_pred = obs_next_pred / np.max(obs_next_pred)
                    
                    obs_next_pred = np.concatenate((obs_next_pred, np.ones((1,3,obs_next_pred.shape[2],1)),samples['obs_next'][[-1], ...]), axis=3)
                    tensorboard.add_images(
                        "obs_forward",obs_next_pred , global_step
                    )
                
                
            global_step += 1
            
    if args.train_single_step:
        torch.save(ss_model, os.path.join(log_path, "single_step_final.pt"))
    if args.train_multi_step:
        torch.save(ms_model, os.path.join(log_path, "multi_step_final.pt"))

    if args.train_obs_forward_model:
        torch.save(obs_forward_model, os.path.join(log_path, "obs_forward_final.pt"))
        

# if __name__ == "__main__":
#     print('\n'.join(sys.argv))
#     parser = ArgumentParser()
#     parser.add_argument("--logdir", type=str, required=True)
#     parser.add_argument("--dataset", type=str, required=True)
    
#     parser.add_argument("--single_step_forward_weight", type=float, default=0.8)
#     parser.add_argument("--single-step-l1-penalty", type=float, default=0.0)
#     parser.add_argument("--single-step-lr", type=float, default=0.0001) # 0.0002 works for 1M (does it??)
#     parser.add_argument("--single-step-resume-path", type=str, default='')#'/home/mrudolph/documents/actbisim/scripts_nav2d/rep_models/single_step_model/single_step_final.pt')
#     parser.add_argument("--single_step_type", type=str, default="inverse_dynamics")
#     parser.add_argument("--ss-stop-epochs", default=1000000, type=int)
#     parser.add_argument("--ss-warmup-epochs", default=0, type=int)
    
#     parser.add_argument("--multi_step_tau", type=float, default=0.005) # model smoothing between target and online model
#     parser.add_argument("--multi_step_sync_freq", type=int, default=1) # how often to sync target and online model
#     parser.add_argument("--multi-step-gamma", type=float, default=0.75) # discount factor for multi-step loss between single step and multi-step model
#     parser.add_argument("--multi-step-lr", type=float, default=0.0001)
#     parser.add_argument("--multi-step-resume-path", default=None)#'/home/mrudolph/documents/actbisim/scripts_nav2d/rep_models/multi_step_model/multi_step_final.pt')
#     parser.add_argument("--k-step-forward-weight", type=float, default=0.0)
#     parser.add_argument("--gamma-schedule", "-gs",  action='store_true')
#     parser.add_argument("--decoder-lr", type=float, default=1e-3)
#     parser.add_argument("--model-to-decode", type=str, default=None)
#     parser.add_argument("--no-train-multi-step", action="store_true")
#     parser.add_argument("--multi-step-forward-loss", type=str, default="l2")
#     parser.add_argument("--train-decode-separately", action="store_true")
#     parser.add_argument("--k-steps", type=int, default=1)
#     parser.add_argument("--use-gt-forward-model", action="store_true")
    
#     parser.add_argument("--single_step_inverse_weight", type=float, default=0.0)# only used by NCE
#     parser.add_argument("--k-steps-dyn", type=int, default=1)
#     parser.add_argument("--single_step_NCE_negative", type=int, default=2)
    
#     parser.add_argument("--batch_size", type=int, default=128)
#     parser.add_argument("--seed", type=int, default=0)
#     parser.add_argument("--n-epochs", type=int, default=40)
    
#     parser.add_argument("--date", "-d", action="store_true")
    
#     parser.add_argument("-sswd", "--single_step_weight_decay", type=float, default=1e-5)
#     parser.add_argument("-mswd", "--multi-step-weight-decay", type=float, default=1e-5)
#     parser.add_argument("--train-autoencoder", action="store_true")
#     parser.add_argument("--no-prompt-overwrite", action="store_true")
#     parser.add_argument("--train-bvae", action="store_true")
    
#     parser.add_argument("--train-single-step", action="store_true")
#     parser.add_argument("--train-multi-step", action="store_true")
#     parser.add_argument("--K-max-steps", type=int, default=5)
#     parser.add_argument("--l1-penalty", type=float, default=0.0)
#     parser.add_argument("--use-states-with-same-action", action="store_true")
#     parser.add_argument("--train-obs-forward-model", action="store_true")
#     parser.add_argument("--use-learned-obs-forward-model", action="store_true")

#     # network arguments
#     parser.add_argument('--encode-hidden-layers', type=int, nargs='+', default=[64,64,64,96,96,128])
#     parser.add_argument('--encode-num-pooling', type=int, default=2)
#     parser.add_argument('--encode-activation', default="relu")
#     parser.add_argument('--encode-layer-norm', action="store_true")
#     parser.add_argument('--post-hidden-layers', type=int, nargs='+', default=[256])
#     parser.add_argument('--post-activation', default="relu")
#     parser.add_argument('--post-layer-norm', action="store_true")
#     parser.add_argument("--use-gen-nets", action="store_true")
#     args = parser.parse_args()

#     train(args)
