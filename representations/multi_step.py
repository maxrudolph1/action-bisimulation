from copy import deepcopy

import numpy as np
import torch.nn
from matplotlib import cm

from models import nets
from models import gen_model_nets
import torch.nn.functional as F
import torch

from . import utils



class MultiStep(torch.nn.Module):
    def __init__(
        self, obs_shape, act_shape, cfg, 
    ): 
        super().__init__()
        
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        encoder_cfg = cfg.algos.multi_step.encoder
        forward_cfg = cfg.algos.multi_step.forward
        self.gamma = cfg.algos.multi_step.gamma
        self.tau = cfg.algos.multi_step.tau
        
        if cfg.algos.multi_step.get("base_case_path"):
            self.ss_encoder = torch.load(cfg.algos.multi_step.get("base_case_path"))['encoder']
        else:
            self.ss_encoder = None
                        
        self.multi_step_forward_loss = cfg.algos.multi_step.multi_step_forward_loss
        self.use_states_with_same_action = cfg.algos.multi_step.use_states_with_same_action
        self.steps_until_sync = 0
        self.sync_freq = cfg.algos.multi_step.sync_freq
        

        if cfg.algos.multi_step.warm_start_ms_with_ss:
            self.encoder = deepcopy(self.ss_encoder).cuda()
        else:
            self.encoder = gen_model_nets.GenEncoder(obs_shape, cfg=encoder_cfg).cuda() 
            
        self.embed_dim = self.encoder.output_dim
        self.forward_model = gen_model_nets.GenForwardDynamics(self.embed_dim, act_shape, forward_cfg).cuda()
            
        self.encoder_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()),
            lr=cfg.algos.multi_step.learning_rate,
            weight_decay=cfg.algos.multi_step.weight_decay,
        )
        
        self.forward_model_optimizer = torch.optim.Adam(
            list(self.forward_model.parameters()),
            lr=cfg.algos.multi_step.learning_rate,
            weight_decay=cfg.algos.multi_step.weight_decay,
        )

        self.target_encoder = deepcopy(self.encoder).cuda()
        self.ss_train_warmup_epochs = cfg.algos.multi_step.ss_train_warmup_epochs
        self.forward_model_steps_per_batch = cfg.algos.multi_step.forward_model_steps_per_batch
        self.reset_forward_model_every = cfg.algos.multi_step.reset_forward_model_every


    # def batch_forward_model(self, obs, act):
    #     actions = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=int).cuda()
    #     idx = torch.nonzero(obs[:, 1, :,:] + 1)
    #     action_vecs = actions[act]
    #     new_locs = torch.clip(idx[:, 1:] + action_vecs, min=0, max=obs.shape[2] - 1)
    #     moved_idx = idx.clone()
    #     moved_idx[:, 1:] = new_locs
    #     next_obs = obs.clone()
    #     obs_notpresent_idx = torch.nonzero(next_obs[moved_idx[:, 0], 0, moved_idx[:, 1], moved_idx[:, 2]] -1)
    #     next_obs[moved_idx[obs_notpresent_idx, 0], 1, moved_idx[obs_notpresent_idx, 1], moved_idx[obs_notpresent_idx, 2]] = 1
    #     next_obs[idx[obs_notpresent_idx, 0], 1, idx[obs_notpresent_idx, 1], idx[obs_notpresent_idx, 2]] = -1
    #     return next_obs
        
    def share_dependant_models(self, models):
        if self.ss_encoder is None:
            self.ss_encoder = models.get("single_step").encoder
            
        
    def train_step(self, batch, epoch, train_step):
        if epoch < self.ss_train_warmup_epochs:
            return {}
        
        obs_x = torch.as_tensor(batch["obs"], device="cuda")  # observation ( x )
        obs_x_next = torch.as_tensor(batch["obs_next"], device="cuda") # next observation ( x' )
        act = torch.as_tensor(batch["action"], device="cuda") # action taken at time step ( a_x )

        if train_step % self.reset_forward_model_every == 0 and self.reset_forward_model_every > 0:
            self.forward_model.reset_weights()
        
        # we shuffle the observations to get a random other observation, may need to change this later because the sampling is biased.
        comp_idxs = np.random.permutation(np.arange(obs_x.shape[0]))
        obs_y = obs_x[comp_idxs]
        obs_y_next = obs_x_next[comp_idxs]

        bs = obs_x.shape[0]

        ## NOT SUPPORTING K-step forward modeling right now
        # kobs_steps = torch.randint(0, self.k_steps-1, (bs,)).numpy().tolist()
        # obs_y = torch.as_tensor(batch["kobs"], device="cuda")[torch.arange(bs), kobs_steps, :,:,:] # random other observation ( y )
        # kobs_x = torch.as_tensor(batch["kobs"], device="cuda")[torch.arange(bs), self.k_steps_dyn, :,:,:] # obsveration k_steps_dyn away
        # kact_x = torch.as_tensor(batch["kaction"], device="cuda") # action taken at time step ( a_x )
        

        if self.use_states_with_same_action:
            act_x = act
            possible_actions = [0,1,2,3]
            similar_action_obs = obs_x.clone()
            similar_action_obs_next = obs_x_next.clone()
            for action in possible_actions:
                cur_act_idx = torch.nonzero(act_x == action)
                shuffled_actions = torch.randperm(len(cur_act_idx))
                similar_action_obs[cur_act_idx] = obs_x[cur_act_idx[shuffled_actions]]
                similar_action_obs_next[cur_act_idx] = obs_x_next[cur_act_idx[shuffled_actions]]
            obs_y = similar_action_obs
            obs_y_next = similar_action_obs_next
            
            
        ox_encoded_online = self.encoder(obs_x)
        oy_encoded_online = self.encoder(obs_y)
         
        with torch.no_grad():
            ox_encoded_target = self.target_encoder(obs_x) # target encoder z = \hat{phi}(x)
            oy_encoded_target = self.target_encoder(obs_y) # target encoder z = \hat{phi}(y)
            # oxk_encoded_target = self.target_encoder(kobs_x)  # target encoder z = \hat{phi}( x^(k) )
            oxn_encoded_target = self.target_encoder(obs_x_next) # target encoder z = \hat{phi}( x' )


        # optimize latent forward model
        for _ in range(self.forward_model_steps_per_batch):
            self.forward_model_optimizer.zero_grad()
            latent_forward_prediction = self.forward_model(ox_encoded_target.detach(), act)

            forward_model_next_loss = F.smooth_l1_loss(
                latent_forward_prediction,
                oxn_encoded_target.detach(),
            )   
            forward_model_next_loss.backward()
            self.forward_model_optimizer.step()

        with torch.no_grad():
            ss_encoded_x = self.ss_encoder(obs_x)
            ss_encoded_y = self.ss_encoder(obs_y)
            ss_diffs = (ss_encoded_x - ss_encoded_y).detach()
            ss_distances = torch.linalg.norm(ss_diffs, ord=1, dim=-1)

            if self.use_states_with_same_action:
                pred_ox_encoded = self.encoder(obs_x_next).unsqueeze(1) 
                pred_oy_encoded = self.encoder(obs_y_next).unsqueeze(1) 
            else:
                pred_ox_encoded = self.forward_model(
                    ox_encoded_target.unsqueeze(1).expand(-1, self.act_shape, -1),
                    torch.arange(self.act_shape, device="cuda")
                    .unsqueeze(0)
                    .expand(ox_encoded_target.shape[0], -1),
                )  # shape (n, 4, e)
                
                pred_oy_encoded = self.forward_model(
                    oy_encoded_target.unsqueeze(1).expand(-1, self.act_shape, -1),
                    torch.arange(self.act_shape, device="cuda")
                    .unsqueeze(0)
                    .expand(ox_encoded_target.shape[0], -1),
                )  # shape (n, 4, e)
            

            target_distances = torch.linalg.norm(
                pred_ox_encoded - pred_oy_encoded, ord=1, dim=-1
            )
            target_distances = torch.mean(target_distances, dim=-1)
            
        distances = torch.linalg.norm(ox_encoded_online - oy_encoded_online, ord=1, dim=-1)
        
        target_ms_distance_size = target_distances.float().mean().detach().item() 
        cur_ms_distance_size = distances.float().mean().detach().item()
            

        
        ms_loss = F.smooth_l1_loss(
            distances, (1 - self.gamma) * ss_distances.detach() + self.gamma * target_distances.detach()
        )
        
        self.encoder_optimizer.zero_grad()
        ms_loss.backward()
        self.encoder_optimizer.step()

        if self.steps_until_sync == 0:
            self._sync_params()
            self.steps_until_sync = self.sync_freq
        else:
            self.steps_until_sync -= 1

        
        log = {
            "total_loss": ms_loss.detach().item(),
            "base_case_distance": ss_distances.float().mean().detach().item(),
            "cur_ms_distance": cur_ms_distance_size,
            "forward_ms_distance": target_ms_distance_size,
            "gamma": self.gamma,
            "forward_model_loss": forward_model_next_loss.detach().item(),
        }
        return log

        
    def _sync_params(self):
        for curr, targ in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            targ.data.copy_(targ.data * (1.0 - self.tau) + curr.data * self.tau)
            
    def save(self, path):
        torch.save(
            {
                "encoder": self.encoder,
                "forward_model": self.forward_model,
            },
            path,
        )
