from copy import deepcopy

import numpy as np
import torch.nn
from matplotlib import cm

from . import nets
from . import gen_model_nets
import torch.nn.functional as F
import torch

from . import utils



class MultiStep(torch.nn.Module):
    def __init__(
        self, obs_shape=None, action_dim=None, ss_encoder=None,ss_inverse_model=None, learning_rate=None, gamma=None,
        tau=None, sync_freq=None, weight_decay=1e-5, k_steps_dyn=1,k_steps=1, k_step_forward_weight=0.25, effective_gamma=False,
        multi_step_forward_loss="l2", learned_obs_forward_model=None, use_gt_forward_model=False, args=None,
    ):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.ss_encoder = ss_encoder
        self.sync_freq = sync_freq
        self.steps_until_sync = self.sync_freq
        self.k_steps = k_steps
        self.k_steps_dyn = k_steps_dyn
        self.action_dim = action_dim
        self.k_step_forward_weight = k_step_forward_weight
        self.effective_gamma = effective_gamma
        self.multi_step_forward_loss = multi_step_forward_loss
        self.use_gt_forward_model = use_gt_forward_model
        self.use_states_with_same_action = args.use_states_with_same_action
        self.use_learned_obs_forward_model = args.use_learned_obs_forward_model
        assert self.use_gt_forward_model == False or self.use_states_with_same_action == False
        
        if effective_gamma:
            self.gamma = self.gamma / (1 + self.gamma)
        
        if args.use_gen_nets:
            self.encoder = gen_model_nets.GenEncoder(obs_shape,
                                                  hidden_layers=args.encode_hidden_layers,
                                                  num_pooling=args.encode_num_pooling,
                                                  acti=args.encode_activation,
                                                  use_layer_norm=args.encode_layer_norm).cuda()
            self.embed_dim = self.encoder.output_dim
            if self.use_learned_obs_forward_model:
                self.forward_model = learned_obs_forward_model
            else:
                self.forward_model = gen_model_nets.GenForwardDynamics(self.encoder.output_dim, action_dim,
                                                                hidden_layers=args.post_hidden_layers,
                                                                acti=args.post_activation,
                                                                use_layer_norm=args.post_layer_norm).cuda()
            self.inverse_model = gen_model_nets.GenInverseDynamics(self.encoder.output_dim, action_dim,
                                                                hidden_layers=args.post_hidden_layers,
                                                                acti=args.post_activation,
                                                                use_layer_norm=args.post_layer_norm).cuda()
        else:
            self.encoder = nets.Encoder(obs_shape).cuda()
            
            self.ss_inverse_model = ss_inverse_model
            
            if self.use_learned_obs_forward_model:
                self.forward_model = learned_obs_forward_model
            else:
                self.forward_model = nets.ForwardDynamics(
                    self.encoder.output_dim, self.action_dim
                ).cuda()
                
            
            self.inverse_model = nets.InverseDynamics(
                self.encoder.output_dim, self.action_dim
            ).cuda()
        self.encoder_optimizer = torch.optim.Adam(
            list(self.encoder.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        self.forward_model_optimizer = torch.optim.Adam(
            list(self.forward_model.parameters()),
            lr=learning_rate,
            weight_decay= weight_decay,
        )
        
        self.inverse_model_optimizer = torch.optim.Adam(
            list(self.inverse_model.parameters()),
            lr=learning_rate,
        )

        self.target_encoder = deepcopy(self.encoder).cuda()


    def batch_forward_model(self, obs, act):
        actions = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=int).cuda()
        idx = torch.nonzero(obs[:, 1, :,:] + 1)
        action_vecs = actions[act]
        new_locs = torch.clip(idx[:, 1:] + action_vecs, min=0, max=obs.shape[2] - 1)
        moved_idx = idx.clone()
        moved_idx[:, 1:] = new_locs
        next_obs = obs.clone()
        obs_notpresent_idx = torch.nonzero(next_obs[moved_idx[:, 0], 0, moved_idx[:, 1], moved_idx[:, 2]] -1)
        next_obs[moved_idx[obs_notpresent_idx, 0], 1, moved_idx[obs_notpresent_idx, 1], moved_idx[obs_notpresent_idx, 2]] = 1
        next_obs[idx[obs_notpresent_idx, 0], 1, idx[obs_notpresent_idx, 1], idx[obs_notpresent_idx, 2]] = -1
        return next_obs
        
    def train_step(self, batch, ss_average=None):
        
        obs_x = torch.as_tensor(batch["obs"], device="cuda")  # observation ( x )
        obs_x_next = torch.as_tensor(batch["obs_next"], device="cuda") # next observation ( x' )
        act = torch.as_tensor(batch["action"], device="cuda") # action taken at time step ( a_x )
        
        bs = obs_x.shape[0]

        kobs_steps = torch.randint(0, self.k_steps-1, (bs,)).numpy().tolist()
        obs_y = torch.as_tensor(batch["kobs"], device="cuda")[torch.arange(bs), kobs_steps, :,:,:] # random other observation ( y )
        
        kobs_x = torch.as_tensor(batch["kobs"], device="cuda")[torch.arange(bs), self.k_steps_dyn, :,:,:] # obsveration k_steps_dyn away
        kact_x = torch.as_tensor(batch["kaction"], device="cuda") # action taken at time step ( a_x )
        

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
            oxk_encoded_target = self.target_encoder(kobs_x)  # target encoder z = \hat{phi}( x^(k) )
            oxn_encoded_target = self.target_encoder(obs_x_next) # target encoder z = \hat{phi}( x' )

        if not self.use_gt_forward_model and not self.use_states_with_same_action and not self.use_learned_obs_forward_model:
            self.forward_model_optimizer.zero_grad()
            forward_prediction = self.forward_model(ox_encoded_target.detach(), act)
            forward_next_prediction = forward_prediction
            forward_predictions = [forward_prediction]
            for k in range(1,self.k_steps):
                forward_prediction = self.forward_model(forward_prediction, kact_x[:,k])
                forward_predictions.append(forward_prediction)
            forward_model_next_loss = F.mse_loss(
                forward_next_prediction,
                oxn_encoded_target.detach(),
            ) if self.multi_step_forward_loss == "l2" else F.smooth_l1_loss(
                forward_next_prediction,
                oxn_encoded_target.detach(),
            )   
            forward_model_k_loss = F.mse_loss(
                forward_predictions[-1],
                oxk_encoded_target.detach(),
            ) if self.multi_step_forward_loss == "l2" else F.smooth_l1_loss(
                forward_predictions[-1],
                oxk_encoded_target.detach(),
            )
            full_forward_model_loss = forward_model_next_loss + forward_model_k_loss * (self.k_step_forward_weight if self.k_steps > 1 else 0)
            full_forward_model_loss.backward()

            self.forward_model_optimizer.step()
            inverse_model_pred = self.inverse_model(ox_encoded_target, oxn_encoded_target)        
            inverse_model_loss = F.cross_entropy(
                inverse_model_pred,
                act,
            )
            self.inverse_model_optimizer.zero_grad()
            inverse_model_loss.backward()
            self.inverse_model_optimizer.step()
            
            forward_loss_logs = {
                "total_loss": full_forward_model_loss.detach().item(),
                "k_loss": forward_model_k_loss.detach().item(),
                "next_loss:": forward_model_next_loss.detach().item(),
                "inverse_acc": (inverse_model_pred.argmax(dim=1) == act).float().mean().detach().item(),
                # "inverse_acc_forward": (inverse_model_pred_forward_model_enc.argmax(dim=1) == act).float().mean().detach().item(),
            }

        with torch.no_grad():
            ss_encoded_x = self.ss_encoder(obs_x)
            ss_encoded_y = self.ss_encoder(obs_y)
            ss_diffs = (ss_encoded_x - ss_encoded_y).detach()
            ss_distances = torch.linalg.norm(ss_diffs, ord=1, dim=-1)

            if self.use_gt_forward_model:
                obs_x_expanded = obs_x.unsqueeze(1).expand(-1,self.action_dim, -1, -1, -1).reshape(bs * self.action_dim, *obs_x.shape[1:])
                obs_y_expanded = obs_y.unsqueeze(1).expand(-1,self.action_dim, -1, -1, -1).reshape(bs * self.action_dim, *obs_x.shape[1:])
      
                all_actions = torch.arange(self.action_dim, device="cuda").unsqueeze(0).expand(obs_x.shape[0], -1).reshape(-1)

                obs_x_next_true = self.batch_forward_model(obs_x_expanded, all_actions)
                obs_y_next_true = self.batch_forward_model(obs_y_expanded, all_actions)
                
                pred_ox_encoded = self.encoder(obs_x_next_true).reshape(bs, self.action_dim, -1)
                pred_oy_encoded = self.encoder(obs_y_next_true).reshape(bs, self.action_dim, -1)
            elif self.use_states_with_same_action:
                pred_ox_encoded = self.encoder(obs_x_next).unsqueeze(1) 
                pred_oy_encoded = self.encoder(obs_y_next).unsqueeze(1) 
            elif self.use_learned_obs_forward_model:
                obs_x_expanded = obs_x.unsqueeze(1).expand(-1,self.action_dim, -1, -1, -1).reshape(bs * self.action_dim, *obs_x.shape[1:])
                obs_y_expanded = obs_y.unsqueeze(1).expand(-1,self.action_dim, -1, -1, -1).reshape(bs * self.action_dim, *obs_x.shape[1:])
                all_actions = torch.arange(self.action_dim, device="cuda").unsqueeze(0).expand(obs_x.shape[0], -1).reshape(-1)
                pred_ox_next = self.forward_model(
                    obs_x_expanded,
                    all_actions,
                )  # shape (n, 4, e)
                pred_oy_next = self.forward_model(
                    obs_y_expanded,
                    all_actions,
                )  # shape (n, 4, e)
                
                pred_ox_encoded = self.encoder(pred_ox_next).reshape(bs, self.action_dim, -1)
                pred_oy_encoded = self.encoder(pred_oy_next).reshape(bs, self.action_dim, -1)
            else:
                pred_ox_encoded = self.forward_model(
                    ox_encoded_target.unsqueeze(1).expand(-1, self.action_dim, -1),
                    torch.arange(self.action_dim, device="cuda")
                    .unsqueeze(0)
                    .expand(ox_encoded_target.shape[0], -1),
                )  # shape (n, 4, e)
                
                pred_oy_encoded = self.forward_model(
                    oy_encoded_target.unsqueeze(1).expand(-1, self.action_dim, -1),
                    torch.arange(self.action_dim, device="cuda")
                    .unsqueeze(0)
                    .expand(ox_encoded_target.shape[0], -1),
                )  # shape (n, 4, e)
            



            target_distances = torch.linalg.norm(
                pred_ox_encoded - pred_oy_encoded, ord=1, dim=-1
            )
            target_distances = torch.mean(target_distances, dim=-1)
            
        distances = torch.linalg.norm(ox_encoded_online - oy_encoded_online, ord=1, dim=-1)
        
        ss_distance_size = ss_distances.float().mean().detach().item() 
        target_ms_distance_size = target_distances.float().mean().detach().item() 
        cur_ms_distance_size = distances.float().mean().detach().item()
            
        ss_multiplier = 1 - self.gamma #if self.normalized_ss_ms_dist else 1
        self.ss_running_average = ss_average if ss_average is not None else 1

        ss_distances = ss_distances /  self.ss_running_average  # normalize by running average
        
        ms_loss = F.smooth_l1_loss(
            distances, ss_multiplier * ss_distances.detach() + self.gamma * target_distances.detach()
        )
        
        
        self.encoder_optimizer.zero_grad()
        ms_loss.backward()
        self.encoder_optimizer.step()

        if self.steps_until_sync == 0:
            self._sync_params()
            self.steps_until_sync = self.sync_freq
        else:
            self.steps_until_sync -= 1

        
        ms_loss_logs = {
            "total_loss": ms_loss.detach().item(),
            "base_case_distance": ss_distances.float().mean().detach().item(),
            "weighted_ss_distance": ss_multiplier * ss_distances.float().mean().detach().item(),
            "cur_ms_distance": cur_ms_distance_size,
            "forward_pred_ms_distance": target_ms_distance_size,
            "gamma": self.gamma,
            "ss_multiplier": ss_multiplier,
        }
        return {"ms": ms_loss_logs} #, "forward": forward_loss_logs if not self.use_gt_forward_model else {}}

        
    def _sync_params(self):
        for curr, targ in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            targ.data.copy_(targ.data * (1.0 - self.tau) + curr.data * self.tau)
