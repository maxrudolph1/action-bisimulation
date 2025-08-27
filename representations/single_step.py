# from copy import deepcopy

import numpy as np
import torch.nn
# from matplotlib import cm

from models import gen_model_nets
import torch.nn.functional as F
import torch

from . import utils

# import pdb


class SingleStep(torch.nn.Module):
    def set_class_weights(self, w: torch.Tensor):
        self.ce_weight = w.to("cuda")

    def set_action_norm(self, mean, std):
        mean = torch.as_tensor(mean, dtype=torch.float32, device="cuda")
        std  = torch.as_tensor(std,  dtype=torch.float32, device="cuda")
        std  = torch.clamp(std, min=1e-6)
        self.action_mean.copy_(mean)
        self.action_std.copy_(std)

    def __init__(
        self, obs_shape, act_shape, cfg,
    ):
        super().__init__()
        encoder_cfg = cfg.algos.single_step.encoder
        forward_cfg = cfg.algos.single_step.forward
        inverse_cfg = cfg.algos.single_step.inverse

        self.ce_weight = None

        self.l2_penalty = cfg.algos.single_step.l2_penalty
        self.use_l2_norm = cfg.algos.single_step.use_l2_norm

        self.l1_penalty = cfg.algos.single_step.l1_penalty
        self.dynamic_l1_penalty = cfg.algos.single_step.dynamic_l1_penalty

        if self.use_l2_norm:
            # raw_encoder = gen_model_nets.GenEncoder(obs_shape, cfg=encoder_cfg).cuda()
            # self.embed_dim = raw_encoder.output_dim
            # self.encoder = torch.nn.Sequential(
            #     raw_encoder,
            #     torch.nn.Softmax(dim=1)
            # )
            # self.encoder.output_dim = self.embed_dim
            self.encoder = gen_model_nets.GenEncoder(obs_shape, cfg=encoder_cfg).cuda()
            self.embed_dim = self.encoder.output_dim
        else:
            self.encoder = gen_model_nets.GenEncoder(obs_shape, cfg=encoder_cfg).cuda()
            self.embed_dim = self.encoder.output_dim

        self.forward_model = gen_model_nets.GenForwardDynamics(self.embed_dim, act_shape, forward_cfg).cuda()
        self.inverse_model = gen_model_nets.GenInverseDynamics(self.embed_dim, act_shape, inverse_cfg).cuda()

        self.learning_rate = cfg.algos.single_step.learning_rate
        self.weight_decay = cfg.algos.single_step.weight_decay
        self.forward_weight = cfg.algos.single_step.forward_weight

        self.train_stop_epochs = cfg.algos.single_step.train_stop_epochs

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.forward_model.parameters()) + list(self.inverse_model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        self.register_buffer("action_mean", torch.zeros((act_shape,), dtype=torch.float32, device="cuda"))
        self.register_buffer("action_std", torch.ones((act_shape,), dtype=torch.float32, device="cuda"))

        self.log_priors = None
        self.tau = 1.3

    def set_class_priors(self, p: torch.Tensor):
        self.log_priors = p.clamp_min(1e-12).log().to("cuda")

    def share_dependant_models(self, models):
        pass

    def train_step(self, batch, epoch, train_step):
        if epoch >= self.train_stop_epochs:
            return {}
        obs = torch.as_tensor(batch["obs"], device="cuda")
        obs_next = torch.as_tensor(batch["obs_next"], device="cuda")

        # NOTE: FOR POINTMAZE ONLY
        obs = obs.float() / 127.5 - 1.0
        obs_next = obs_next.float() / 127.5 - 1.0

        # act = torch.as_tensor(batch["action"], device="cuda") # NOTE: changed for pointmaze
        # act = torch.as_tensor(batch["action"], device="cuda").squeeze(-1).long()
        act = torch.as_tensor(batch["action"], device="cuda", dtype=torch.float32)  # (B, 2)
        a_norm = (act - self.action_mean) / self.action_std

        o_encoded = self.encoder(obs)
        on_encoded = self.encoder(obs_next)

        if self.use_l2_norm:
            o_encoded = F.normalize(o_encoded, p=2, dim=1)
            on_encoded = F.normalize(on_encoded, p=2, dim=1)

        if self.forward_weight > 0:
            target = on_encoded.detach()
            # forward_model_loss = F.mse_loss(self.forward_model(o_encoded, act), target)
            # forward_model_loss = F.mse_loss(self.forward_model(o_encoded, act), on_encoded)
            forward_model_loss = F.mse_loss(self.forward_model(o_encoded, a_norm), target)
        else:
            # forward_model_loss = 0
            forward_model_loss = torch.zeros((), device="cuda")

        if self.l1_penalty > 0 and not self.use_l2_norm:
            l1_loss = (
                torch.linalg.vector_norm(o_encoded, ord=1, dim=1).mean()
                + torch.linalg.vector_norm(on_encoded, ord=1, dim=1).mean()
            ) / 2
            # pdb.set_trace()
        else:
            l1_loss = torch.zeros(1, device="cuda")

        inv_pred_norm = self.inverse_model(o_encoded, on_encoded)   # expect shape (B, 2)
        inverse_model_loss = F.smooth_l1_loss(inv_pred_norm, a_norm)  # Huber is robust

        # De-normalize for metrics
        inv_pred = inv_pred_norm * self.action_std + self.action_mean

        # inverse_model_pred = self.inverse_model(o_encoded, on_encoded)
        # if self.log_priors is not None:
        #     inverse_model_pred = inverse_model_pred - self.tau * self.log_priors
        # inverse_model_loss = F.cross_entropy(inverse_model_pred, act)

        # with torch.no_grad():
        #     probs = inverse_model_pred.softmax(dim=1)
        #     top1 = probs.argmax(dim=1)
        #
        #     num_actions = probs.shape[1]
        #     label_hist = torch.bincount(act, minlength=num_actions).to(torch.float32)
        #     pred_hist  = torch.bincount(top1, minlength=num_actions).to(torch.float32)
        #
        #     avg_conf = probs.max(dim=1).values.mean()
        #     pred_entropy = (-probs.clamp_min(1e-12).log() * probs).sum(dim=1).mean()
        #
        #
        # accuracy = torch.mean(
        #     (torch.argmax(inverse_model_pred, dim=1) == act).float()
        # )

        # if self.dynamic_l1_penalty:
        #     gain = 5
        #     cur_l1_penalty = self.l1_penalty * np.exp(- gain * (accuracy.detach().item() - 1) ** 2)
        # else:
        #     cur_l1_penalty = self.l1_penalty
        #
        # # gives us the mean of the encoded states
        # pre_penalized_l1_loss = l1_loss.detach().item()  # NOTE: mostly for debugging
        #
        # l1_loss = cur_l1_penalty * l1_loss
        # # pdb.set_trace()
        # forward_model_loss = self.forward_weight * forward_model_loss
        #
        # # l2 loss stuff
        # if self.use_l2_norm:
        #     l2_per = torch.linalg.norm(o_encoded, ord=2, dim=1)
        #     l2_loss = l2_per.mean() * self.l2_penalty
        #     total_loss = (forward_model_loss + l2_loss + inverse_model_loss)
        # else:
        #     total_loss = (forward_model_loss + l1_loss + inverse_model_loss)
        if self.dynamic_l1_penalty:
            with torch.no_grad():
                inv_mae = torch.mean(torch.abs(inv_pred - act)).item()
            # map MAE to (0,1] scale roughly; tune k as you like
            k = 1.0
            score = torch.exp(torch.tensor(-k * inv_mae))  # higher better
            cur_l1_penalty = float(self.l1_penalty) * float(score)
        else:
            cur_l1_penalty = float(self.l1_penalty)

        pre_penalized_l1 = l1_loss.detach().item()
        l1_loss = cur_l1_penalty * l1_loss
        forward_model_loss = self.forward_weight * forward_model_loss

        # Total
        if self.use_l2_norm:
            l2_per = torch.linalg.norm(o_encoded, ord=2, dim=1)
            l2_loss = l2_per.mean() * self.l2_penalty
            total_loss = forward_model_loss + l2_loss + inverse_model_loss
        else:
            total_loss = forward_model_loss + l1_loss + inverse_model_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        # mean_element_magnitude = torch.abs(o_encoded).float().mean().detach().item()

        l2_loss_val = 0.0
        if self.use_l2_norm:
            l2_loss_val = l2_loss.detach().item()

        with torch.no_grad():
            # Angle error (degrees) and cosine similarity between predicted & true actions
            eps = 1e-7
            cos = F.cosine_similarity(inv_pred, act, dim=1)
            cos_clamped = torch.clamp(cos, -1+eps, 1-eps)
            angle_deg = torch.rad2deg(torch.acos(cos_clamped)).mean()

            mse = F.mse_loss(inv_pred, act)
            mae = torch.mean(torch.abs(inv_pred - act))

        ret = {
            "inverse_loss": float(inverse_model_loss.detach().item()),
            "forward_loss": float(forward_model_loss.detach().item()),
            "l1_loss": float(l1_loss.detach().item()) if not self.use_l2_norm else 0.0,
            "loss": float(total_loss.detach().item()),
            "angle_deg": float(angle_deg.item()),
            "cos_sim": float(cos.mean().item()),
            "inv_mse": float(mse.detach().item()),
            "inv_mae": float(mae.detach().item()),
            "cur_l1_penalty": cur_l1_penalty,
            "mean_encoded_magnitude": pre_penalized_l1,  # your existing debug proxy
        }

        # ret = {
        #     "inverse_loss": inverse_model_loss.detach().item(),
        #     "l1_loss": float(l1_loss.detach().item()) if not self.use_l2_norm else 0.0,
        #     "l2_loss": l2_loss_val,
        #     "mean_encoded_magnitude": pre_penalized_l1_loss,  # purely for debugging and logging
        #     # for pre-penalized loss, expect lower l1 values to result in higher pre-penalty
        #     # aka: 0.01 should have a LOWER pre-penalty
        #     # aka: 0.0001 should have a HIGHER pre-penalty
        #     "loss": total_loss.detach().item(),
        #     "accuracy": torch.mean((top1 == act).float()).item(),
        #     "cur_l1_penalty": cur_l1_penalty,
        #
        #     "avg_conf": avg_conf.item(),
        #     "pred_entropy": pred_entropy.item(),
        #     "label_hist": label_hist.cpu(),
        #     "pred_hist": pred_hist.cpu(),
        #     # "mean_element_magnitude": mean_element_magnitude,
        #     # "mean_representation_magnitude": torch.linalg.vector_norm(o_encoded, ord=1, dim=1).mean().detach().item(),
        # }

        self.last_ret = ret
        return ret

    def save(self, path):
        torch.save(
            {
                "encoder": self.encoder,
                "forward_model": self.forward_model,
                "inverse_model": self.inverse_model,
                "optimizer": self.optimizer,
            },
            path,
        )
