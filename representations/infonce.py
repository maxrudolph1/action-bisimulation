from models import gen_model_nets

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils

class InfoNCE(nn.Module):
    def __init__(self, obs_shape, act_shape, cfg):
        super().__init__()
        encoder_cfg = cfg.algos.infonce.encoder
        self.encoder = gen_model_nets.GenEncoder(obs_shape, cfg=encoder_cfg).cuda()
        self.embed_dim = self.encoder.output_dim

        
        self.learning_rate = cfg.algos.infonce.learning_rate 
        self.weight_decay = cfg.algos.infonce.weight_decay
        self.train_stop_epochs = cfg.algos.infonce.train_stop_epochs

        self.temperature = cfg.algos.infonce.temperature  #TODO: check this..ig follows paper code for now set at .1 ? basically a hyperparam to stabilize logits?

        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def share_dependant_models(self, models):
        pass  

    def infonce_loss(self, z1, z2):
        similarity = z1 @ z2.T  # multipleis the two matrixes via dot prod
        similarity /= self.temperature #TODO: verify this, found from the pytorch impl of infonce but its not in the paper equations
        labels = torch.arange(z1.size(0), device=z1.device) #correct index shoule be  on the diagonal
        return F.cross_entropy(similarity, labels) 

    def train_step(self, batch, epoch, train_step):
        if epoch >= self.train_stop_epochs:
            return {}

        
        # get obs/obs_next from batch, encode and normalize
        obs = torch.as_tensor(batch["obs"], device="cuda")
        obs_next = torch.as_tensor(batch["obs_next"], device="cuda")        
        enc_obs      = self.encoder(obs)        
        enc_obs_next = self.encoder(obs_next)
        norm_enc_obs = F.normalize(enc_obs, p=2, dim=1)
        norm_enc_obs_next = F.normalize(enc_obs_next, p=2, dim=1)

        # perform the actual infonce loss
        loss = self.infonce_loss(norm_enc_obs, norm_enc_obs_next)

        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # diagnostics
        with torch.no_grad():
            sim_matrix = norm_enc_obs @ norm_enc_obs_next.T
            diag_sim = sim_matrix.diag().mean().item()
            offdiag_sim = ((sim_matrix.sum() - sim_matrix.diag().sum()) / (sim_matrix.size(0)**2 - sim_matrix.size(0))).item()
            std_embedding = enc_obs.std().item()
            norm_after = norm_enc_obs.norm(p=2, dim=1).mean().item()  # should always be ~1.0

        ret = {
            "infonce_loss": loss.item(),
            "mean_embedding_norm": enc_obs.norm(p=2, dim=1).mean().item(),  # raw norm
            "embedding_std": std_embedding,
            "normalized_embedding_norm": norm_after,
            "diag_cosine_sim": diag_sim,
            "offdiag_cosine_sim": offdiag_sim
        }
        self.last_ret = ret
        return ret

    def save(self, path):
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
