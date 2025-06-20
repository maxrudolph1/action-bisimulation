import torch
import torch.nn.functional as F
from models import gen_model_nets

import numpy as np
import pdb


class NCE(torch.nn.Module):
    def __init__(self, obs_shape, act_shape, cfg):
        super().__init__()
        encoder_cfg = cfg.algos.nce.encoder

        self.encoder = gen_model_nets.GenEncoder(obs_shape, cfg=encoder_cfg).cuda()

        num_actions = act_shape if isinstance(act_shape, int) else 4
        self.action_encoder = torch.nn.Embedding(
            num_actions,
            2 * self.encoder.output_dim
        ).cuda()

        self.embed_dim      = self.encoder.output_dim
        self.temperature    = cfg.algos.nce.temperature
        self.learning_rate  = cfg.algos.nce.learning_rate
        self.weight_decay   = cfg.algos.nce.weight_decay

        self.method = cfg.algos.nce.method
        self.k_steps = cfg.algos.nce.k_steps

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.action_encoder.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def distances(self, state1, state2):
        z1 = self.encoder(state1)  # (batch, D)
        z2 = self.encoder(state2)  # (batch, D)
        h  = torch.cat([z1, z2], dim=1)  # concat: (batch, 2D)
        w = self.action_encoder.weight

        if self.method == "cos":
            h_norm = F.normalize(h, dim=1)        # (batch, 2D)
            W_norm = F.normalize(w, dim=1)        # (num_actions, 2D)
            # cosine similarity matrix: (batch, num_actions)
            return -(h_norm @ W_norm.t())
        elif self.method == "l1":
            # L1 distances: shape (batch, num_actions)
            # then we negate to turn into similarity
            diff = h.unsqueeze(1) - w.unsqueeze(0)      # (batch, num_actions, 2D)
            return torch.norm(diff, p=1, dim=2)        # (batch, num_actions)
        else:
            raise ValueError(f"Unknown distance method {self.method}. Only cos or l1 are accepted")

    def share_dependant_models(self, models):
        pass

    def ao(self, obs):
        """
        Just a debugging function that prints the obstacle layer of the obs
        """
        obs = obs.detach().cpu().numpy()

        obs[0] = np.where(obs[0] == -1, 0, obs[0])
        obs[1] = np.where(obs[1] == -1, 0, obs[1])

        print('-' * 80)
        for row in obs[0]:
            print(" ".join(str(int(v)) for v in row))
        print('-' * 80)
        for row in obs[1]:
            print(" ".join(str(int(v)) for v in row))

    def loss(self, obs, obs_next, act):
        # 1) get all pairwise similarities
        sims = self.distances(obs, obs_next)   # (batch, A)

        # 2) build logits = -d/temperature to match
        #    e^{-d/τ} numerator/denominator.
        logits = sims / self.temperature

        return F.cross_entropy(logits, act)
        pass

    def train_step(self, batch, epoch, train_step):  # k=1 is next obs.
        obs = torch.as_tensor(batch["obs"], device="cuda")
        if self.k_steps <= 1:
            obs_next = torch.as_tensor(batch["obs_next"], device="cuda")
        else:
            obs_next = torch.as_tensor(batch["kobs"][:, self.k_steps - 1], device="cuda")
        act = torch.as_tensor(batch["action"], dtype=torch.long, device="cuda")

        # forward + backward
        loss = self.loss(obs, obs_next, act)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # compute accuracy
        with torch.no_grad():
            sims   = self.distances(obs, obs_next)
            logits = sims / self.temperature
            preds  = logits.argmax(dim=1)
            acc    = (preds == act).float().mean().item()

        ret = {"loss": loss.item(), "accuracy": acc}
        self.last_ret = ret
        return ret

    def save(self, path):
        torch.save({
            "encoder":        self.encoder,
            "action_encoder": self.action_encoder,
            "optimizer":      self.optimizer,
        }, path)
