# InfoNCE strategy:
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import nets
from . import utils
import numpy as np


#     Copy infoNCE loss from https://github.com/RElbers/info-nce-pytorch/blob/main/info_nce/__init__.py
#     Create network (linear) to get the embeddings for the actions
#     use the same embedding model for the single step values

def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.
    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.
    Returns:
         Value of the InfoNCE Loss.
     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    # print(query.shape, positive_key.shape, negative_keys.shape)
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)

def cudastring(iscuda):
    return "cuda" if iscuda else "cpu"

class InfoNCE(nn.Module):

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)

class KActionEncoder(torch.nn.Module):
    def __init__(self, action_dim, embed_dim):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Conv1d(action_dim, 256, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(256, embed_dim, 1),
        )

    def forward(self, actions):
        unsqueeze = False
        if len(actions.shape) == 2:
            actions = actions.unsqueeze(1)
            unsqueeze = True
        output = self.fc(actions.transpose(-2,-1)).transpose(-2,-1)
        if unsqueeze: return output[:,0,:]
        return output


class NCEEncoder(nn.Module):
    def __init__(self, action_num, obs_dim, num_steps, discrete=True, iscuda=True):
        super().__init__()
        self.action_num = action_num # the number of actions if discrete, continuous actions if continuous
        self.state_encoder = nets.Encoder(obs_dim)
        self.output_dim = self.state_encoder.output_dim
        self.k_action_encoder = KActionEncoder(self.action_num * num_steps, self.output_dim * 2) 
        self.discrete = discrete
        self.num_steps = num_steps
        self.iscuda =iscuda

    def cuda(self):
        super().cuda()
        self.iscuda = True
        return self

    def cpu(self):
        super().cpu()
        self.iscuda = False
        return self


    def forward(self, obs):
        return self.state_encoder(obs)

    def encode(self, obs, obs_next, acts, kvalid, num_negative=2):
        ''' @param obs: the observation of state [batch, *obs_dims]
        @param obs_next: the observation of state after k time steps [batch, *obs_dims]
        @param acts: the sequence of actions in between these observations 
                        discrete: [batch, k, 1] # action indexes
                        continuous: [batch, k, continuous_act_dim]
                        This is a numpy array
        returns: infoNCE encodings of query, positive key, negative_keys
        '''
        batch_size = obs.shape[0]
        enc = self.state_encoder(obs)
        next_enc = self.state_encoder(obs_next)
        query = (enc, next_enc)
        npacts = acts.cpu().numpy()
        if self.discrete:
            hot_actions = F.one_hot(torch.as_tensor(acts, device=cudastring(self.iscuda)), self.action_num).float().reshape(batch_size, -1)
            # print(hot_actions)
            negative_keys = self.k_action_encoder(
                        create_negative_discrete(npacts, num_negative, self.action_num,self.iscuda)
                        .float()
                        )
            # print(negative_keys.shape, create_negative_discrete(npacts, num_negative, self.action_num,self.iscuda)
            #             .float().shape, hot_actions.shape)
            positive_key = self.k_action_encoder(hot_actions)
        else:
            negative_keys = self.k_action_encoder(
                    create_negative_continuous(npacts, num_negative, self.iscuda)
                    )
            positive_key = self.k_action_encoder(hot_actions)
        # print(self.output_dim, create_negative_discrete(npacts, num_negative, self.action_num,self.iscuda)
        #                 .reshape(batch_size, -1).float().shape, self.k_action_encoder)
        return query, positive_key, negative_keys

def transpose(x):
    return x.transpose(-2, -1)

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

def create_negative_discrete(actions, num_negative, action_num,cuda=True):
    '''
    @param actions: batch of length k sequences of action indexes [batch, k]
    @param num_negative: number of negative samples to generate per value
    '''
    batch_size = actions.shape[0]
    new_actions = list()
    for i in range(num_negative):
        random_reassignment = np.random.randint(1,action_num, size=actions.shape)
        new_action = torch.as_tensor(np.mod(actions + random_reassignment, action_num), device=cudastring(cuda)) # randomly revalues to values not equal to the original
        new_actions.append(F.one_hot(new_action, action_num).reshape(batch_size, -1))
    # print(torch.stack(new_actions, dim = 1), torch.stack(new_actions, dim = 1).shape)
    return torch.stack(new_actions, dim = 1) # num_negative action sequences for each action sequence in actions

def create_negative_continuous(actions, num_negative, cuda=True):
    new_actions = list()
    for i in range(num_negative):
        new_action = np.random.uniform(-1,1,size=actions.shape) # randomly revalues to values, assumes actions between -1,1
        new_actions.append(torch.as_tensor(new_action, device=cudastring(cuda)))
    return torch.stack(new_actions, dim = 1) # num_negative action sequences for each action sequence in actions


class KNCEStep(torch.nn.Module):
    def __init__(
        self, obs_shape, action_dim, learning_rate, forward_model_weight, inverse_model_weight, l1_penalty,weight_decay=1e-5,num_steps=1,num_negative=2 
    ):
        super().__init__()
        # for now, discrete and cuda is True
        self.encoder = NCEEncoder(action_dim, obs_shape, num_steps, discrete=True, iscuda=True).cuda()
        self.embed_dim = self.encoder.output_dim
        self.action_dim = action_dim
        self.num_negative = num_negative
        self.num_steps = num_steps

        # We could regularize with an inverse weight
        self.forward_model = nets.ForwardDynamics(self.embed_dim, action_dim).cuda()
        self.inverse_model = nets.InverseDynamics(self.embed_dim, action_dim).cuda()
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.forward_model.parameters())
            + list(self.inverse_model.parameters()),
            lr=learning_rate,
            # weight_decay=h["weight_decay"],
            weight_decay=weight_decay,
        )

        self.forward_model_weight = forward_model_weight
        self.inverse_model_weight = inverse_model_weight
        self.l1_penalty = l1_penalty

    def train_step(self, batch):
        obs = torch.as_tensor(batch["obs"], device="cuda")
        act = torch.as_tensor(batch["action"], device="cuda") if self.num_steps == 1 else torch.as_tensor(batch["kaction"], device="cuda")
        single_act = torch.as_tensor(batch["action"], device="cuda")
        kvalid = torch.as_tensor(batch["kvalid"], device="cuda")
        obs_next = torch.as_tensor(batch["obs_next"], device="cuda") if self.num_steps == 1 else torch.as_tensor(batch["kobs"], device="cuda")

        valid = 1 - torch.as_tensor(kvalid).cuda().float()

        oon_enc, positive_key, negative_keys = self.encoder.encode(obs, obs_next, act, kvalid, num_negative=2)
        o_encoded, on_encoded = oon_enc
        query = torch.cat(oon_enc, dim=-1) ## the query is the combined values

        nce_loss = info_nce(query, positive_key, negative_keys=negative_keys, temperature=0.1, reduction='mean', negative_mode='paired')

        if self.forward_model_weight > 0:
            forward_model_loss = F.mse_loss(
                self.forward_model(o_encoded, single_act),
                on_encoded,
            )
        else:
            forward_model_loss = 0

        if self.l1_penalty > 0:
            l1_loss = (
                torch.linalg.vector_norm(o_encoded, ord=1, dim=1).mean()
                + torch.linalg.vector_norm(on_encoded, ord=1, dim=1).mean()
            ) / 2
        else:
            l1_loss = 0

        if self.inverse_model_weight > 0:
            inverse_model_pred = self.inverse_model(o_encoded, on_encoded)
            inverse_model_loss = F.cross_entropy(
                inverse_model_pred,
                act,
            )
        else:
            inverse_model_loss = torch.tensor(0)
            inverse_model_pred = F.one_hot(torch.zeros(act.shape).long().cuda(), self.encoder.action_num)

        total_loss = (
            self.forward_model_weight * forward_model_loss
            + self.l1_penalty * l1_loss
            + self.inverse_model_weight * inverse_model_loss
            + nce_loss
        )
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        ret = {
            "inverse": inverse_model_loss.detach().item(),
            "forward": forward_model_loss.detach().item(),
            "nce": nce_loss.detach().item(),
            "total": total_loss.detach().item(),
            "accuracy": torch.mean(
                (torch.argmax(inverse_model_pred, dim=-1) == act).float()
            )
            .detach()
            .item(),
        }
        if self.forward_model_weight > 0:
            ret["forward"] = forward_model_loss.detach().item()
        if self.l1_penalty > 0:
            ret["l1_penalty"] = l1_loss.detach().item()

        return ret