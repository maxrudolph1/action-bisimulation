import torch
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
#from nav2d_representation.pointmass.d4rl_maze2d import VisualMazeEnv
from environments.nav2d.nav2d import Navigate2D
# from nav2d_representation.nav2d.nav2d_po import Navigate2DPO

ENV_DICT = {
    "pointmass": 0,#VisualMazeEnv,
    "nav2d": Navigate2D,
    # "nav2dpo": Navigate2DPO,
}


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu

def action_set(obs):
    N, c, h, w = obs.shape
    obs += 1
    obs /= 2
    obstacles = obs[:, 0, :, :]
    potential_moves = torch.zeros_like(obstacles)
    pos_idx = torch.argmax(obs[:,1,:,:].flatten(start_dim=1), dim=1).unsqueeze(-1)
    pos_idx = torch.cat([torch.div(pos_idx,h, rounding_mode='floor'), pos_idx % w], dim=1) # find the position of the agent
    move_idx = pos_idx.unsqueeze(0).repeat(4, 1, 1) # repeat the position 4 times, once for each move
    
    move_idx[0,:,0] += 1 
    move_idx[1,:,0] -= 1
    move_idx[2,:,1] += 1
    move_idx[3,:,1] -= 1
    
    move_idx = move_idx.transpose(1,0) 
    batch_idx = torch.arange(0,N).unsqueeze(-1).repeat(1,4).unsqueeze(-1).cuda()

    move_idx = torch.cat([batch_idx, move_idx], dim=-1)
    move_idx = move_idx.flatten(start_dim=0, end_dim=1)
    
    move_idx[:,0] = torch.clamp(move_idx[:,0], 0, N-1) # clamp the moves to the grid
    move_idx[:,1] = torch.clamp(move_idx[:,1], 0, h-1)
    move_idx[:,2] = torch.clamp(move_idx[:,2], 0, w-1)
    
    potential_moves[move_idx[:,0], move_idx[:,1], move_idx[:,2]] = 1
    
    valid_moves_on_grid = potential_moves * (1 - obstacles)
    valid_moves_on_grid *= (1 - obs[:,1,:,:])
    
    valid_moves = torch.zeros(N, 4)
    
    down_moves = move_idx[::4, :] # reordered becaues of the way the moves are ordered
    up_moves = move_idx[1::4, :]
    right_moves = move_idx[2::4, :]
    left_moves = move_idx[3::4, :]
    
    valid_moves[:,0] = valid_moves_on_grid[up_moves[:,0], up_moves[:,1], up_moves[:,2]]
    valid_moves[:,1] = valid_moves_on_grid[down_moves[:,0], down_moves[:,1], down_moves[:,2]]
    valid_moves[:,2] = valid_moves_on_grid[left_moves[:,0], left_moves[:,1], left_moves[:,2]]
    valid_moves[:,3] = valid_moves_on_grid[right_moves[:,0], right_moves[:,1], right_moves[:,2]]

    return valid_moves

def action_set_onehot(obs):
    valid_moves = action_set(obs).cuda()
    N, _ = valid_moves.shape
    batch_idx = torch.arange(0,4).unsqueeze(0).repeat(N,1).cuda()
    
    onehot_idx = torch.pow(2, batch_idx) * valid_moves

    onehot = torch.zeros(N, 4**2)
    onehot[torch.arange(N), torch.sum(onehot_idx, dim=1).long()] = 1
    return onehot

def pairwise(a, b):
    assert a.shape == b.shape
    batch_size = a.shape[0]
    a_expand = torch.broadcast_to(a, [batch_size] + [-1] * len(a.shape))
    b_expand = torch.broadcast_to(b, [batch_size] + [-1] * len(b.shape))
    a_flat = a_expand.reshape([batch_size**2] + list(a.shape[1:]))
    b_flat = b_expand.transpose(0, 1).reshape([batch_size**2] + list(b.shape[1:]))
    return a_flat, b_flat


def pairwise_l1_distance(x):
    a, b = pairwise(x, x)
    return torch.linalg.norm(a - b, ord=1, dim=-1)


def angular_distance(a, b):
    assert a.shape == b.shape
    assert len(a.shape) == 2
    numerator = torch.sum(a * b, dim=-1)
    denominator = torch.linalg.vector_norm(a, dim=-1) * torch.linalg.vector_norm(
        b, dim=-1
    )
    cos_similarity = numerator / denominator
    return torch.atan2(
        torch.sqrt(torch.clamp(1 - cos_similarity**2, min=1e-9)), cos_similarity
    )


def mico_distance(a, b, beta):
    assert a.shape == b.shape
    assert len(a.shape) == 2
    norm = (torch.sum(a**2, dim=-1) + torch.sum(b**2, dim=-1)) / 2
    ang = angular_distance(a, b)
    return 500 * norm + ang, norm, ang


def grad_heatmap(obs, encoder):
    player_pos = np.argwhere(obs[1] == 1)[0]

    obs = torch.as_tensor(obs, device="cuda").requires_grad_()
    obs.grad = None
    norm = torch.linalg.vector_norm(encoder(obs.unsqueeze(0)).squeeze(0), ord=1)
    norm.backward()

    heatmap = obs.grad.cpu().numpy()
    # grad_heatmap = np.sum(grad_heatmap, axis=0)
    heatmap /= np.max(heatmap, axis=(1, 2), keepdims=True)
    print(f"is leaf: {obs.is_leaf}")
    obstacle_heatmap = cm.gray(heatmap[0])[:, :, :3]
    agent_heatmap = cm.gray(heatmap[1])[:, :, :3]

    obstacle_heatmap[player_pos[0], player_pos[1]] = [0, 1, 0]

    return agent_heatmap, obstacle_heatmap


if __name__=="__main__":
    
    K = 20
    obs = np.zeros((3,K,K))
    
    obs[0,:,:] = np.random.random((K,K))
    obs[obs > 0.75] = 1
    obs[obs <= 0.75] = 0
    obs[1,:,:] = 0
    
   
    obs = obs[np.newaxis,:,:,:]
    obs = np.concatenate([obs,obs, obs], axis=0)

    obs[0,1,0,0] = 1
    obs[1,1, 1,2] = 1
    obs[2,1, 3,3]   = 1
    obs[obs < 1] = -1
    # action_set = action_set(torch.as_tensor(obs))
    act_set = action_set_onehot(torch.as_tensor(obs).cuda())
    print(act_set)
    # imgs = np.zeros((3,K,K,3))
    # for i in range(3):
    #     imgs[i, :,:,0] = grids[i]
    #     imgs[i, :,:,1] = obs[i,0,:,:]
    #     imgs[i, :,:,2] = obs[i,1,:,:]

    #     img = np.concatenate([imgs[i,:,:,[0]], imgs[i,:,:,[1]], imgs[i,:,:,[2]]], axis=0)
    #     img = img.transpose(2,1,0)
        
    #     # img = np.concatenate([imgs[i,:,:,[0]], imgs[i,:,:,[1]], imgs[i,:,:,[2]]], axis=1)
    #     # img = np.concatenate([img, img, img], axis=0).transpose(2,1,0)
    #     # print(img.shape)
    #     # plt.imshow(img)
    #     # plt.show()

    #     plt.imsave(f"img{i}.png", img)
    
    
    