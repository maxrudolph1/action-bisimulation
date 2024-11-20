from matplotlib import cm
import numpy as np
import torch


def render(obs):
    if isinstance(obs, torch.Tensor):
        obs = obs.detach().cpu().numpy()
    obs = (obs + 1) / 2
    _, h, w = obs.shape
    img = np.zeros((3, h, w))
    img[0][obs[0] != 0] = 1
    img[1][obs[1] != 0] = 1
    if obs.shape[0] > 2:
        img[2][obs[2] != 0] = 1
    return img


def return_encoded_vector(obs, encoder):
    c, h, w = obs.shape
    encoded = (
        encoder(torch.as_tensor(obs, device="cuda").unsqueeze(0))
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    return encoded


def perturb_heatmap(obs, encoder):
    c, h, w = obs.shape
    encoded = (
        encoder(torch.as_tensor(obs, device="cuda").unsqueeze(0))
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    obs_perturbed = np.broadcast_to(obs, [h * w, c, h, w]).copy()
    mask = (-np.eye(h * w) * 2 + 1).reshape(h * w, h, w)
    obs_perturbed[:, 0] *= mask
    encoded_perturbed = (
        encoder(torch.as_tensor(obs_perturbed, device="cuda")).detach().cpu().numpy()
    )

    distances = np.linalg.norm(encoded - encoded_perturbed, ord=1, axis=-1).reshape(
        h, w
    )
    player_pos = np.argwhere(obs[1] == 1)[0]
    distances[player_pos[0], player_pos[1]] = 0
    if obs.shape[0] > 2:
        goal_pos = np.argwhere(obs[2] == 1)
        if len(goal_pos) > 0:
            distances[goal_pos[0, 0], goal_pos[0, 1]] = 0
    # distances /= np.max(distances)
    #handling possible division by 0
    max_distance = np.max(distances)
    if max_distance != 0:
        distances /= max_distance
    else:
        print("nav2d::utils.py - max_distance is 0\n")
        distances = distances

    heatmap = (
        cm.gray(distances)[:, :, :3]
        .transpose([2, 0, 1])
        .repeat(2, axis=1)
        .repeat(2, axis=2)
    )

    img = render(obs)
    img = img.repeat(2, axis=1).repeat(2, axis=2)
    img[:, :, -1] = [[0], [1], [0]]
    heatmap[:, :, 0] = [[0], [1], [0]]


    return img, heatmap
