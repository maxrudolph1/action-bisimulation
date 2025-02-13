# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
# import os
import random
import time
# from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from environments.nav2d.nav2d_sb3 import Navigate2D
import hydra
from omegaconf import DictConfig, OmegaConf
from models.gen_model_nets import GenEncoder
import stable_baselines3 as sb3
# import copy
import imageio


def make_env(env_kwargs=dict(num_obstacles=0, grid_size=10, static_goal=True, obstacle_diameter=2)):
    capture_video = False

    def thunk():
        if capture_video:
            env = Navigate2D(**env_kwargs)
            env = gym.wrappers.RecordVideo(env, "videos")
        else:
            env = Navigate2D(**env_kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(0)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, encoder_cfg):
        super().__init__()
        obs_shape = env.single_observation_space.shape
        self.encoder = GenEncoder(obs_shape, cfg=encoder_cfg).cuda()
        self.output_dim = self.encoder.output_dim
        self.q_value_head = nn.Linear(self.output_dim, env.single_action_space.n)
        self.encoder_cfg = encoder_cfg
        self.obs_shape = obs_shape

    def forward(self, x):
        x = x.float() / 255.0
        return self.q_value_head(self.encoder(x))

    @classmethod
    def from_encoder_checkpoint(cls, env, encoder_checkpoint_path,):
        encoder_checkpoint = torch.load(encoder_checkpoint_path)
        encoder_cfg = encoder_checkpoint.get("cfg")
        model = cls(env, encoder_cfg)
        model.encoder.load_state_dict(encoder_checkpoint["state_dict"])
        return model

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
            poetry run pip install "stable_baselines3==2.0.0a1"
            """)
    assert cfg.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{cfg.exp_name}__{cfg.seed}__{int(time.time())}"
    if cfg.use_wandb:
        import wandb

        wandb.init(
            project=cfg.wandb_project_name,
            entity=cfg.wandb_entity,
            sync_tensorboard=True,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in OmegaConf.to_container(cfg, resolve=True).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_kwargs=cfg.env) for i in range(cfg.num_envs)]
    )
    eval_env = make_env(env_kwargs=cfg.env)()
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    if cfg.encoder.path:
        q_network = QNetwork.from_encoder_checkpoint(envs, cfg.encoder.path).to(device)
    else:
        q_network = QNetwork(envs, encoder_cfg=cfg.encoder).to(device)

    optimizer = optim.Adam(q_network.parameters(), lr=cfg.rl.learning_rate)
    target_network = QNetwork(envs, encoder_cfg=q_network.encoder_cfg).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        cfg.rl.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    succ_ = np.zeros(50)
    ep_idx = 0
    obs, _ = envs.reset(seed=cfg.seed)
    for global_step in range(cfg.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(cfg.rl.start_epsilon, cfg.rl.end_epsilon, cfg.rl.exploration_fraction * cfg.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            # if (obs.shape == (3, 7, 7)): HACK: this is a bit hardcoded. the below should work better for different grid_sizes
            if obs.ndim == 3:
                obs = np.expand_dims(obs, axis=0)
            # BUG: previous issue was that it wants obs of shape (1, 3, 7, 7), but it was missing the first dim sometimes
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    succ_[ep_idx % 50] = (info['episode']['r'] > -50)
                    ep_idx += 1

                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    writer.add_scalar("charts/success_rate", succ_.mean(), global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # # TODO: Logic for representation model input here
        # if (cfg.encoder_path is not None):
        #     pass

        # ALGO LOGIC: training.
        if global_step > cfg.rl.learning_starts:
            if global_step % cfg.rl.train_frequency == 0:
                data = rb.sample(cfg.rl.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + cfg.rl.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    if not cfg.use_wandb:
                        print("SPS:", int(global_step / (time.time() - start_time)), "Global Step:", global_step, "Loss:", loss.item())
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % cfg.rl.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        cfg.rl.tau * q_network_param.data + (1.0 - cfg.rl.tau) * target_network_param.data
                    )

        if global_step % cfg.render_freq == 0:
            frames = []
            for _ in range(5):
                obs, _ = eval_env.reset()
                terminated, truncated = False, False
                while not (terminated or truncated):
                    q_values = q_network(torch.Tensor(obs).unsqueeze(0).to(device))
                    actions = torch.argmax(q_values, dim=1).cpu().numpy().squeeze()
                    obs, reward, terminated, truncated, infos = eval_env.step(actions)
                    frames.append(obs)
            frames = np.array(frames).transpose(0, 2, 3, 1)
            if cfg.use_wandb:
                from moviepy import ImageSequenceClip

                # Create a clip from the list of frames
                clip = ImageSequenceClip(list(frames), fps=5)
                temp_video_path = f"render_{global_step}.mp4"
                # Write video file with a specific codec (libx264)
                clip.write_videofile(
                    temp_video_path,
                    codec="libx264",
                    audio=False,
                    logger=None
                )
                # Log the video file to wandb
                wandb.log({"render": wandb.Video(temp_video_path, format="mp4")})

                # video_tensor = torch.tensor(frames).permute(0, 3, 1, 2)[None]  
                # writer.add_video("render_video", video_tensor, global_step, fps=5)
            else:
                imageio.mimsave('test.gif', frames, fps=5)
            print("rendered test.gif at global_step", global_step)


    if cfg.save_model:
        model_path = f"runs/{run_name}/{cfg.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from rl.clearnrl_dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            cfg.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
