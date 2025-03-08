# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from environments.nav2d.nav2d_sb3 import Navigate2D
import hydra
from omegaconf import DictConfig, OmegaConf
from models.gen_model_nets import GenEncoder
import stable_baselines3 as sb3
import imageio
import datetime
import cv2
from tqdm import tqdm
from moviepy import ImageSequenceClip


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
        obs_shape = env.single_observation_space.shape  # (3, 7, 7)

        latent_path = encoder_cfg.get("latent_encoder_path")
        if (latent_path) and (len(latent_path) > 0):
            # FIXME: Try both freezing and not freezing the encoder (not sure which one is right now?)
            self.encoder = torch.load(latent_path)['encoder']  # 64 dim latent space
            # ASK:Help, this is only trained on 1, 3, 15, 15 images

            # freeze the encoder weights
            # for param in self.encoder.parameters():
            #     param.requires_grad = False
        else:
            self.encoder = GenEncoder(obs_shape, cfg=encoder_cfg).cuda()  # base CNN; 64 dim latent space

        self.output_dim = self.encoder.output_dim
        self.q_value_head = nn.Linear(self.output_dim, env.single_action_space.n)
        self.encoder_cfg = encoder_cfg
        self.obs_shape = obs_shape

    def forward(self, x):
        x = x.float() / 255.0
        latent = self.encoder(x)
        return self.q_value_head(latent)

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
    run_rl(cfg)


def run_rl(cfg: DictConfig):
    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
            poetry run pip install "stable_baselines3==2.0.0a1"
            """)
    assert cfg.num_envs == 1, "vectorized envs are not supported at the moment"

    cur_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.exp_name}__grid_{cfg.env.grid_size}_obstacles_{cfg.env.num_obstacles}__datetime_{cur_date_time}"
    if cfg.use_wandb:
        import wandb

        wandb.init(
            project=cfg.wandb_project_name,
            entity=cfg.wandb_entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=run_name,
            monitor_gym=True,
            save_code=True,
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
    step_list = []

    succ_ = np.zeros(50)
    ep_idx = 0
    obs, _ = envs.reset(seed=cfg.seed)

    for global_step in tqdm(range(cfg.total_timesteps), desc="global_steps", unit="step"):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(cfg.rl.start_epsilon, cfg.rl.end_epsilon, cfg.rl.exploration_fraction * cfg.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            if obs.ndim == 3:
                obs = np.expand_dims(obs, axis=0)
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)  # rewards contain only 0s or a -1s

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # BUG: this is never happening because of the way the envs.step is setup. There isn't a final_infos
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    succ_[ep_idx % 50] = (info['episode']['r'] > -50)
                    ep_idx += 1

                    if cfg.use_wandb:
                        # converting to wandb logging:
                        wandb.log({"metrics/episodic_return": info["episode"]["r"]}, step=global_step)
                        wandb.log({"metrics/episodic_length": info["episode"]["l"]}, step=global_step)
                        wandb.log({"metrics/success_rate": succ_.mean()}, step=global_step)

        if ("final_observation" in infos) or ("terminal_observation" in infos):
            step_list.append(infos["steps_taken"])
            if (cfg.use_wandb):
                wandb.log({"reward_metrics/steps_to_goal": infos["steps_taken"]}, step=global_step)
                wandb.log({"reward_metrics/episodic_return": infos["episodic_return"]}, step=global_step)
                wandb.log({"reward_metrics/optimal_path_len": infos["optimal_path_length"]}, step=global_step)


        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

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
                    if cfg.use_wandb:
                        wandb.log({"losses/td_loss": loss}, step=global_step)
                        wandb.log({"losses/q_values": old_val.mean().item()}, step=global_step)
                        wandb.log({"metrics/SPS": int(global_step / (time.time() - start_time))}, step=global_step)

                # optimize the model (training the q_network)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % cfg.rl.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        cfg.rl.tau * q_network_param.data + (1.0 - cfg.rl.tau) * target_network_param.data
                    )

        if (global_step % cfg.eval_freq == 0) or (global_step == cfg.total_timesteps - 1):
            q_network.eval()
            rewards = []
            successes = []
            optimal_path_lengths = []
            my_path_lengths = []

            for _ in range(10):
                obs, _ = eval_env.reset()
                terminated, truncated = False, False
                while not (terminated or truncated):
                    q_values = q_network(torch.Tensor(obs).unsqueeze(0).to(device))
                    action = torch.argmax(q_values, dim=1).cpu().numpy().squeeze()
                    obs, reward, terminated, truncated, info = eval_env.step(action)

                rewards.append(info["episodic_return"])
                successes.append(info["success"])
                optimal_path_lengths.append(info["optimal_path_length"])
                my_path_lengths.append(info["steps_taken"])

            diff_path_lengths = np.array(my_path_lengths) - np.array(optimal_path_lengths)
            path_length_ratio = np.array(optimal_path_lengths) / np.array(my_path_lengths)

            if cfg.use_wandb:
                wandb.log({"evals/avg_episodic_return": np.mean(rewards)}, step=global_step)
                wandb.log({"evals/avg_success_rate": np.mean(successes)}, step=global_step)
                wandb.log({"evals/avg_path_length_diff": np.mean(diff_path_lengths)}, step=global_step)
                wandb.log({"evals/avg_path_length_ratio": np.mean(path_length_ratio)}, step=global_step)

            q_network.train()

        if (global_step % cfg.render_freq == 0) or (global_step == cfg.total_timesteps - 1):  # makes sure that it renders at the end too
            frames = []
            # ASK: Do I need to set this to eval mode?
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
                scale_factor = 50
                upscaled_frames = []
                for frame in frames:
                    new_width = frame.shape[1] * scale_factor
                    new_height = frame.shape[0] * scale_factor
                    upscaled_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                    upscaled_frames.append(upscaled_frame)


                clip = ImageSequenceClip(list(upscaled_frames), fps=5)
                temp_video_path = f"render_{global_step}.mp4"

                clip.write_videofile(
                    temp_video_path,
                    codec="libx264",
                    audio=False,
                    logger=None
                )
                # Log the video file to wandb
                wandb.log({f"render/eval": wandb.Video(temp_video_path, format="mp4")})

                os.remove(temp_video_path)
            else:
                imageio.mimsave('test.gif', frames, fps=5)

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
            if cfg.use_wandb:
                wandb.log({"eval/episodic_return": episodic_return}, step=idx)

    envs.close()


if __name__ == "__main__":
    main()
