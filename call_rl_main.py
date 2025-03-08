import wandb
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from rl.main import run_rl


def call_rl(
    name="sandbox_rl",
    grid_size=15,
    num_obstacles=10,
    max_timesteps=100,
    total_timesteps=1000000,
    latent_encoder_path="",
    use_wandb=True,
):
    GlobalHydra.instance().clear()

    with hydra.initialize(config_path="rl/configs"):
        cfg = hydra.compose(config_name="config", overrides=[
            f"use_wandb={use_wandb}",
            f"exp_name='{name}'",
            f"total_timesteps={total_timesteps}",
            f"env.grid_size={grid_size}",
            f"env.num_obstacles={num_obstacles}",
            f"env.max_timesteps={max_timesteps}",
            f"encoder.latent_encoder_path='{latent_encoder_path}'",
        ])

    print(OmegaConf.to_yaml(cfg))
    run_rl(cfg)
    wandb.finish()


if __name__ == "__main__":
    call_rl(use_wandb=False)
