cd rl
# python main.py use_wandb=True env.grid_size=30 env.num_obstacles=5
# python main.py use_wandb=True env.grid_size=10 env.num_obstacles=3

# Grid matches ms dataset trained on (15x15), 10 obstacles
# python main.py use_wandb=True env.grid_size=15 env.num_obstacles=10 exp_name="dqn_k2_ms"
# python main.py use_wandb=True env.grid_size=15 env.num_obstacles=10 exp_name="dqn_no_latent" encoder.latent_encoder_path=""

# BIG GRID
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=15 env.max_timesteps=100 total_timesteps=600000 exp_name="dqn_acro_k2_1e-4"
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=15 env.max_timesteps=100 total_timesteps=600000 exp_name="dqn_no_latent" encoder.latent_encoder_path=""
