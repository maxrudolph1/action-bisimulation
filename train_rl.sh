cd rl
# python main.py use_wandb=True env.grid_size=30 env.num_obstacles=5
# python main.py use_wandb=True env.grid_size=10 env.num_obstacles=3

# python main.py use_wandb=True env.grid_size=15 env.num_obstacles=10 exp_name="dqn_k2_ms"  # these params are what ms was trained with
# python main.py use_wandb=True env.grid_size=15 env.num_obstacles=10 exp_name="dqn_no_latent" encoder.latent_encoder_path=""  # these params are what ms was trained with
python main.py use_wandb=True env.grid_size=15 env.num_obstacles=10 exp_name="dqn_k2_ms_frozen"  # these params are what ms was trained with
