cd ./rl
python main.py use_wandb=False env.grid_size=15 env.num_obstacles=20 env.max_timesteps=50 exp_name="vanilla_SANDBOX_dqn_s5" seed=5 encoder.latent_encoder_path=""
