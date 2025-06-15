cd ./rl

python main.py use_wandb=True env.grid_size=15 env.num_obstacles=20 env.max_timesteps=50 exp_name="dqn_nce_s1" seed=1 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/results/nce_temp_0001_2025-06-14_00-42-14_ts_48830/nce.pt"
python main.py use_wandb=True env.grid_size=15 env.num_obstacles=20 env.max_timesteps=50 exp_name="dqn_nce_s2" seed=2 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/results/nce_temp_0001_2025-06-14_00-42-14_ts_48830/nce.pt"
