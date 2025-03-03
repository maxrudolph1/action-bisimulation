cd rl
# python main.py use_wandb=True env.grid_size=30 env.num_obstacles=5
# python main.py use_wandb=True env.grid_size=10 env.num_obstacles=3

# Grid matches ms dataset trained on (15x15), 10 obstacles
# python main.py use_wandb=True env.grid_size=15 env.num_obstacles=10 exp_name="dqn_k2_ms"
# python main.py use_wandb=True env.grid_size=15 env.num_obstacles=10 exp_name="dqn_no_latent" encoder.latent_encoder_path=""

# BIG GRID
# python main.py use_wandb=True env.grid_size=30 env.num_obstacles=15 env.max_timesteps=100 total_timesteps=600000 exp_name="dqn_acro_k2_1e-4"
# python main.py use_wandb=True env.grid_size=30 env.num_obstacles=15 env.max_timesteps=100 total_timesteps=600000 exp_name="dqn_vanilla" encoder.latent_encoder_path=""

# BIG GRID MS sweeps trial
# python main.py use_wandb=True env.grid_size=30 env.num_obstacles=15 env.max_timesteps=100 total_timesteps=600000 exp_name="dqn_k2_1e-4_ms_gamma_0.95" encoder.latent_encoder_path="/home/ekuo/bisim/master-branch-actbisim/results/ms_acro_k2_gamma_95_2025-02-21_21-34-58_ts_117240/multi_step.pt"
# python main.py use_wandb=True env.grid_size=30 env.num_obstacles=15 env.max_timesteps=100 total_timesteps=600000 exp_name="dqn_k2_1e-4_ms_gamma_0.88" encoder.latent_encoder_path="/home/ekuo/bisim/master-branch-actbisim/results/ms_acro_k2_gamma_88_2025-02-21_19-21-03_ts_117240/multi_step.pt"
# python main.py use_wandb=True env.grid_size=30 env.num_obstacles=15 env.max_timesteps=100 total_timesteps=600000 exp_name="dqn_k2_1e-4_ms_gamma_0.85" encoder.latent_encoder_path="/home/ekuo/bisim/master-branch-actbisim/results/ms_acro_k2_gamma_85_2025-02-21_17-08-04_ts_117240/multi_step.pt"
# python main.py use_wandb=True env.grid_size=30 env.num_obstacles=15 env.max_timesteps=100 total_timesteps=600000 exp_name="dqn_k2_1e-4_ms_gamma_0.82" encoder.latent_encoder_path="/home/ekuo/bisim/master-branch-actbisim/results/ms_acro_k2_gamma_82_2025-02-21_14-54-35_ts_117240/multi_step.pt"


# grid 30
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=50 env.max_timesteps=70 total_timesteps=550000 exp_name="vanilla_dqn" encoder.latent_encoder_path=""
# python main.py use_wandb=True env.grid_size=30 env.num_obstacles=55 env.max_timesteps=70 total_timesteps=550000 exp_name="vanilla_dqn" encoder.latent_encoder_path=""
# python main.py use_wandb=True env.grid_size=30 env.num_obstacles=60 env.max_timesteps=70 total_timesteps=550000 exp_name="vanilla_dqn" encoder.latent_encoder_path=""

# grid 40
# python main.py use_wandb=True env.grid_size=40 env.num_obstacles=25 env.max_timesteps=90 total_timesteps=600000 exp_name="vanilla_dqn" encoder.latent_encoder_path=""
# python main.py use_wandb=True env.grid_size=40 env.num_obstacles=30 env.max_timesteps=90 total_timesteps=600000 exp_name="vanilla_dqn" encoder.latent_encoder_path=""

# grid 45
# python main.py use_wandb=True env.grid_size=45 env.num_obstacles=30 env.max_timesteps=100 total_timesteps=600000 exp_name="vanilla_dqn" encoder.latent_encoder_path=""
# python main.py use_wandb=True env.grid_size=45 env.num_obstacles=35 env.max_timesteps=100 total_timesteps=600000 exp_name="vanilla_dqn" encoder.latent_encoder_path=""
