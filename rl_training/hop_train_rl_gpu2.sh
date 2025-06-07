cd ./rl

# Chosen SS training runs
python main.py use_wandb=True env.grid_size=15 env.num_obstacles=20 env.max_timesteps=50 exp_name="dqn_ss_chosen_s1" seed=1 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/results/ss_l1_002_swps_grd_15_obstcls_20_smpls_1250000_2025-05-29_05-55-32_ts_68362/single_step.pt"
python main.py use_wandb=True env.grid_size=15 env.num_obstacles=20 env.max_timesteps=50 exp_name="dqn_ss_chosen_s2" seed=2 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/results/ss_l1_002_swps_grd_15_obstcls_20_smpls_1250000_2025-05-29_05-55-32_ts_68362/single_step.pt"
python main.py use_wandb=True env.grid_size=15 env.num_obstacles=20 env.max_timesteps=50 exp_name="dqn_ss_chosen_s3" seed=3 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/results/ss_l1_002_swps_grd_15_obstcls_20_smpls_1250000_2025-05-29_05-55-32_ts_68362/single_step.pt"
python main.py use_wandb=True env.grid_size=15 env.num_obstacles=20 env.max_timesteps=50 exp_name="dqn_ss_chosen_s4" seed=4 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/results/ss_l1_002_swps_grd_15_obstcls_20_smpls_1250000_2025-05-29_05-55-32_ts_68362/single_step.pt"
python main.py use_wandb=True env.grid_size=15 env.num_obstacles=20 env.max_timesteps=50 exp_name="dqn_ss_chosen_s5" seed=5 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/results/ss_l1_002_swps_grd_15_obstcls_20_smpls_1250000_2025-05-29_05-55-32_ts_68362/single_step.pt"
