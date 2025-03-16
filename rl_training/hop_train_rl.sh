cd ../rl


# VANILLA
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=100 env.max_timesteps=170 exp_name="vanilla_dqn" seed=3 encoder.latent_encoder_path=""
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=100 env.max_timesteps=170 exp_name="vanilla_dqn" seed=5 encoder.latent_encoder_path=""
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=100 env.max_timesteps=170 exp_name="vanilla_dqn" seed=7 encoder.latent_encoder_path=""

# ACRO K=1
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=100 env.max_timesteps=170 exp_name="acro_k1_dqn" seed=3 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/saved_models/grid_30_obstcls_100_smpls_1250000/acro_sweeps_k1_l1_0.0001.pt"
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=100 env.max_timesteps=170 exp_name="acro_k1_dqn" seed=5 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/saved_models/grid_30_obstcls_100_smpls_1250000/acro_sweeps_k1_l1_0.0001.pt"
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=100 env.max_timesteps=170 exp_name="acro_k1_dqn" seed=7 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/saved_models/grid_30_obstcls_100_smpls_1250000/acro_sweeps_k1_l1_0.0001.pt"

# ACRO K=2
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=100 env.max_timesteps=170 exp_name="acro_k2_dqn" seed=3 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/saved_models/grid_30_obstcls_100_smpls_1250000/acro_sweeps_k2_l1_0.002.pt"
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=100 env.max_timesteps=170 exp_name="acro_k2_dqn" seed=5 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/saved_models/grid_30_obstcls_100_smpls_1250000/acro_sweeps_k2_l1_0.002.pt"
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=100 env.max_timesteps=170 exp_name="acro_k2_dqn" seed=7 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/saved_models/grid_30_obstcls_100_smpls_1250000/acro_sweeps_k2_l1_0.002.pt"

# ACRO K=3
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=100 env.max_timesteps=170 exp_name="acro_k3_dqn" seed=3 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/saved_models/grid_30_obstcls_100_smpls_1250000/acro_sweeps_k3_l1_0.0002.pt"
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=100 env.max_timesteps=170 exp_name="acro_k3_dqn" seed=5 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/saved_models/grid_30_obstcls_100_smpls_1250000/acro_sweeps_k3_l1_0.0002.pt"
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=100 env.max_timesteps=170 exp_name="acro_k3_dqn" seed=7 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/saved_models/grid_30_obstcls_100_smpls_1250000/acro_sweeps_k3_l1_0.0002.pt"

# MS_ACRO K=1
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=100 env.max_timesteps=170 exp_name="ms_acro_k1" seed=3 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/saved_models/grid_30_obstcls_100_smpls_1250000/ms_acro_k1_l1_0001_swps_gamma_0.9.pt"
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=100 env.max_timesteps=170 exp_name="ms_acro_k1" seed=5 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/saved_models/grid_30_obstcls_100_smpls_1250000/ms_acro_k1_l1_0001_swps_gamma_0.9.pt"
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=100 env.max_timesteps=170 exp_name="ms_acro_k1" seed=7 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/saved_models/grid_30_obstcls_100_smpls_1250000/ms_acro_k1_l1_0001_swps_gamma_0.9.pt"

# MS_ACRO K=2
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=100 env.max_timesteps=170 exp_name="ms_acro_k2" seed=3 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/saved_models/grid_30_obstcls_100_smpls_1250000/ms_acro_k2_l1_002_swps_gamma_0.8.pt"
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=100 env.max_timesteps=170 exp_name="ms_acro_k2" seed=5 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/saved_models/grid_30_obstcls_100_smpls_1250000/ms_acro_k2_l1_002_swps_gamma_0.8.pt"
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=100 env.max_timesteps=170 exp_name="ms_acro_k2" seed=7 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/saved_models/grid_30_obstcls_100_smpls_1250000/ms_acro_k2_l1_002_swps_gamma_0.8.pt"

# MS_ACRO K=3
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=100 env.max_timesteps=170 exp_name="ms_acro_k3" seed=3 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/saved_models/grid_30_obstcls_100_smpls_1250000/ms_acro_k3_l1_0002_swps_gamma_0.85.pt"
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=100 env.max_timesteps=170 exp_name="ms_acro_k3" seed=5 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/saved_models/grid_30_obstcls_100_smpls_1250000/ms_acro_k3_l1_0002_swps_gamma_0.85.pt"
python main.py use_wandb=True env.grid_size=30 env.num_obstacles=100 env.max_timesteps=170 exp_name="ms_acro_k3" seed=7 encoder.latent_encoder_path="/home/ekuo/bisim/action-bisimulation/saved_models/grid_30_obstcls_100_smpls_1250000/ms_acro_k3_l1_0002_swps_gamma_0.85.pt"
