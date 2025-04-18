
# first, need to run the data collection: 
python collect_nav2d_data.py --grid-size 15 --num-obstacles 20 --obstacle-size 1 --k-step-action 4 --epsilon 0.5 --seed 11 --size 1000000 --save-path ./data --name paper_default






# second, train infonce encoder

python main.py algos='[infonce]'  wandb=True train_evaluators=False eval_encoder='infonce'

# attempt to increase temperature and lower learning rate
python main.py algos='[infonce]' algos.infonce.temperature=0.3 algos.infonce.learning_rate=1e-4 wandb=True train_evaluators=False eval_encoder='infonce'

# attempt run to add projector
python main.py algos='[infonce]' algos.infonce.temperature=0.3 algos.infonce.learning_rate=1e-4 wandb=True train_evaluators=False eval_encoder='infonce'



# ------

# python main.py algos='[infonce]' n_epochs=14 algos.acro.l1_penalty=0.002  algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=1 wandb=True train_evaluators=False eval_encoder='acro'
python main.py algos='[multi_step]' name='ms_infonce' n_epochs=20 wandb=True train_evaluators=False algos.multi_step.base_case_path='/data/rhearai/action-bisimulation/results/infonce_2025-04-11_09-08-01_ts_19540/infonce.pt'

# python main.py use_wandb=True env.grid_size=30 env.num_obstacles=50 env.max_timesteps=70 total_timesteps=550000 exp_name="vanilla_dqn" encoder.latent_encoder_path=""

# python main.py algos='[multi_step]' name='ms_acro_k2_gamma_85' wandb=True train_evaluators=False algos.multi_step.base_case_path='/home/ekuo/bisim/evan-master-action-bisimulation/results/acro_sweeps_k2_l1_0.0001_2025-01-25_13-44-59_ts_625120/acro.pt'
python main.py use_wandb=True env.grid_size=30 env.max_timesteps=70 total_timesteps=550000  env.num_obstacles=15 exp_name="infonce_ms_rl" encoder.latent_encoder_path="/data/rhearai/action-bisimulation/results/infonce_2025-04-11_09-33-47_ts_19540/multi_step.pt"