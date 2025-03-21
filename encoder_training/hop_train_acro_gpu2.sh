# SWEEPING OVER L1 PENALTY (0.0001, 0.0005, 0.001, 0.005, 0.01)
# K=2
python main.py algos='[acro]' n_epochs=35 algos.acro.l1_penalty=0.0001 algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=2 wandb=True train_evaluators=False eval_encoder='acro'
python main.py algos='[acro]' n_epochs=35 algos.acro.l1_penalty=0.0005 algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=2 wandb=True train_evaluators=False eval_encoder='acro'
python main.py algos='[acro]' n_epochs=35 algos.acro.l1_penalty=0.001  algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=2 wandb=True train_evaluators=False eval_encoder='acro'
python main.py algos='[acro]' n_epochs=35 algos.acro.l1_penalty=0.005  algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=2 wandb=True train_evaluators=False eval_encoder='acro'
python main.py algos='[acro]' n_epochs=35 algos.acro.l1_penalty=0.01   algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=2 wandb=True train_evaluators=False eval_encoder='acro'
