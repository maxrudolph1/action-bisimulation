# python main.py algos='[multi_step]' algos.multi_step.use_states_with_same_action=False algos.multi_step.reset_forward_model_every=1e6 wandb=True
# python main.py algos='[multi_step]' algos.multi_step.use_states_with_same_action=False wandb=True
# python main.py algos='[multi_step]' wandb=False
python main.py algos='[multi_step]' algos.multi_step.use_states_with_same_action=True algos.multi_step.train_detached_forward_model=True wandb=True