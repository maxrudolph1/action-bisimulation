# python main.py algos='[multi_step]' algos.multi_step.use_states_with_same_action=False algos.multi_step.reset_forward_model_every=1e6 wandb=True
# python main.py algos='[multi_step]' algos.multi_step.use_states_with_same_action=False wandb=True
# python main.py algos='[multi_step]' wandb=False
# 3000 is every epoch
python main.py algos='[multi_step]' algos.multi_step.use_states_with_same_action=True algos.multi_step.train_detached_forward_model=True algos.multi_step.reset_forward_model_every=15000 wandb=True
