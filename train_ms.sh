# python main.py algos='[multi_step]' algos.multi_step.use_states_with_same_action=False algos.multi_step.reset_forward_model_every=1e6 wandb=True
# python main.py algos='[multi_step]' algos.multi_step.use_states_with_same_action=False wandb=True
# python main.py algos='[multi_step]' wandb=False
# 3000 is every epoch
# python main.py algos='[multi_step]' algos.multi_step.use_states_with_same_action=True algos.multi_step.reset_forward_model_every=15000 wandb=True

# ACRO with BISIM
python main.py algos='[multi_step]' algos.multi_step.gamma=0.5  algos.multi_step.use_states_with_same_action=True algos.multi_step.reset_forward_model_every=40000 wandb=True
python main.py algos='[multi_step]' algos.multi_step.gamma=0.6  algos.multi_step.use_states_with_same_action=True algos.multi_step.reset_forward_model_every=100000 wandb=True
python main.py algos='[multi_step]' algos.multi_step.gamma=0.7  algos.multi_step.use_states_with_same_action=True algos.multi_step.reset_forward_model_every=100000 wandb=True
python main.py algos='[multi_step]' algos.multi_step.gamma=0.8  algos.multi_step.use_states_with_same_action=True algos.multi_step.reset_forward_model_every=40000 wandb=True
python main.py algos='[multi_step]' algos.multi_step.gamma=0.82 algos.multi_step.use_states_with_same_action=True algos.multi_step.reset_forward_model_every=40000 wandb=True
python main.py algos='[multi_step]' algos.multi_step.gamma=0.85 algos.multi_step.use_states_with_same_action=True algos.multi_step.reset_forward_model_every=40000 wandb=True
python main.py algos='[multi_step]' algos.multi_step.gamma=0.88 algos.multi_step.use_states_with_same_action=True algos.multi_step.reset_forward_model_every=40000 wandb=True
python main.py algos='[multi_step]' algos.multi_step.gamma=0.9  algos.multi_step.use_states_with_same_action=True algos.multi_step.reset_forward_model_every=40000 wandb=True
python main.py algos='[multi_step]' algos.multi_step.gamma=0.95 algos.multi_step.use_states_with_same_action=True algos.multi_step.reset_forward_model_every=40000 wandb=True
python main.py algos='[multi_step]' algos.multi_step.gamma=0.98 algos.multi_step.use_states_with_same_action=True algos.multi_step.reset_forward_model_every=40000 wandb=True
