# python main.py algos=multi_step wandb=True algos.multi_step.forward_model_steps_per_batch=1 algos.multi_step.warm_start_ms_with_ss=False
# python main.py algos=multi_step wandb=True algos.multi_step.forward_model_steps_per_batch=3 algos.multi_step.warm_start_ms_with_ss=False
# python main.py algos=multi_step wandb=True algos.multi_step.forward_model_steps_per_batch=5 algos.multi_step.warm_start_ms_with_ss=False
# python main.py algos=multi_step wandb=True algos.multi_step.forward_model_steps_per_batch=7 algos.multi_step.warm_start_ms_with_ss=False

python main.py algos=multi_step wandb=True algos.multi_step.forward_model_steps_per_batch=1 algos.multi_step.warm_start_ms_with_ss=True algos.multi_step.use_states_with_same_action=False
python main.py algos=multi_step wandb=True algos.multi_step.forward_model_steps_per_batch=3 algos.multi_step.warm_start_ms_with_ss=True algos.multi_step.use_states_with_same_action=False
python main.py algos=multi_step wandb=True algos.multi_step.forward_model_steps_per_batch=5 algos.multi_step.warm_start_ms_with_ss=True algos.multi_step.use_states_with_same_action=False
python main.py algos=multi_step wandb=True algos.multi_step.forward_model_steps_per_batch=7 algos.multi_step.warm_start_ms_with_ss=True algos.multi_step.use_states_with_same_action=False
