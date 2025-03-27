# SWEEPING OVER L1 PENALTY (0.0001, 0.0005, 0.001, 0.005, 0.01)
python main.py algos='[single_step]' name='ss_l1_0001_swps' n_epochs=14 algos.single_step.l1_penalty=0.0001 algos.acro.dynamic_l1_penalty=False wandb=True train_evaluators=False eval_encoder='single_step'
python main.py algos='[single_step]' name='ss_l1_0005_swps' n_epochs=14 algos.single_step.l1_penalty=0.0005 algos.acro.dynamic_l1_penalty=False wandb=True train_evaluators=False eval_encoder='single_step'
python main.py algos='[single_step]' name='ss_l1_001_swps'  n_epochs=14 algos.single_step.l1_penalty=0.001  algos.acro.dynamic_l1_penalty=False wandb=True train_evaluators=False eval_encoder='single_step'
python main.py algos='[single_step]' name='ss_l1_005_swps'  n_epochs=14 algos.single_step.l1_penalty=0.005  algos.acro.dynamic_l1_penalty=False wandb=True train_evaluators=False eval_encoder='single_step'
python main.py algos='[single_step]' name='ss_l1_01_swps'   n_epochs=14 algos.single_step.l1_penalty=0.01   algos.acro.dynamic_l1_penalty=False wandb=True train_evaluators=False eval_encoder='single_step'
