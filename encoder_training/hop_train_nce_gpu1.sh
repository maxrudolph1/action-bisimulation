# K2
python main.py algos='[nce]' name='nce_l1_temp_05_k2' algos.nce.temperature=0.05 algos.nce.k_steps=2 n_epochs=3 wandb=True train_evaluators=False eval_encoder='nce'
python main.py algos='[nce]' name='nce_l1_temp_02_k2' algos.nce.temperature=0.02 algos.nce.k_steps=2 n_epochs=3 wandb=True train_evaluators=False eval_encoder='nce'
python main.py algos='[nce]' name='nce_l1_temp_01_k2' algos.nce.temperature=0.01 algos.nce.k_steps=2 n_epochs=3 wandb=True train_evaluators=False eval_encoder='nce'

# K3
python main.py algos='[nce]' name='nce_l1_temp_05_k3' algos.nce.temperature=0.05 algos.nce.k_steps=3 n_epochs=3 wandb=True train_evaluators=False eval_encoder='nce'
python main.py algos='[nce]' name='nce_l1_temp_02_k3' algos.nce.temperature=0.02 algos.nce.k_steps=3 n_epochs=3 wandb=True train_evaluators=False eval_encoder='nce'
python main.py algos='[nce]' name='nce_l1_temp_01_k3' algos.nce.temperature=0.01 algos.nce.k_steps=3 n_epochs=3 wandb=True train_evaluators=False eval_encoder='nce'

# K5
python main.py algos='[nce]' name='nce_l1_temp_05_k5' algos.nce.temperature=0.05 algos.nce.k_steps=5 n_epochs=4 wandb=True train_evaluators=False eval_encoder='nce'
python main.py algos='[nce]' name='nce_l1_temp_02_k5' algos.nce.temperature=0.02 algos.nce.k_steps=5 n_epochs=4 wandb=True train_evaluators=False eval_encoder='nce'
python main.py algos='[nce]' name='nce_l1_temp_01_k5' algos.nce.temperature=0.01 algos.nce.k_steps=5 n_epochs=4 wandb=True train_evaluators=False eval_encoder='nce'
