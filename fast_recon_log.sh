# python main.py algos='[multi_step, single_step, ms_reconstruction]' wandb=True save_ss=False n_epochs=5 batch_size=128
# python main.py algos='[multi_step, single_step, ms_reconstruction]' wandb=False save_ss=False n_epochs=5 batch_size=128

# python main.py algos='[single_step, ms_reconstruction]' wandb=False save_ss=False n_epochs=5 batch_size=128

# NOTE: SINGLE STEP RECON
python main.py algos='[single_step, ms_reconstruction]' wandb=True save_ss=False
# python main.py algos='[single_step, ms_reconstruction]' wandb=False save_ss=False
