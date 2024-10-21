# python main.py algos='[multi_step, single_step, reconstruction]' wandb=True save_ss=False n_epochs=5 batch_size=128
# python main.py algos='[multi_step, single_step, reconstruction]' wandb=False save_ss=False n_epochs=5 batch_size=128

# python main.py algos='[single_step, reconstruction]' wandb=False save_ss=False n_epochs=5 batch_size=128

# NOTE: SINGLE STEP RECON
python main.py algos='[single_step, reconstruction]' wandb=True save_ss=False
# python main.py algos='[single_step, reconstruction]' wandb=False save_ss=False
