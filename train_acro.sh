# python main.py algos='[acro]' algos.acro.k_steps=2 wandb=True
# python main.py algos='[acro]' n_epochs=20 algos.acro.k_steps=5 wandb=True

# l1 regularization sweeps 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1
# python main.py algos='[acro]' n_epochs=20 algos.acro.l1_penalty=0.00001 algos.acro.dynamic_l1_penalty=True algos.acro.k_steps=5 wandb=True
# python main.py algos='[acro]' n_epochs=20 algos.acro.l1_penalty=0.00001 algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=5 wandb=True

# python main.py algos='[acro]' n_epochs=20 algos.acro.l1_penalty=0.0001 algos.acro.dynamic_l1_penalty=True algos.acro.k_steps=5 wandb=True
# python main.py algos='[acro]' n_epochs=20 algos.acro.l1_penalty=0.0001 algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=5 wandb=True


# python main.py algos='[acro]' n_epochs=20 algos.acro.l1_penalty=1 algos.acro.dynamic_l1_penalty=True algos.acro.k_steps=5 wandb=True
# python main.py algos='[acro]' n_epochs=20 algos.acro.l1_penalty=1 algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=5 wandb=True

# python main.py algos='[acro]' n_epochs=20 algos.acro.l1_penalty=0.01 algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=3 wandb=True
# python main.py algos='[acro]' n_epochs=20 algos.acro.l1_penalty=0.1 algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=3 wandb=True
# python main.py algos='[acro]' n_epochs=20 algos.acro.l1_penalty=0.001 algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=3 wandb=True

# these are probably the best ones right now
# python main.py algos='[acro]' n_epochs=80 algos.acro.l1_penalty=0.001 algos.acro.dynamic_l1_penalty=True algos.acro.k_steps=2 wandb=True # DEFAULT L1
# python main.py algos='[acro]' n_epochs=80 algos.acro.l1_penalty=0.01 algos.acro.dynamic_l1_penalty=True algos.acro.k_steps=2 wandb=True
# python main.py algos='[acro]' n_epochs=80 algos.acro.l1_penalty=0.1 algos.acro.dynamic_l1_penalty=True algos.acro.k_steps=2 wandb=True

# python main.py algos='[acro]' n_epochs=80 algos.acro.l1_penalty=0.001 algos.acro.dynamic_l1_penalty=True algos.acro.k_steps=3 wandb=True # DEFAULT L1
# python main.py algos='[acro]' n_epochs=80 algos.acro.l1_penalty=0.01 algos.acro.dynamic_l1_penalty=True algos.acro.k_steps=3 wandb=True
# python main.py algos='[acro]' n_epochs=80 algos.acro.l1_penalty=0.1 algos.acro.dynamic_l1_penalty=True algos.acro.k_steps=3 wandb=True

# python main.py algos='[acro]' n_epochs=80 algos.acro.l1_penalty=0.001 algos.acro.dynamic_l1_penalty=True algos.acro.k_steps=4 wandb=True # DEFAULT L1
# python main.py algos='[acro]' n_epochs=80 algos.acro.l1_penalty=0.01 algos.acro.dynamic_l1_penalty=True algos.acro.k_steps=4 wandb=True
# python main.py algos='[acro]' n_epochs=80 algos.acro.l1_penalty=0.1 algos.acro.dynamic_l1_penalty=True algos.acro.k_steps=4 wandb=True

# sweeps over k (1 = obs_next = single step)
# python main.py algos='[acro]' n_epochs=5 algos.acro.k_steps=3 wandb=False

# python main.py algos='[acro]' n_epochs=14 algos.acro.k_steps=1 wandb=False
# python main.py algos='[acro]' n_epochs=80 algos.acro.k_steps=2 wandb=True
# python main.py algos='[acro]' n_epochs=80 algos.acro.k_steps=3 wandb=True

python main.py algos='[acro]' n_epochs=14 algos.acro.l1_penalty=0.01   algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=1 wandb=True
python main.py algos='[acro]' n_epochs=14 algos.acro.l1_penalty=0.005  algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=1 wandb=True
python main.py algos='[acro]' n_epochs=14 algos.acro.l1_penalty=0.002  algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=1 wandb=True
python main.py algos='[acro]' n_epochs=14 algos.acro.l1_penalty=0.001  algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=1 wandb=True
python main.py algos='[acro]' n_epochs=14 algos.acro.l1_penalty=0.0005 algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=1 wandb=True
python main.py algos='[acro]' n_epochs=14 algos.acro.l1_penalty=0.0002 algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=1 wandb=True
python main.py algos='[acro]' n_epochs=14 algos.acro.l1_penalty=0.0001 algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=1 wandb=True

python main.py algos='[acro]' n_epochs=80 algos.acro.l1_penalty=0.01   algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=2 wandb=True
python main.py algos='[acro]' n_epochs=80 algos.acro.l1_penalty=0.005  algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=2 wandb=True
python main.py algos='[acro]' n_epochs=80 algos.acro.l1_penalty=0.002  algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=2 wandb=True
python main.py algos='[acro]' n_epochs=80 algos.acro.l1_penalty=0.001  algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=2 wandb=True
python main.py algos='[acro]' n_epochs=80 algos.acro.l1_penalty=0.0005 algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=2 wandb=True
python main.py algos='[acro]' n_epochs=80 algos.acro.l1_penalty=0.0002 algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=2 wandb=True
python main.py algos='[acro]' n_epochs=80 algos.acro.l1_penalty=0.0001 algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=2 wandb=True

python main.py algos='[acro]' n_epochs=80 algos.acro.l1_penalty=0.01   algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=3 wandb=True
python main.py algos='[acro]' n_epochs=80 algos.acro.l1_penalty=0.005  algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=3 wandb=True
python main.py algos='[acro]' n_epochs=80 algos.acro.l1_penalty=0.002  algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=3 wandb=True
python main.py algos='[acro]' n_epochs=80 algos.acro.l1_penalty=0.001  algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=3 wandb=True
python main.py algos='[acro]' n_epochs=80 algos.acro.l1_penalty=0.0005 algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=3 wandb=True
python main.py algos='[acro]' n_epochs=80 algos.acro.l1_penalty=0.0002 algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=3 wandb=True
python main.py algos='[acro]' n_epochs=80 algos.acro.l1_penalty=0.0001 algos.acro.dynamic_l1_penalty=False algos.acro.k_steps=3 wandb=True
