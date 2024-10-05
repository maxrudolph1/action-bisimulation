python main.py algos='single_step' wandb=True algos.single_step.l1_penalty=0.0
python main.py algos='single_step' wandb=True algos.single_step.l1_penalty=1e-6
python main.py algos='single_step' wandb=True algos.single_step.l1_penalty=1e-5
python main.py algos='single_step' wandb=True algos.single_step.l1_penalty=1e-4
python main.py algos='single_step' wandb=True algos.single_step.l1_penalty=1e-3
python main.py algos='single_step' wandb=True algos.single_step.l1_penalty=1e-2 # had best performance efficient-glade-34
python main.py algos='single_step' wandb=True algos.single_step.l1_penalty=1e-1
python main.py algos='single_step' wandb=True algos.single_step.l1_penalty=0.5
python main.py algos='single_step' wandb=True algos.single_step.l1_penalty=1.0
python main.py algos='single_step' wandb=True # test default # also had pretty good performance laced-snow-38
# default with dynamic penalty set to false: fancy-hill-39