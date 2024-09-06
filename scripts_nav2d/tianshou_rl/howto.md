# Hello Caleb!

If you want to train tianshou models here, you can use the `run_rl.sh`. That file will call `train_dqn.py` which has a bunch of flags you can change; the most important being: 
- `--pretrained-encoder-path` 
- `--freeze-encoder`. 


If you want to modify the neural network architecture that the DQN is using, you will need to go to line 128-134 of `train_dqn.py`.

The networks are kept in `nav2d_representation/nets.py`. It is currently using DQNHER model but that's just from legacy code. If you want, you can clean it up a bit to your liking ;). Otherwise, if nothing is making sense, I can clean it up.

Warm Regards from 5.408F, 
Max