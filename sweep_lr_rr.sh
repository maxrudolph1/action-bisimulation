# sweeping over fw model reset rate and learning rate for fw : ms update ratio of 4 : 1

forward_model_reset_rates=(3000 6000 9000 12000)
learning_rates=(0.0001 0.00005 0.00001)


ratio=20 #fixed ratio of n_epochs/encoder_update_freq - keep at 20?

for lr in "${learning_rates[@]}"; do
  for reset_rate in "${forward_model_reset_rates[@]}"; do
    encoder_update_freq=4
    n_epochs=$((ratio * encoder_update_freq))

    cmd="python main.py algos='[multi_step]' \
    algos.multi_step.use_states_with_same_action=True \
    algos.multi_step.train_detached_forward_model=True \
    algos.multi_step.reset_forward_model_every=${reset_rate} \
    n_epochs=${n_epochs} \
    algos.multi_step.encoder_update_freq=${encoder_update_freq} \
    algos.multi_step.learning_rate=${lr} \
    wandb=True"

    echo "Running command: $cmd"
    eval $cmd

  done
done


# old stuff
# python main.py algos='[multi_step]' algos.multi_step.use_states_with_same_action=True algos.multi_step.train_detached_forward_model=True algos.multi_step.reset_forward_model_every=3000 n_epochs=80 algos.multi_step.encoder_update_freq=4  wandb=True
# python main.py algos='[multi_step]' algos.multi_step.use_states_with_same_action=True algos.multi_step.train_detached_forward_model=True algos.multi_step.reset_forward_model_every=6000 n_epochs=80 algos.multi_step.encoder_update_freq=4  wandb=True
# python main.py algos='[multi_step]' algos.multi_step.use_states_with_same_action=True algos.multi_step.train_detached_forward_model=True algos.multi_step.reset_forward_model_every=9000 n_epochs=80 algos.multi_step.encoder_update_freq=4  wandb=True
# python main.py algos='[multi_step]' algos.multi_step.use_states_with_same_action=True algos.multi_step.train_detached_forward_model=True algos.multi_step.reset_forward_model_every=12000 n_epochs=80 algos.multi_step.encoder_update_freq=4  wandb=True

# # python main.py algos='[multi_step]' algos.multi_step.use_states_with_same_action=True algos.multi_step.train_detached_forward_model=True algos.multi_step.reset_forward_model_every=6000 wandb=True
# # python main.py algos='[multi_step]' algos.multi_step.use_states_with_same_action=True algos.multi_step.train_detached_forward_model=True algos.multi_step.reset_forward_model_every=9000 wandb=True
# # python main.py algos='[multi_step]' algos.multi_step.use_states_with_same_action=True algos.multi_step.train_detached_forward_model=True algos.multi_step.reset_forward_model_every=12000 wandb=True
