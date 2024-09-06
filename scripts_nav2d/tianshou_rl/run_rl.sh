#!/bin/bash

# Define the values for num-obstacles and grid-size
num_obstacle=10
grid_size=15
runs=(0 1 2 3 4 5)

# encoder_path="/home/mrudolph/documents/actbisim/scripts_nav2d/debug/acro/acro_final.pt"
# encoder_path="/home/mrudolph/documents/actbisim/scripts_nav2d/debug/curl/curl_final.pt"
# encoder_path="/home/mrudolph/documents/actbisim/scripts_nav2d/debug/multi_step/multi_step_final.pt"
# encoder_path="/home/mrudolph/documents/actbisim/scripts_nav2d/debug/multi_step/single_step_final.pt"
# encoder_path="/home/mrudolph/documents/actbisim/scripts_nav2d/debug/multi_step_same_act_ss_warmup+continual_training/multi_step_final.pt"
# encoder_path="/home/mrudolph/documents/actbisim/scripts_nav2d/debug/multi_step_no_gt/multi_step_final.pt"
# encoder_path="/home/mrudolph/documents/actbisim/scripts_nav2d/debug/bvae/bvae_final.pt"
# encoder_path="/home/mrudolph/documents/actbisim/scripts_nav2d/debug/curl_normal_noise/curl_final.pt"
# encoder_path="/home/mrudolph/documents/actbisim/scripts_nav2d/debug/acro_long_train/acro_final.pt"
encoder_path="/home/mrudolph/documents/actbisim/scripts_nav2d/debug/learned-obs-forward-test/multi_step_final.pt"

# encoder_path="/home/mrudolph/documents/actbisim/scripts_nav2d/debug/gamma_0.99_gt_forward_long/single_step_final.pt"

for run in "${runs[@]}"
do

    name_flag="bisim-learned-obs-forward_unfrozen_run_${run}"
    # Execute the Python command with the appropriate arguments
    python train_dqn.py --num-obstacles $num_obstacle --grid-size $grid_size --name $name_flag --pretrained-encoder-path $encoder_path #--freeze-encoder

done

# python scripts_nav2d/tianshou_rl/train_dqn.py --num-obstacles 10 --grid-size 15 --name 0 --pretrained-encoder-path $encoder_path
# python scripts_nav2d/tianshou_rl/train_dqn.py --num-obstacles 10 --grid-size 15 --name base --pretrained-encoder-path /data/calebc/bisim_data/datasets/model/multi_step_final.pt --freeze-encoder --logdir /data/calebc/bisim_data/datasets/rl/ --use-gen-nets