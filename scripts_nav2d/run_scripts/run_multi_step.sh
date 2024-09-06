cd ../

# Define an array of gamma values
gamma_values=(0.99)

path="/data/calebc/bisim_data/"
# Loop through each gamma value and run the Python script
for gamma in "${gamma_values[@]}"; do
    python scripts_nav2d/offline.py --logdir "debug/model_idk" \
#     python offline.py --logdir "debug/learned-obs-forward-test" \
        --decoder-lr 0.00005 \
        --dataset "${path}"datasets/nav2d_dataset_s0_e0.5_size1000000__k_steps_10.hdf5 \
        --multi-step-gamma "${gamma}" \
        --model-to-decode ms \
        --k-steps 7 \
        --k-steps-dyn 2 \
        --k-step-forward-weight 1.0 \
        --multi-step-forward-loss l1 \
	    --n-epochs 160 \
        --ss-stop-epochs 100 \
        --ss-warmup-epochs 2 \
        --batch_size 1024 \
        --use-gt-forward-model \
        --train-single-step \
        --train-multi-step \
        --use-learned-obs-forward-model \
        --learned-obs-forward-model-path /home/mrudolph/documents/actbisim/scripts_nav2d/debug/forward_model_20/obs_forward_final.pt
        # --use-states-with-same-action
done
# debug/ms_gamma_0.99_large_only_k_step_2 \

# --single-step-resume-path /home/mrudolph/documents/actbisim/scripts_nav2d/ablations/single_step_reg/single_step_reg_0.0001/single_step_final.pt \
# datasets/nav2d_dataset_s0_e0.5_size1000000_small_grid_2_10.hdf5 \
# --single-step-resume-path /home/mrudolph/documents/actbisim/scripts_nav2d/ablations/single_step_reg/single_step_reg_0.0001/single_step_final.pt \

# python scripts_nav2d/offline.py --logdir "/data/calebc/bisim_data/datasets/model" --decoder-lr 0.00005 --dataset "/data/calebc/bisim_data/datasets/nav2d_dataset_s0_e0.5_size1000000_frozen_k_steps_10.hdf5" --multi-step-gamma 0.99 --model-to-decode ms --k-steps 7 --k-steps-dyn 2 --k-step-forward-weight 1.0 --multi-step-forward-loss l1 --n-epochs 160 --ss-stop-epochs 15 --batch_size 1024 --use-gt-forward-model --train-single-step
