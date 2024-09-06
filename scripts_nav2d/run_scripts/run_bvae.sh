cd ../

# Define an array of gamma values
gamma_values=(0.99)

# Loop through each gamma value and run the Python script
for gamma in "${gamma_values[@]}"; do
    python offline.py --logdir "debug/bvae" \
        --decoder-lr 0.00005 \
        --dataset datasets/nav2d_dataset_s0_e0.5_size1000000__k_steps_10.hdf5 \
        --k-steps 7 \
        --k-steps-dyn 2 \
        --k-step-forward-weight 1.0 \
        --multi-step-forward-loss l1 \
	    --n-epochs 160 \
        --ss-stop-epochs 15 \
        --batch_size 1024 \
        --use-gt-forward-model \
        --train-bvae 
done
# debug/ms_gamma_0.99_large_only_k_step_2 \

# --single-step-resume-path /home/mrudolph/documents/actbisim/scripts_nav2d/ablations/single_step_reg/single_step_reg_0.0001/single_step_final.pt \
# datasets/nav2d_dataset_s0_e0.5_size1000000_small_grid_2_10.hdf5 \
# --single-step-resume-path /home/mrudolph/documents/actbisim/scripts_nav2d/ablations/single_step_reg/single_step_reg_0.0001/single_step_final.pt \
