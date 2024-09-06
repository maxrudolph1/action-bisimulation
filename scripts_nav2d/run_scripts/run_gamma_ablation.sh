cd ../
# Define an array of gamma values
gamma_values=(0.20 0.30 0.40)

# Loop through each gamma value and run the Python script
for gamma in "${gamma_values[@]}"; do
    python offline.py --logdir "debug/ms_gamma_${gamma}" \
        --decoder-lr 0.00005 \
        --dataset datasets/nav2d_dataset_s0_e0.5_size1000000_reconstruction_data_2_10.hdf5 \
        --multi-step-gamma "${gamma}" \
        --model-to-decode ms \
        --single-step-resume-path /home/mrudolph/documents/actbisim/scripts_nav2d/debug/single_step_reg_0.0001/single_step_final.pt
done