cd ../
gamma=10.0
python offline.py --logdir "debug/ms_gamma_${gamma}_k_step_forward_1.0_normalized_gamma_l1_forward_decode" \
    --decoder-lr 0.00005 \
    --dataset datasets/nav2d_dataset_s0_e0.5_size1000000_reconstruction_data_2_10.hdf5 \
    --multi-step-gamma "${gamma}" \
    --model-to-decode ms \
    --multi-step-resume-path "/home/mrudolph/documents/actbisim/scripts_nav2d/debug/ms_gamma_${gamma}_k_step_forward_1.0_normalized_gamma_l1_forward/multi_step_final.pt" 