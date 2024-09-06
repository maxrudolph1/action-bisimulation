cd ../

python offline.py --logdir "debug/single-step-l1-0.0001" \
    --decoder-lr 0.00005 \
    --dataset datasets/nav2d_dataset_s0_e0.5_size1000000__k_steps_10.hdf5 \
    --train-decode-separately \
    --train-single-step \
    --model-to-decode ss \
    --n-epochs 160 \
    --ss-stop-epochs 160 \
    --batch_size 1024 \
    --no-train-multi-step \
    --single-step-l1-penalty 0.0001
