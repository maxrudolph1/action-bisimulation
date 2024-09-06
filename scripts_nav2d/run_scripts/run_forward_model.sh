cd ../


# Loop through each gamma value and run the Python script

python offline.py --logdir "debug/forward_model_20" \
    --decoder-lr 0.00005 \
    --dataset datasets/nav2d_dataset_s0_e0.5_size1000000__k_steps_10.hdf5 \
    --n-epochs 20 \
    --batch_size 1024 \
    --train-obs-forward-model