
cd /home/mrudolph/documents/actbisim/scripts_nav2d
python offline.py --dataset datasets/random_goal/nav2d_dataset_s0_e0.5_size1000000_default_2.hdf5 \
--logdir rep_models/6-5-2023/test_kstep2 \
--k-steps 2 \
--single-step-resume-path /home/mrudolph/documents/actbisim/scripts_nav2d/rep_models/single_step_model4/single_step_final.pt \
--ss-warmup-steps 0

