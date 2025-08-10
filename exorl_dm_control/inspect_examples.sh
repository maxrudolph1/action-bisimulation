cd /home/ekuo/bisim/action-bisimulation/exorl_dm_control/

python inspect_pointmaze_examples.py \
  --dataset '/home/ekuo/bisim/exorl/datasets/point_mass_maze/rnd/all_eps_with_eplen.hdf5' \
  --obs-buffer-size 4 \
  --max-transitions 1500000 \
  --n-per-action 3 \
  # --encoder-checkpoint '/home/ekuo/bisim/action-bisimulation/results/pointmaze_testing_sandbox_2025-08-10_08-23-44_ts_5860/single_step.pt' \ # with softmax still
  --encoder-checkpoint '/home/ekuo/bisim/action-bisimulation/results/pointmaze_testing_sandbox_2025-08-10_23-43-18_ts_5860/single_step.pt' \ # without softmax, but still l2
  --wandb-entity 'evan-kuo-edu' \
  --wandb-project nav2d \
  --seed 0
