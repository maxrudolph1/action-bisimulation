cd /home/ekuo/bisim/action-bisimulation/exorl_dm_control/

# with softmax still
# python inspect_pointmaze_examples.py
#   --dataset '/home/ekuo/bisim/exorl/datasets/point_mass_maze/rnd/all_eps_with_eplen.hdf5' \
#   --obs-buffer-size 4 \
#   --max-transitions 1500000 \
#   --n-per-action 3 \
#   --encoder-checkpoint '/home/ekuo/bisim/action-bisimulation/results/pointmaze_testing_sandbox_2025-08-10_08-23-44_ts_5860/single_step.pt' \
#   --wandb-entity 'evan-kuo-edu' \
#   --wandb-project nav2d \
#   --seed 0

# without softmax, but still l2
# python inspect_pointmaze_examples.py \
#   --dataset '/home/ekuo/bisim/exorl/datasets/point_mass_maze/rnd/all_eps_with_eplen.hdf5' \
#   --obs-buffer-size 4 \
#   --max-transitions 1500000 \
#   --n-per-action 3 \
#   --encoder-checkpoint '/home/ekuo/bisim/action-bisimulation/results/pointmaze_testing_sandbox_2025-08-10_23-43-18_ts_5860/single_step.pt' \
#   --wandb-entity 'evan-kuo-edu' \
#   --wandb-project nav2d \
#   --seed 0


# python inspect_pointmaze_examples.py \
#   --dataset '/home/ekuo/bisim/exorl/datasets/point_mass_maze/rnd/all_eps_with_eplen.hdf5' \
#   --obs-buffer-size 4 \
#   --max-transitions 1500000 \
#   --n-per-action 3 \
#   --encoder-checkpoint '/home/ekuo/bisim/action-bisimulation/results/pointmaze_testing_sandbox_2025-08-11_02-57-11_ts_5860/single_step.pt' \
#   --wandb-entity 'evan-kuo-edu' \
#   --wandb-project nav2d \
#   --seed 0

# python inspect_pointmaze_examples.py \
#   --dataset '/home/ekuo/bisim/exorl/datasets/point_mass_maze/rnd/all_eps_with_eplen.hdf5' \
#   --obs-buffer-size 4 \
#   --max-transitions 1500000 \
#   --n-per-action 3 \
#   --encoder-checkpoint '/home/ekuo/bisim/action-bisimulation/results/pointmaze_testing_sandbox_2025-08-11_03-28-43_ts_5860/single_step.pt' \
#   --wandb-entity 'evan-kuo-edu' \
#   --wandb-project nav2d \
#   --seed 0


# pointmaze_testing_sandbox_2025-08-11_17-53-48
# first run after balancing with epochs = 3
# python inspect_pointmaze_examples.py \
#   --dataset '/home/ekuo/bisim/exorl/datasets/point_mass_maze/rnd/all_eps_with_eplen.hdf5' \
#   --obs-buffer-size 4 \
#   --max-transitions 1500000 \
#   --n-per-action 3 \
#   --encoder-checkpoint '/home/ekuo/bisim/action-bisimulation/results/pointmaze_testing_sandbox_2025-08-11_17-53-48_ts_5820/single_step.pt' \
#   --wandb-entity 'evan-kuo-edu' \
#   --wandb-project nav2d \
#   --seed 0




python inspect_pointmaze_examples.py \
  --dataset '/home/ekuo/bisim/exorl/datasets/point_mass_maze/rnd/all_eps_with_eplen.hdf5' \
  --obs-buffer-size 4 \
  --max-transitions 1500000 \
  --n-per-action 3 \
  --encoder-checkpoint '/home/ekuo/bisim/action-bisimulation/results/pointmaze_ss_forwardWeight_3_2025-08-11_18-24-38_ts_17460/single_step.pt' \
  --wandb-entity 'evan-kuo-edu' \
  --wandb-project nav2d \
  --seed 0

