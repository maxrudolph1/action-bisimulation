cd /home/ekuo/bisim/action-bisimulation/exorl_dm_control/

# python build_balanced_subset.py \
#   --dataset /home/ekuo/bisim/exorl/datasets/point_mass_maze/rnd/all_eps_with_eplen.hdf5 \
#   --obs-buffer-size 4 \
#   --num-actions 9 \
#   --seed 0 \
#   --out /home/ekuo/bisim/exorl/datasets/point_mass_maze/rnd/balanced_valid_t_all_eps_eplen.npy


# python build_balanced_subset.py \
#   --dataset /home/ekuo/bisim/exorl/datasets/point_mass_maze/rnd/all_eps_with_eplen.hdf5 \
#   --obs-buffer-size 4 \
#   --num-actions 9 \
#   --seed 0 \
#   --out /home/ekuo/bisim/exorl/datasets/point_mass_maze/rnd/balanced_uniq_stall.npy


# python build_balanced_subset.py \
#   --dataset /home/ekuo/bisim/exorl/datasets/point_mass_maze/rnd/all_eps_with_eplen.hdf5 \
#   --obs-buffer-size 4 \
#   --num-actions 9 \
#   --scan-quantiles

python build_balanced_subset.py \
  --dataset /home/ekuo/bisim/exorl/datasets/point_mass_maze/rnd/all_eps_with_eplen.hdf5 \
  --obs-buffer-size 4 \
  --num-actions 9 \
  --stall-eps-noop 0.00153569 \
  --stall-eps-move 0.001612 \
  --seed 0 \
  --out /home/ekuo/bisim/exorl/datasets/point_mass_maze/rnd/balanced_uniq_stall.npy
