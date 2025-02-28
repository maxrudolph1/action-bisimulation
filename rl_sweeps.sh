#!/bin/bash

cd rl 

# 10 20 30 40 50
for grid_size in $(seq 10 10 50); do
  
  
  min_obstacles=$((grid_size * grid_size * 1 / 5 / 4))
  max_obstacles=$((grid_size * grid_size * 3 / 5 / 4))

  # pick up to 10 highest num of obstacels
  obstacle_numbers=($(seq $min_obstacles 10 $max_obstacles))
  highest_obstacles=("${obstacle_numbers[@]: -10}")

  for num_obstacles in "${highest_obstacles[@]}"; do
    python main.py env.grid_size=$grid_size env.num_obstacles=$num_obstacles env.obstacle_diameter=1
  done
done


