#!/bin/bash

cd rl

# Define the different configurations
grid_sizes=(10 15 20 25 30 35 40 45 50 55 60)
num_obstacles=(5 10 15)
obstacle_diameters=(1 2 3)

# Loop through each combination of configurations
for grid_size in "${grid_sizes[@]}"; do
  for num_obstacle in "${num_obstacles[@]}"; do
    for obstacle_diameter in "${obstacle_diameters[@]}"; do
      python main.py env.grid_size=$grid_size env.num_obstacles=$num_obstacle env.obstacle_diameter=$obstacle_diameter
    done
  done
done