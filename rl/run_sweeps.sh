#!/bin/bash

# sweep hyperparams
grid_sizes=(10 15 20 25 30 35 40)
num_obstacles=(0 5 10 15 20 25 30)
obstacle_diameters=(1 2 3 4)

#test
# grid_sizes=(10 15)
# num_obstacles=(0 5)
# obstacle_diameters=(1 2)

for grid_size in "${grid_sizes[@]}"; do
  for num_obstacle in "${num_obstacles[@]}"; do
    for obstacle_diameter in "${obstacle_diameters[@]}"; do
      echo "Running with grid_size=${grid_size}, num_obstacles=${num_obstacle}, obstacle_diameter=${obstacle_diameter}"
      
      python /nfs/homes/bisim/rrai/action-bisimulation/rl/main.py \
        hydra.run.dir=. \
        env.grid_size=${grid_size} \
        env.num_obstacles=${num_obstacle} \
        env.obstacle_diameter=${obstacle_diameter}
      
      echo "Finished run with grid_size=${grid_size}, num_obstacles=${num_obstacle}, obstacle_diameter=${obstacle_diameter}"
      echo "----------------------------------------------"
    done
  done
done
