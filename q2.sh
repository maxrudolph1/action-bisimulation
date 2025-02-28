#!/bin/bash

cd rl

# 20x20, obst 20 to 40 (6)
# python main.py env.grid_size=20 env.num_obstacles=20 env.obstacle_diameter=1
# python main.py env.grid_size=20 env.num_obstacles=24 env.obstacle_diameter=1
# python main.py env.grid_size=20 env.num_obstacles=28 env.obstacle_diameter=1
# python main.py env.grid_size=20 env.num_obstacles=32 env.obstacle_diameter=1
# python main.py env.grid_size=20 env.num_obstacles=36 env.obstacle_diameter=1
# python main.py env.grid_size=20 env.num_obstacles=40 env.obstacle_diameter=1

# # 30x30, obst 15 to 40 (6)
# python main.py env.grid_size=30 env.num_obstacles=15 env.obstacle_diameter=1
# python main.py env.grid_size=30 env.num_obstacles=20 env.obstacle_diameter=1
# python main.py env.grid_size=30 env.num_obstacles=25 env.obstacle_diameter=1
# python main.py env.grid_size=30 env.num_obstacles=30 env.obstacle_diameter=1
# python main.py env.grid_size=30 env.num_obstacles=35 env.obstacle_diameter=1
# python main.py env.grid_size=30 env.num_obstacles=40 env.obstacle_diameter=1

# # 40x40, obst 30 to 120 (10)
python main.py env.grid_size=40 env.num_obstacles=30 env.obstacle_diameter=1
python main.py env.grid_size=40 env.num_obstacles=40 env.obstacle_diameter=1
python main.py env.grid_size=40 env.num_obstacles=50 env.obstacle_diameter=1
python main.py env.grid_size=40 env.num_obstacles=60 env.obstacle_diameter=1
python main.py env.grid_size=40 env.num_obstacles=70 env.obstacle_diameter=1
python main.py env.grid_size=40 env.num_obstacles=80 env.obstacle_diameter=1
python main.py env.grid_size=40 env.num_obstacles=90 env.obstacle_diameter=1
python main.py env.grid_size=40 env.num_obstacles=100 env.obstacle_diameter=1
python main.py env.grid_size=40 env.num_obstacles=110 env.obstacle_diameter=1
python main.py env.grid_size=40 env.num_obstacles=120 env.obstacle_diameter=1

# #50x50, obst 50 to 130 (9)
# python main.py env.grid_size=50 env.num_obstacles=50 env.obstacle_diameter=1
# python main.py env.grid_size=50 env.num_obstacles=60 env.obstacle_diameter=1
# python main.py env.grid_size=50 env.num_obstacles=70 env.obstacle_diameter=1
# python main.py env.grid_size=50 env.num_obstacles=80 env.obstacle_diameter=1
# python main.py env.grid_size=50 env.num_obstacles=90 env.obstacle_diameter=1
# python main.py env.grid_size=50 env.num_obstacles=100 env.obstacle_diameter=1
# python main.py env.grid_size=50 env.num_obstacles=110 env.obstacle_diameter=1
# python main.py env.grid_size=50 env.num_obstacles=120 env.obstacle_diameter=1
# python main.py env.grid_size=50 env.num_obstacles=130 env.obstacle_diameter=1
