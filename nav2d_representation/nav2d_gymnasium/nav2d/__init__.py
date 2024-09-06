from gymnasium.envs.registration import register

register(
     id="Nav2D-v0",
     entry_point="nav2d.envs.nav2d:Navigate2D",
     max_episode_steps=300,
     kwargs={"num_obstacles": 15},
)