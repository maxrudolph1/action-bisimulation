import gymnasium as gym
import nav2d
env = gym.make("Nav2D-v0", obstacle_diameter=1)

obs, rew = env.reset()

obs[obs < 0] = 0
print(obs[0,:,:] + 2 * obs[2,:,:])

