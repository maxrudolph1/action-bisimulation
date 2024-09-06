from nav2d_representation.nav2d.nav2d_po import Navigate2DPO

env = Navigate2DPO(num_obstacles=10, static_goal=True)

obs = env.reset()
for _ in range(10):
    print(env.reset() + 1)
