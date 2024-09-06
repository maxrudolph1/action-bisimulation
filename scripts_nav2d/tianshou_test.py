import gymnasium as gym
import tianshou as ts
import torch, numpy as np
from torch import nn
from nav2d_representation import nets
import envpool
import nav2d

class CNNPolicy(nn.Module):
    def __init__(self, state_shape, action_dim=4, atoms=1):
        super().__init__()
        self.encoder = nets.Encoder(state_shape)
        self.dqn = nets.DQN(self.encoder.output_dim, 4 * atoms)
        self.atoms = atoms
        self.action_dim = action_dim
        
    def forward(self, obs, state=None, info={}):
        obs = torch.as_tensor(obs, device="cuda:0", dtype=torch.float32)
        
        embed = self.encoder(obs)
        if self.atoms == 1:
            logits = self.dqn(embed).view(-1,self.action_dim).cpu()
        else:
            logits = self.dqn(embed).view(-1,self.action_dim, self.atoms).cpu()
        return logits, state
class Net(nn.Module):
    def __init__(self, state_shape, action_shape, atoms=1):
        super().__init__()
        self.atoms = atoms
        self.action_shape = action_shape
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape * self.atoms)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        
        logits = self.model(obs.view(batch, -1))
        
        if self.atoms > 1:
            logits = logits.view(-1,self.action_shape, self.atoms)
        return logits, state
    
env_lambda = lambda: gym.make("Nav2D-v0", grid_size=20, num_obstacles=1)
# env_lambda = lambda: gym.make("CartPole-v0")

env = env_lambda()

train_envs = ts.env.DummyVectorEnv([env_lambda for _ in range(10)])
test_envs = ts.env.DummyVectorEnv([env_lambda  for _ in range(100)])

atoms = 51
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
# net = Net(state_shape, action_shape, atoms=1)
net = CNNPolicy(state_shape, action_shape, atoms=1).cuda()
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

# policy = ts.policy.RainbowPolicy(net, optim, discount_factor=0.9, estimation_step=3, num_atoms=atoms, target_update_freq=320)
policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)
train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)


result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector,
    max_epoch=100, step_per_epoch=10000, step_per_collect=10,
    update_per_step=0.1, episode_per_test=100, batch_size=64,
    train_fn=lambda epoch, env_step: policy.set_eps(0.1),
    test_fn=lambda epoch, env_step: policy.set_eps(0.05))
    # stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold)
print(f'Finished training! Use {result["duration"]}')