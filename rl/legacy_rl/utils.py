from stable_baselines3.common.callbacks import EvalCallback
import wandb
import time
import numpy as np
import torch
class WandbEvalCallback(EvalCallback):
    def __init__(self, eval_env, render_freq=1000, **kwargs):
        # Separate EvalCallback specific arguments and WandbEvalCallback specific arguments
        evalcallback_args = {}
        wandb_args = {}
        self.render_freq = render_freq  
        evalcallback_params = EvalCallback.__init__.__code__.co_varnames
        for key, value in kwargs.items():
            if key in evalcallback_params:
                evalcallback_args[key] = value
            else:
                wandb_args[key] = value

        super(WandbEvalCallback, self).__init__(eval_env, **evalcallback_args)
        self.best_mean_reward = -float('inf')
        self.start_time = time.time()

    def _on_step(self) -> bool:
        result = super(WandbEvalCallback, self)._on_step()
        # print(self.num_timesteps, self.eval_freq, self.num_timesteps % self.eval_freq)
        if self.n_calls % self.eval_freq == 0:
            # Get the current mean reward
            obs = self.eval_env.reset()
            done = False
            # while not done:
            #     action, _ = self.model.predict(obs)
            #     obs, rew, done ,info = self.eval_env.step(action)
            #     done = done[0]
            #     print(action)
            mean_reward = self.last_mean_reward
            log_dict = {"eval/mean_reward": mean_reward, "n_steps": self.num_timesteps, "fps": self.num_timesteps/(time.time() - self.start_time)}
            wandb.log(log_dict, step=self.num_timesteps)
            
            # # If the current mean reward is the best we have seen, log it as the best mean reward
            # if mean_reward > self.best_mean_reward:
            #     self.best_mean_reward = mean_reward
            #     wandb.log({"eval/best_mean_reward": self.best_mean_reward}, step=self.num_timesteps)

        if self.n_calls % self.render_freq == 0:
            images = []
            obs = self.eval_env.reset()
            done = False
            rews = []
            infos = []
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                # action = [self.model.q_net(torch.tensor(obs)).argmax().item()]
                obs, rew, done, info = self.eval_env.step(action)
                done = done[0]
                img = self.eval_env.render(mode='rgb_array')[:480, :480, :]
                rews.append(rew)
                images.append(img)
                infos.append(info)

            rews = np.array(rews)
            images = np.array(images).transpose(0, 3, 1, 2)[:100] * 255
            images[:, :, 0,0] = (rews[:100] + 1)* 255
            wandb.log({"eval/video": wandb.Video(images, fps=5, format="gif")}, step=self.num_timesteps)
            B, C, H, W = obs.shape
            values = np.zeros(( H, W))
            for i in range(7):
                for j in range(7):
                    obs[0,2,:,:] = 0
                    obs[0,2,i,j] = 255
                    value = self.model.q_net(torch.tensor(obs)).max().item()
                    values[i,j] = value
            disp_values = ((values - values.min()) / (values.max() - values.min()) * 255).astype(np.uint8)
            wandb.log({"eval/values": wandb.Image(disp_values)}, step=self.num_timesteps)
        return result