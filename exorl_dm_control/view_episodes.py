import os, sys
import numpy as np
import imageio

import pdb

# ensure ExoRL is on your path
sys.path.insert(0, os.path.expanduser('~/bisim/exorl'))
import dmc

def make_gif_from_episode(ep_path, gif_path, fps=20):
    data = np.load(ep_path)
    pdb.set_trace()
    actions = data['action']    # shape (T,2)
    
    # build the exact pixels env
    env = dmc.make(
        'point_mass_maze_reach_top_left',
        obs_type='pixels',      # pixel observations
        frame_stack=1,
        action_repeat=1,
        seed=0
    )

    # reset and grab the initial frame
    ts = env.reset()
    # obs is a dict with key 'pixels'
    frames = [ ts.observation['pixels'][..., :3] ]  

    # step through every recorded action
    for a in actions:
        ts = env.step(a)
        frames.append(ts.observation['pixels'][..., :3])

    # write out a gif
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"Wrote {gif_path}")

if __name__ == "__main__":
    input_dir = os.path.expanduser('~/bisim/exorl/datasets/point_mass_maze/rnd/processed_true')
    output_dir = os.path.expanduser('~/bisim/exorl/datasets/point_mass_maze/rnd/episode_gifs')
    os.makedirs(output_dir, exist_ok=True)

    # pick first 3 episodes
    eps = sorted(f for f in os.listdir(input_dir) if f.endswith('.npz'))[:3]
    for idx, fn in enumerate(eps):
        ep_path  = os.path.join(input_dir, fn)
        gif_path = os.path.join(output_dir, f'episode_{idx:03d}.gif')
        make_gif_from_episode(ep_path, gif_path)
