import os, sys, glob
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm

# make sure your ExoRL clone is on the path
sys.path.insert(0, os.path.expanduser('~/bisim/exorl'))
import dmc


# -----------------------------------------------------------------------------
# Worker function: processes one chunk of episodes, dumping parts as we go
# -----------------------------------------------------------------------------
def process_chunk(args):
    task_name, file_list, out_dir, chunk_id = args

    try:
        # each process builds its own env once
        env = dmc.make(
            task_name,
            obs_type='pixels',
            frame_stack=1,
            action_repeat=1,
            seed=chunk_id  # different seed per process
        )

        os.makedirs(out_dir, exist_ok=True)

        for ep in tqdm(file_list, desc=f"Worker {chunk_id}", position=chunk_id, leave=False):
            data = np.load(ep)
            phys = data['physics']
            images = []
            for state in phys:
                with env.physics.reset_context():
                    env.physics.set_state(state)
                frame = env.physics.render(width=64, height=64, camera_id=0)
                images.append(frame)

            images = np.stack(images)
            ep_id = os.path.basename(ep).split('.')[0]

            np.savez_compressed(
                f"{out_dir}/{ep_id}.npz",
                images=images,
                physics=phys,
            )
        return len(file_list)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 0


# -----------------------------------------------------------------------------
# Main: split files, launch Pool, collect stats, write metadata + samples
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    task = 'point_mass_maze_reach_top_left'
    buffer_dir = os.path.expanduser('~/bisim/exorl/datasets/point_mass_maze/rnd/buffer')
    out_dir    = os.path.expanduser('~/bisim/exorl/datasets/point_mass_maze/rnd/processed_true')

    # gather all episodes
    all_eps = sorted(glob.glob(f"{buffer_dir}/*.npz"))
    N = min(max(cpu_count()-2, 1), len(all_eps))

    # split into N roughly-equal chunks
    chunks = [all_eps[i::N] for i in range(N)]
    args = [(task, chunks[i], out_dir, i) for i in range(N)]

    # launch
    with Pool(N) as pool:
        counts = list(tqdm(
            pool.imap_unordered(process_chunk, args),
            total=len(args),
            desc="Chunks done"
        ))

    total_frames = sum(counts)
    print(f"Rendered a total of {total_frames} frames across {N} workers.")
    print("Done. Final images in ", out_dir)
