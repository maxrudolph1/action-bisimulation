import cv2
import numpy as np

def create_video(frames, output_path, fps):
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"HFYU"), fps, frames[0].shape[1:])
    for frame in frames:
        video.write(
            cv2.cvtColor(
                frame.transpose(1, 2, 0),
                cv2.COLOR_RGB2BGR,
            )
        )
    video.release()