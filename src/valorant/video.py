from __future__ import annotations

import cv2
import numpy as np

def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0:
        raise ValueError(f"Invalid FPS: {video_path}")
    return fps, total_frames, total_frames / fps


def get_frame_at_sec(video_path, t_sec):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = int(round(t_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise ValueError(f"Failed to read frame at {t_sec:.2f}s")
    return frame

def get_frames(video_path,start_sec,num_frames=8,fps=4):
    """
    video_pathの動画をstart_secからnum_frames枚fps間隔で読む
    return:
        frames:np.ndarray,shape[T,H,W,3],RGB
    """

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    for i in range(num_frames):
        t_sec = start_sec + i / fps
        frame_idx = int(round(t_sec * video_fps))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            cap.release()
            raise ValueError(
                f"Failed to read frame: video={video_path}, "
                f"t_sec={t_sec:.3f}, frame_idx={frame_idx}"
            )

        # OpenCVはBGRなのでRGBに変換
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    return np.stack(frames, axis=0)