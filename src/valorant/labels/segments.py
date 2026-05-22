from __future__ import annotations

import cv2
import pandas as pd

from valorant.regions import crop_ui_region


def compare_ui_roi(frame, template_gray):
    roi = crop_ui_region(frame)
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.mean(cv2.absdiff(roi_gray, template_gray))[0]


def scan_ui_diff(video_path, template, interval_sec=1.0, threshold=30.0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        raise ValueError("Invalid FPS")

    step = max(1, int(round(interval_sec * fps)))
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    rows = []

    frame_idx = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        t_sec = frame_idx / fps
        diff = compare_ui_roi(frame, template_gray)
        rows.append(
            {
                "frame_idx": frame_idx,
                "t_sec": t_sec,
                "diff": diff,
                "is_round_like": diff < threshold,
            }
        )
        frame_idx += step

    cap.release()
    return pd.DataFrame(rows)


def make_flag_segments(df, flag_col="is_round_like", time_col="t_sec"):
    d = df.copy().sort_values(time_col).reset_index(drop=True)
    d["run_id"] = d[flag_col].ne(d[flag_col].shift()).cumsum()
    dt = d[time_col].diff().median()
    if pd.isna(dt):
        dt = 0.0

    segments = (
        d.groupby("run_id")
        .agg(
            value=(flag_col, "first"),
            start_sec=(time_col, "first"),
            end_sec=(time_col, "last"),
            n_samples=(time_col, "size"),
            mean_diff=("diff", "mean"),
            min_diff=("diff", "min"),
            max_diff=("diff", "max"),
        )
        .reset_index(drop=True)
    )
    segments["duration_sec"] = segments["end_sec"] - segments["start_sec"] + dt
    return d, segments


def build_round_candidates(segments, min_true_duration_sec=1.0, sample_offset_sec=0.0):
    candidates = segments[
        (segments["value"]) & (segments["duration_sec"] >= min_true_duration_sec)
    ].copy()
    candidates["t_sec"] = candidates["start_sec"] + sample_offset_sec
    candidates["t_min"] = candidates["t_sec"] / 60.0
    return candidates.reset_index(drop=True)
