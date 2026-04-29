from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def crop_ui_region(frame):
    h, w = frame.shape[:2]
    x1 = int(w * 0.35)
    x2 = int(w * 0.65)
    y1 = int(h * 0.0)
    y2 = int(h * 0.06)
    return frame[y1:y2, x1:x2]


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


def crop_score_banner(frame):
    h, w = frame.shape[:2]
    y1 = int(h * 0.005)
    y2 = int(h * 0.06)
    x1 = int(w * 0.31)
    x2 = int(w * 0.69)
    return frame[y1:y2, x1:x2]


def crop_left_right_scores(banner):
    h, w = banner.shape[:2]
    left_crop = banner[int(h * 0.12) : int(h * 0.88), int(w * 0.25) : int(w * 0.38)]
    right_crop = banner[int(h * 0.12) : int(h * 0.88), int(w * 0.62) : int(w * 0.75)]
    return left_crop, right_crop


def preprocess_score_crop(crop, scale=4):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    return th


def crop_round_region(frame):
    h, w = frame.shape[:2]
    x1 = int(w * 0.51)
    x2 = int(w * 0.527)
    y1 = int(h * 0.01)
    y2 = int(h * 0.025)
    return frame[y1:y2, x1:x2]


def preprocess_round_crop(crop, scale=3):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, th_fixed = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )

    return {
        "gray": gray,
        "fixed": th_fixed,
        "otsu": th_otsu,
        "adaptive": th_adaptive,
    }


def sanitize_score(x, max_score=25):
    if x is None:
        return None
    if 0 <= x <= max_score:
        return x
    return None


def ocr_score_crop(proc_img, reader, max_score=25):
    results = reader.readtext(proc_img, detail=0, allowlist="0123456789")
    text = "".join(results).strip()
    match = re.search(r"\d+", text)
    if not match:
        return None
    return sanitize_score(int(match.group()), max_score=max_score)


def read_round_number_from_crop(crop, reader, min_round=1, max_round=45):
    processed_imgs = preprocess_round_crop(crop)

    for img in processed_imgs.values():
        results = reader.readtext(img, detail=0, allowlist="0123456789")
        if not results:
            continue

        text = "".join(results).strip()
        if not text.isdigit():
            continue

        round_no = int(text)
        if min_round <= round_no <= max_round:
            return round_no

    return None


def majority_vote(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def read_score_from_frame(frame, reader):
    banner = crop_score_banner(frame)
    left_crop, right_crop = crop_left_right_scores(banner)
    left_score = ocr_score_crop(preprocess_score_crop(left_crop), reader)
    right_score = ocr_score_crop(preprocess_score_crop(right_crop), reader)
    return left_score, right_score


def read_score_at_sec(video_path, t_sec, reader):
    frame = get_frame_at_sec(video_path, t_sec)
    left_score, right_score = read_score_from_frame(frame, reader)
    return {"t_sec": t_sec, "left_score": left_score, "right_score": right_score}


def read_round_number_at_sec(video_path, t_sec, reader):
    frame = get_frame_at_sec(video_path, t_sec)
    crop = crop_round_region(frame)
    return read_round_number_from_crop(crop, reader)


def read_round_numbers_near_time(video_path, t_sec, reader, offsets):
    results = []

    for offset in offsets:
        sample_t_sec = t_sec + offset
        try:
            round_no = read_round_number_at_sec(video_path, sample_t_sec, reader)
        except Exception as exc:
            round_no = None
            error = str(exc)
        else:
            error = None

        result = {
            "t_sec": sample_t_sec,
            "offset": offset,
            "round_no": round_no,
        }
        if error is not None:
            result["error"] = error
        results.append(result)

    return results


def majority_round_number(samples):
    return majority_vote([sample["round_no"] for sample in samples])


def scan_round_ocr_between(video_path, start_sec, end_sec, reader, step=3):
    rows = []
    if pd.isna(start_sec) or pd.isna(end_sec) or end_sec < start_sec:
        return pd.DataFrame(columns=["t_sec", "round_no_ocr"])

    t_sec = start_sec
    while t_sec <= end_sec:
        try:
            round_no = read_round_number_at_sec(video_path, t_sec, reader)
        except Exception as exc:
            round_no = None
            error = str(exc)
        else:
            error = None

        row = {"t_sec": t_sec, "round_no_ocr": round_no}
        if error is not None:
            row["error"] = error
        rows.append(row)
        t_sec += step

    return pd.DataFrame(rows)


def read_score_multi_sec(video_path, t_list, reader):
    raw_results = []
    left_scores = []
    right_scores = []

    for t_sec in t_list:
        try:
            result = read_score_at_sec(video_path, t_sec, reader)
        except Exception as exc:
            result = {
                "t_sec": t_sec,
                "error": str(exc),
                "left_score": None,
                "right_score": None,
            }
        raw_results.append(result)
        left_scores.append(result["left_score"])
        right_scores.append(result["right_score"])

    return {
        "left_score": majority_vote(left_scores),
        "right_score": majority_vote(right_scores),
        "left_candidates": left_scores,
        "right_candidates": right_scores,
        "raw_results": raw_results,
    }


def add_scores_to_candidates(video_path, candidates, reader, offsets=(0, 5, 10)):
    out = candidates.copy()
    left_scores = []
    right_scores = []
    left_candidates = []
    right_candidates = []

    for t_sec in out["t_sec"]:
        res = read_score_multi_sec(video_path, [t_sec + off for off in offsets], reader)
        left_scores.append(res["left_score"])
        right_scores.append(res["right_score"])
        left_candidates.append(res["left_candidates"])
        right_candidates.append(res["right_candidates"])

    out["left_score"] = left_scores
    out["right_score"] = right_scores
    out["left_score_candidates"] = left_candidates
    out["right_score_candidates"] = right_candidates
    return out


def add_map_and_round_numbers(df):
    out = df.copy()
    out["score_total"] = out["left_score"] + out["right_score"]
    out["round_no_from_score"] = out["score_total"] + 1

    map_no = []
    current_map = 1
    prev_left = None
    prev_right = None

    for _, row in out.iterrows():
        left = row["left_score"]
        right = row["right_score"]
        if (
            prev_left is not None
            and prev_right is not None
            and pd.notna(left)
            and pd.notna(right)
            and (left < prev_left or right < prev_right)
        ):
            current_map += 1

        map_no.append(current_map)
        if pd.notna(left) and pd.notna(right):
            prev_left = left
            prev_right = right

    out["map_no"] = map_no
    out["round_no"] = out.groupby("map_no").cumcount() + 1
    out["round_no_from_score"] = out["round_no_from_score"].astype("Int64")
    return out


def add_left_win_label(df):
    out = df.copy()
    out["_next_map"] = out["map_no"].shift(-1)
    out["_next_left"] = out["left_score"].shift(-1)
    out["_next_right"] = out["right_score"].shift(-1)
    out["_map_changed"] = out["map_no"] != out["_next_map"]

    def judge_row(row):
        next_left = row["_next_left"]
        next_right = row["_next_right"]
        if pd.isna(next_left) or pd.isna(next_right):
            return np.nan

        if not row["_map_changed"]:
            if next_left > row["left_score"] and next_right == row["right_score"]:
                return 1
            if next_right > row["right_score"] and next_left == row["left_score"]:
                return 0
            return np.nan

        if next_left > next_right:
            return 1
        if next_left < next_right:
            return 0
        return np.nan

    out["left_win_label"] = out.apply(judge_row, axis=1)
    return out.drop(columns=["_next_map", "_next_left", "_next_right", "_map_changed"])


def add_round_ocr_samples(
    video_path,
    df,
    reader,
    start_offsets=(1, 3, 5),
    end_offsets=(-5, -3, -1),
):
    out = df.copy()
    if "start_sec" not in out.columns:
        out["start_sec"] = out["t_sec"]
    if "end_sec" not in out.columns:
        out["end_sec"] = out["t_sec"]

    out["round_ocr_start_samples"] = out["start_sec"].apply(
        lambda t: read_round_numbers_near_time(video_path, t, reader, start_offsets)
    )
    out["round_ocr_end_samples"] = out["end_sec"].apply(
        lambda t: read_round_numbers_near_time(video_path, t, reader, end_offsets)
    )
    out["round_no_ocr_start"] = out["round_ocr_start_samples"].apply(majority_round_number)
    out["round_no_ocr_end"] = out["round_ocr_end_samples"].apply(majority_round_number)
    return out


def classify_sequence_status(row):
    if pd.isna(row["round_no_ocr_start"]):
        return "ocr_failed"
    if row["_row_idx"] == 0:
        return "first"
    if pd.isna(row["prev_round_no_ocr"]):
        return "map_change_or_ocr_error"
    if pd.isna(row["ocr_step"]):
        return "map_change_or_ocr_error"
    if row["ocr_step"] == 1:
        return "ok"
    if row["ocr_step"] == 0 and row["gap_from_prev"] < 30:
        return "merge_same_round"
    if row["ocr_step"] == 0 and row["gap_from_prev"] >= 30:
        return "drop_previous_candidate"
    if row["ocr_step"] > 1:
        return "missing_round_between"
    if row["ocr_step"] < 0:
        return "map_change_or_ocr_error"
    return "map_change_or_ocr_error"


def sequence_status_to_action(status):
    if status in {"ok", "first"}:
        return "keep"
    if status == "merge_same_round":
        return "merge_with_previous"
    if status == "drop_previous_candidate":
        return "drop_previous"
    if status == "missing_round_between":
        return "inspect_between"
    if status in {"map_change_or_ocr_error", "ocr_failed"}:
        return "manual_check"
    return "manual_check"


def add_round_sequence_check(df):
    out = df.copy()
    out["_row_idx"] = np.arange(len(out))
    out["prev_round_no_ocr"] = out["round_no_ocr_start"].shift(1)
    out["prev_end_like"] = out["end_sec"].shift(1)
    out["gap_from_prev"] = out["start_sec"] - out["prev_end_like"]
    out["ocr_step"] = out["round_no_ocr_start"] - out["prev_round_no_ocr"]
    out["sequence_status"] = out.apply(classify_sequence_status, axis=1)
    out["action_candidate"] = out["sequence_status"].apply(sequence_status_to_action)
    return out.drop(columns=["_row_idx"])


def add_round_ocr_checks(
    video_path,
    df,
    reader,
    start_offsets=(1, 3, 5),
    end_offsets=(-5, -3, -1),
):
    out = add_round_ocr_samples(
        video_path,
        df,
        reader,
        start_offsets=start_offsets,
        end_offsets=end_offsets,
    )
    return add_round_sequence_check(out)


def choose_default_video(vods_dir):
    videos = sorted(Path(vods_dir).glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No .mp4 files found in {vods_dir}")
    return videos[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--vods-dir", type=Path, default=Path("data/vods"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/round_labels"))
    parser.add_argument("--template-sec", type=float, default=3000.0)
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=30.0)
    parser.add_argument("--min-true-duration-sec", type=float, default=1.0)
    parser.add_argument("--score-offsets", type=float, nargs="+", default=[0.0, 5.0, 10.0])
    parser.add_argument("--round-start-offsets", type=float, nargs="+", default=[1.0, 3.0, 5.0])
    parser.add_argument("--round-end-offsets", type=float, nargs="+", default=[-5.0, -3.0, -1.0])
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    video_path = args.video or choose_default_video(args.vods_dir)
    _, _, duration_sec = get_video_info(video_path)
    template_sec = min(max(args.template_sec, 0.0), max(0.0, duration_sec - 1.0))
    template = crop_ui_region(get_frame_at_sec(video_path, template_sec))

    print(f"video: {video_path}")
    print(f"template_sec: {template_sec:.2f}")
    print("scan_ui_diff...")
    scan_df = scan_ui_diff(
        video_path,
        template,
        interval_sec=args.interval_sec,
        threshold=args.threshold,
    )
    _, segments = make_flag_segments(scan_df)
    candidates = build_round_candidates(
        segments,
        min_true_duration_sec=args.min_true_duration_sec,
    )

    print(f"round-like candidates: {len(candidates)}")
    import easyocr

    reader = easyocr.Reader(["en"], gpu=args.gpu)
    labeled = add_scores_to_candidates(
        video_path,
        candidates,
        reader,
        offsets=tuple(args.score_offsets),
    )
    labeled = add_map_and_round_numbers(labeled)
    labeled = add_left_win_label(labeled)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    scan_path = args.out_dir / f"{stem}_ui_diff.csv"
    segments_path = args.out_dir / f"{stem}_ui_segments.csv"
    labels_path = args.out_dir / f"{stem}_round_labels.csv"
    checked_labels_path = args.out_dir / f"{stem}_round_labels_checked.csv"
    scan_df.to_csv(scan_path, index=False)
    segments.to_csv(segments_path, index=False)
    labeled.to_csv(labels_path, index=False)

    labels_for_check = pd.read_csv(labels_path)
    checked = add_round_ocr_checks(
        video_path,
        labels_for_check,
        reader,
        start_offsets=tuple(args.round_start_offsets),
        end_offsets=tuple(args.round_end_offsets),
    )
    checked.to_csv(checked_labels_path, index=False)

    print(f"wrote: {scan_path}")
    print(f"wrote: {segments_path}")
    print(f"wrote: {labels_path}")
    print(f"wrote: {checked_labels_path}")
    preview_cols = [
        "map_no",
        "round_no",
        "round_no_from_score",
        "t_sec",
        "left_score",
        "right_score",
        "left_win_label",
    ]
    print(labeled[preview_cols].head(20))
    print(checked["sequence_status"].value_counts(dropna=False))

    inspect_actions = {"manual_check", "inspect_between"}
    inspect_preview_cols = [
        "map_no",
        "round_no",
        "start_sec",
        "end_sec",
        "round_no_ocr_start",
        "round_no_ocr_end",
        "prev_round_no_ocr",
        "prev_end_like",
        "gap_from_prev",
        "ocr_step",
        "sequence_status",
        "action_candidate",
    ]
    available_preview_cols = [
        col for col in inspect_preview_cols if col in checked.columns
    ]
    preview = checked[checked["action_candidate"].isin(inspect_actions)]
    print(preview[available_preview_cols])


if __name__ == "__main__":
    main()
