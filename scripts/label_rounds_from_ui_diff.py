from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from valorant.labels.segments import (
    build_round_candidates,
    compare_ui_roi,
    make_flag_segments,
    scan_ui_diff,
)
from valorant.labels.sequence import (
    add_left_win_label,
    add_map_and_round_numbers,
    add_round_ocr_checks,
    add_round_ocr_samples,
    add_round_sequence_check,
    classify_sequence_status,
    sequence_status_to_action,
)
from valorant.ocr.preprocess import preprocess_round_crop, preprocess_score_crop
from valorant.ocr.round import (
    majority_round_number,
    read_round_number_at_sec,
    read_round_number_from_crop,
    read_round_numbers_near_time,
    scan_round_ocr_between,
)
from valorant.ocr.score import (
    add_scores_to_candidates,
    ocr_score_crop,
    read_score_at_sec,
    read_score_from_frame,
    read_score_multi_sec,
    sanitize_score,
)
from valorant.regions import (
    crop_left_right_scores,
    crop_round_region,
    crop_score_banner,
    crop_ui_region,
)
from valorant.utils import majority_vote_raw as majority_vote
from valorant.video import get_frame_at_sec, get_video_info


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
