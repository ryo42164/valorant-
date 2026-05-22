from __future__ import annotations

import numpy as np
import pandas as pd

from valorant.ocr.round import majority_round_number, read_round_numbers_near_time


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
