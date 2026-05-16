from __future__ import annotations

"""
Lightweight post-processing for *_round_labels_checked.csv files.

Example:
python scripts/fix_round_labels_from_actions.py ^
  --input "data/processed/round_labels/M8 vs. EDG - VALORANT Masters Santiago - SWISS_round_labels_checked.csv" ^
  --video "data/vods/M8 vs. EDG - VALORANT Masters Santiago - SWISS.mp4" ^
  --gpu
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

MIN_SPLIT_FRAGMENT_SEC = 20.0

try:
    from label_rounds_from_ui_diff import (
        get_frame_at_sec,
        majority_vote,
        read_round_number_at_sec,
        read_score_multi_sec,
    )
except ImportError:

    def get_frame_at_sec(video_path, t_sec):
        cv2 = require_cv2()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = int(round(float(t_sec) * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise ValueError(f"Failed to read frame at {float(t_sec):.2f}s")
        return frame

    def majority_vote(values):
        vals = [v for v in values if v is not None and not pd.isna(v)]
        if not vals:
            return None
        return Counter(vals).most_common(1)[0][0]

    def read_round_number_at_sec(video_path, t_sec, reader):
        cv2 = require_cv2()
        frame = get_frame_at_sec(video_path, t_sec)
        h, w = frame.shape[:2]
        crop = frame[int(h * 0.01) : int(h * 0.025), int(w * 0.51) : int(w * 0.527)]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        results = reader.readtext(th, detail=0, allowlist="0123456789")
        text = "".join(results).strip()
        if not text.isdigit():
            return None
        round_no = int(text)
        return round_no if 1 <= round_no <= 45 else None

    def read_score_multi_sec(video_path, t_list, reader):
        raise ImportError(
            "read_score_multi_sec could not be imported from label_rounds_from_ui_diff.py"
        )


VALID_MAPS = {
    "ASCENT",
    "BIND",
    "SPLIT",
    "HAVEN",
    "PEARL",
    "LOTUS",
    "SUNSET",
    "ABYSS",
    "BREEZE",
    "ICEBOX",
    "FRACTURE",
    "CORRODE",
}

LOG_COLUMNS = [
    "input_index",
    "action_candidate",
    "sequence_status",
    "fix_applied",
    "detail",
    "prev_start_sec",
    "prev_end_sec",
    "cur_start_sec",
    "cur_end_sec",
    "dropped_index",
    "inserted_round_no",
    "inserted_start_sec",
    "inserted_end_sec",
    "split_from_index",
    "boundary_sec",
    "ocr_scan_results",
]


def require_cv2():
    import cv2

    return cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-process *_round_labels_checked.csv using action_candidate and sequence_status."
    )
    parser.add_argument("--input", required=True, help="Path to *_round_labels_checked.csv")
    parser.add_argument("--video", help="Original video path for OCR checks")
    parser.add_argument("--output", help="Path to write *_round_labels_fixed.csv")
    parser.add_argument("--log-output", help="Path to write *_round_labels_fix_log.csv")
    parser.add_argument("--gpu", action="store_true", help="Pass gpu=True to easyocr.Reader")
    parser.add_argument(
        "--round-search-step",
        type=float,
        default=5.0,
        help="OCR interval in seconds for missing_round_between scans",
    )
    parser.add_argument(
        "--apply-drop",
        action="store_true",
        default=True,
        help="Apply drop_previous by dropping the shorter candidate. Default: enabled.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write CSV files")
    return parser.parse_args()


def default_output_paths(input_path: Path) -> tuple[Path, Path]:
    name = input_path.name
    if name.endswith("_round_labels_checked.csv"):
        stem = name[: -len("_round_labels_checked.csv")]
        fixed = input_path.with_name(f"{stem}_round_labels_fixed.csv")
        log = input_path.with_name(f"{stem}_round_labels_fix_log.csv")
    else:
        fixed = input_path.with_name(f"{input_path.stem}_fixed.csv")
        log = input_path.with_name(f"{input_path.stem}_fix_log.csv")
    return fixed, log


def is_missing(value) -> bool:
    return value is None or pd.isna(value)


def none_if_missing(value):
    return None if is_missing(value) else value


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if is_missing(value):
        return None
    return str(value)


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float):
        return None if np.isnan(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if is_missing(value):
        return None
    return value


def dumps_json(value):
    return json.dumps(
        make_json_safe(value),
        ensure_ascii=False,
        default=json_default,
        allow_nan=False,
    )



def to_float(value):
    if is_missing(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def to_int(value):
    if is_missing(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def append_fix(existing, fix_name):
    if is_missing(existing) or existing == "":
        return fix_name
    parts = str(existing).split("|")
    if fix_name in parts:
        return existing
    return f"{existing}|{fix_name}"


def row_to_dict(row, input_index=None):
    data = row.to_dict()
    data["_input_index"] = input_index if input_index is not None else row.name
    if "fix_applied" not in data:
        data["fix_applied"] = ""
    if "needs_review" not in data:
        data["needs_review"] = False
    return data


def make_log(row, fix_applied, detail="", prev=None, cur=None, **extra):
    cur_row = cur if cur is not None else row
    prev_row = prev if prev is not None else {}
    log = {col: np.nan for col in LOG_COLUMNS}
    log.update(
        {
            "input_index": cur_row.get("_input_index", row.get("_input_index", np.nan)),
            "action_candidate": cur_row.get("action_candidate", np.nan),
            "sequence_status": cur_row.get("sequence_status", np.nan),
            "fix_applied": fix_applied,
            "detail": detail,
            "prev_start_sec": prev_row.get("start_sec", np.nan),
            "prev_end_sec": prev_row.get("end_sec", np.nan),
            "cur_start_sec": cur_row.get("start_sec", np.nan),
            "cur_end_sec": cur_row.get("end_sec", np.nan),
        }
    )
    log.update(extra)
    return log


def crop_top_left_banner(frame):
    h, w = frame.shape[:2]
    return frame[0 : int(h * 0.03), 0 : int(w * 0.35)]


def preprocess_map_for_ocr(img):
    cv2 = require_cv2()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    return th


def normalize_map_text(text):
    text = str(text).upper()
    text = text.replace("|", "I")
    text = text.replace(";", ":")
    text = text.replace(".", ":")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_current_map(ocr_text):
    text = normalize_map_text(ocr_text)
    match = re.search(r"CURRENT\s*:?\s*([A-Z]+)", text)
    if match:
        candidate = match.group(1)
        if candidate in VALID_MAPS:
            return candidate

    for map_name in VALID_MAPS:
        if f"CURRENT: {map_name}" in text or f"CURRENT {map_name}" in text:
            return map_name
    return None


def read_current_map_from_frame(frame, reader):
    crop = crop_top_left_banner(frame)
    proc = preprocess_map_for_ocr(crop)
    results = reader.readtext(proc, detail=0)
    ocr_text = " ".join(results)
    return {
        "current_map": extract_current_map(ocr_text),
        "ocr_text": ocr_text,
    }


def read_current_map_at_sec(video_path, t_sec, reader):
    frame = get_frame_at_sec(video_path, t_sec)
    result = read_current_map_from_frame(frame, reader)
    result["t_sec"] = t_sec
    return result


def read_map_for_round(video_path, start_sec, reader):
    results = []
    maps = []
    for offset in (1.0, 3.0, 5.0):
        t_sec = to_float(start_sec) + offset
        try:
            result = read_current_map_at_sec(video_path, t_sec, reader)
        except Exception as exc:
            result = {"t_sec": t_sec, "current_map": None, "error": str(exc)}
        results.append(result)
        maps.append(result.get("current_map"))
    return majority_vote(maps), results


def normalize_timer_text(timer_text):
    if is_missing(timer_text):
        return ""

    text = str(timer_text).strip()
    text = text.replace("O", "0").replace("o", "0")
    text = text.replace("I", "1").replace("l", "1")
    text = text.replace(".", ":").replace(",", ":").replace(";", ":")
    text = text.replace(" ", "")
    return text


def parse_timer_text_to_sec(timer_text):
    if is_missing(timer_text):
        return None

    text = normalize_timer_text(timer_text)
    match = re.search(r"^(\d{1,2}):(\d{2})$", text)
    if match:
        minute = int(match.group(1))
        second = int(match.group(2))
        if second < 60:
            return minute * 60 + second
        return None

    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 3:
        minute = int(digits[0])
        second = int(digits[1:])
    elif len(digits) == 4:
        minute = int(digits[:-2])
        second = int(digits[-2:])
    else:
        return None

    if second >= 60:
        return None
    return minute * 60 + second


def crop_timer_region(frame):
    h, w = frame.shape[:2]
    x1 = int(w * 0.48)
    x2 = int(w * 0.52)
    y1 = int(h * 0.025)
    y2 = int(h * 0.06)
    return frame[y1:y2, x1:x2]


def preprocess_timer_for_ocr(crop):
    cv2 = require_cv2()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def read_timer_from_frame(frame, reader):
    if frame is None:
        return {
            "timer_text": "",
            "timer_sec": None,
            "raw_text": [],
        }
    crop = crop_timer_region(frame)
    proc = preprocess_timer_for_ocr(crop)
    results = reader.readtext(
        proc,
        allowlist="0123456789:.;,",
        detail=0,
        paragraph=False
    )
    text = "".join(results).strip() if results else ""
    timer_sec = parse_timer_text_to_sec(text)
    return {
        "timer_text": text,
        "timer_sec": timer_sec,
        "raw_text": results,
    }

def read_timer_near_sec(video_path, t_sec, reader, offsets=(-0.4, -0.2, 0.0, 0.2, 0.4)):
    timer_secs = []
    timer_texts = []

    for off in offsets:
        tt = t_sec + off
        if tt < 0:
            continue

        frame = get_frame_at_sec(video_path, tt)
        result = read_timer_from_frame(frame, reader)

        timer_sec = result.get("timer_sec")
        timer_text = result.get("timer_text")

        timer_texts.append(timer_text)

        if timer_sec is not None and not pd.isna(timer_sec):
            timer_secs.append(float(timer_sec))

    if len(timer_secs) == 0:
        return {
            "timer_sec": None,
            "timer_texts": timer_texts,
            "timer_secs": timer_secs,
        }

    timer_sec = float(np.median(timer_secs))

    return {
        "timer_sec": timer_sec,
        "timer_texts": timer_texts,
        "timer_secs": timer_secs,
    }

def add_start_timer_offset_columns(df, video_path, reader):
    out = df.copy()

    start_timer_secs = []
    offsets = []
    search_start_secs = []
    start_timer_raws = []

    if video_path is None or reader is None:
        out["start_timer_sec"] = np.nan
        out["start_search_offset_sec"] = np.nan
        out["search_start_sec"] = np.nan
        out["start_timer_raws"] = None
        return out

    for _, row in out.iterrows():
        start_sec = to_float(row.get("start_sec"))

        if pd.isna(start_sec):
            start_timer_secs.append(np.nan)
            offsets.append(np.nan)
            search_start_secs.append(np.nan)
            start_timer_raws.append([])
            continue

        try:
            timer_result = read_timer_near_sec(video_path, start_sec, reader)
            timer_sec = timer_result.get("timer_sec")
            raw_timer_secs = timer_result.get("timer_secs", [])
        except Exception:
            timer_sec = None
            raw_timer_secs = []

        start_timer_raws.append(raw_timer_secs)

        if timer_sec is None or pd.isna(timer_sec):
            offset = np.nan
            search_start_sec = np.nan
            timer_sec_out = np.nan

        elif timer_sec >= 60:
            offset = 0.0
            search_start_sec = start_sec
            timer_sec_out = timer_sec

        else:
            offset = float(max(timer_sec - 1, 0))
            search_start_sec = start_sec + offset
            timer_sec_out = timer_sec

        start_timer_secs.append(timer_sec_out)
        offsets.append(offset)
        search_start_secs.append(search_start_sec)

    out["start_timer_sec"] = start_timer_secs
    out["start_search_offset_sec"] = offsets
    out["search_start_sec"] = search_start_secs
    out["start_timer_raws"] = start_timer_raws

    return out


def scan_round_numbers_between(video_path, start_sec, end_sec, reader, step):
    scan_results = []
    if pd.isna(start_sec) or pd.isna(end_sec) or end_sec < start_sec:
        return scan_results

    t_sec = start_sec
    while t_sec <= end_sec:
        try:
            round_no = read_round_number_at_sec(video_path, t_sec, reader)
            error = None
        except Exception as exc:
            round_no = None
            error = str(exc)
        scan_results.append({"t_sec": round(t_sec, 3), "round_no": round_no, "error": error})
        t_sec += step

    if not scan_results or scan_results[-1]["t_sec"] < round(end_sec, 3):
        try:
            round_no = read_round_number_at_sec(video_path, end_sec, reader)
            error = None
        except Exception as exc:
            round_no = None
            error = str(exc)
        scan_results.append({"t_sec": round(end_sec, 3), "round_no": round_no, "error": error})

    return scan_results


def valid_round_observations(scan_results):
    observations = []
    for result in scan_results:
        round_no = to_int(result.get("round_no"))
        t_sec = to_float(result.get("t_sec"))
        if round_no is not None and not pd.isna(t_sec):
            observations.append({"t_sec": t_sec, "round_no": round_no})
    return observations


def estimate_round_interval(round_no, observations, fallback_start_sec, fallback_end_sec):
    times = [obs["t_sec"] for obs in observations if obs["round_no"] == round_no]
    if not times:
        return None

    first_t = min(times)
    last_t = max(times)
    prev_obs = [
        obs
        for obs in observations
        if obs["t_sec"] < first_t and obs["round_no"] != round_no
    ]
    next_obs = [
        obs
        for obs in observations
        if obs["t_sec"] > last_t and obs["round_no"] != round_no
    ]

    if prev_obs:
        prev_t = max(obs["t_sec"] for obs in prev_obs)
        start_sec = (prev_t + first_t) / 2.0
    else:
        start_sec = fallback_start_sec

    if next_obs:
        next_t = min(obs["t_sec"] for obs in next_obs)
        end_sec = (last_t + next_t) / 2.0
    else:
        end_sec = fallback_end_sec

    if end_sec < start_sec:
        return None
    return start_sec, end_sec


def find_transition_boundary(from_round, to_round, observations):
    prev_t = None
    next_t = None
    for obs in observations:
        if obs["round_no"] == from_round:
            prev_t = obs["t_sec"]
        elif obs["round_no"] == to_round and prev_t is not None:
            next_t = obs["t_sec"]
            break
    if prev_t is None or next_t is None or next_t < prev_t:
        return None
    return (prev_t + next_t) / 2.0


def row_has_round_transition(row):
    start_round = to_int(row.get("round_no_ocr_start"))
    end_round = to_int(row.get("round_no_ocr_end"))
    return start_round is not None and end_round is not None and start_round != end_round


def make_inserted_missing_round(
    missing_round_no,
    start_sec,
    end_sec,
    prev_row,
    cur_row,
    prev_round_no,
    previous_boundary_sec,
    step,
):
    inserted = {
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": end_sec - start_sec,
        "round_no_ocr_start": missing_round_no,
        "round_no_ocr_end": missing_round_no,
        "prev_round_no_ocr": prev_round_no,
        "gap_from_prev": start_sec - previous_boundary_sec,
        "ocr_step": step,
        "sequence_status": "inserted_missing_round",
        "action_candidate": "inserted",
        "map_no": prev_row.get("map_no", cur_row.get("map_no", np.nan)),
        "round_no": np.nan,
        "left_score": np.nan,
        "right_score": np.nan,
        "left_win_label": np.nan,
        "fix_applied": "insert_missing_round_between",
        "needs_review": True,
        "_input_index": np.nan,
    }
    if "map" in prev_row or "map" in cur_row:
        inserted["map"] = prev_row.get("map", cur_row.get("map", np.nan))
    return inserted


def scan_missing_rounds(video_path, prev_row, cur_row, reader, step):
    prev_end = to_float(prev_row.get("end_sec"))
    cur_start = to_float(cur_row.get("start_sec"))
    if pd.isna(prev_end) or pd.isna(cur_start) or cur_start < prev_end:
        return [], [], [], []

    prev_round = to_int(prev_row.get("round_no_ocr_start"))
    if prev_round is None:
        prev_round = to_int(prev_row.get("round_no_ocr_end"))
    cur_round = to_int(cur_row.get("round_no_ocr_start"))
    if prev_round is None or cur_round is None or cur_round <= prev_round + 1:
        return [], [], [], []

    missing_rounds = list(range(prev_round + 1, cur_round))
    scan_results = scan_round_numbers_between(video_path, prev_end, cur_start, reader, step)
    observations = valid_round_observations(scan_results)

    inserted_rows = []
    previous_boundary_sec = prev_end
    prev_round_no = prev_round
    unresolved_rounds = []
    for missing_round_no in missing_rounds:
        interval = estimate_round_interval(
            missing_round_no,
            observations,
            fallback_start_sec=previous_boundary_sec,
            fallback_end_sec=cur_start,
        )
        if interval is None:
            unresolved_rounds.append(missing_round_no)
            continue
        start_sec, end_sec = interval
        inserted = make_inserted_missing_round(
            missing_round_no,
            start_sec,
            end_sec,
            prev_row,
            cur_row,
            prev_round_no,
            previous_boundary_sec,
            step,
        )
        inserted_rows.append(inserted)
        previous_boundary_sec = inserted["end_sec"]
        prev_round_no = missing_round_no

    return inserted_rows, scan_results, missing_rounds, unresolved_rounds


def split_row_on_round_transition(row, video_path, reader, step):
    start_round = to_int(row.get("round_no_ocr_start"))
    end_round = to_int(row.get("round_no_ocr_end"))
    start_sec = to_float(row.get("start_sec"))
    end_sec = to_float(row.get("end_sec"))
    if (
        start_round is None
        or end_round is None
        or start_round == end_round
        or pd.isna(start_sec)
        or pd.isna(end_sec)
        or end_sec <= start_sec
    ):
        return None, [], []

    scan_results = scan_round_numbers_between(video_path, start_sec, end_sec, reader, step)
    observations = valid_round_observations(scan_results)
    boundary_sec = find_transition_boundary(start_round, end_round, observations)
    if boundary_sec is None or boundary_sec <= start_sec or boundary_sec >= end_sec:
        return None, scan_results, []

    first_duration = boundary_sec - start_sec
    second_duration = end_sec - boundary_sec
    if (
        first_duration <= MIN_SPLIT_FRAGMENT_SEC
        and second_duration <= MIN_SPLIT_FRAGMENT_SEC
    ):
        return None, scan_results, []

    first = row.copy()
    second = row.copy()
    split_detail = {
        "boundary_sec": boundary_sec,
        "first_duration": first_duration,
        "second_duration": second_duration,
        "dropped_fragment": None,
    }

    first["end_sec"] = boundary_sec
    first["duration_sec"] = first_duration
    first["round_no_ocr_start"] = start_round
    first["round_no_ocr_end"] = start_round
    first["sequence_status"] = "split_round_transition"
    first["action_candidate"] = "split"
    first["fix_applied"] = append_fix(first.get("fix_applied"), "split_round_transition")
    first["needs_review"] = True

    second["start_sec"] = boundary_sec
    second["duration_sec"] = second_duration
    second["round_no_ocr_start"] = end_round
    second["round_no_ocr_end"] = end_round
    second["prev_round_no_ocr"] = start_round
    second["gap_from_prev"] = 0.0
    second["sequence_status"] = "split_round_transition"
    second["action_candidate"] = "split"
    second["fix_applied"] = append_fix(second.get("fix_applied"), "split_round_transition")
    second["needs_review"] = True

    if (
        first_duration <= MIN_SPLIT_FRAGMENT_SEC
        and second_duration > MIN_SPLIT_FRAGMENT_SEC
    ):
        second["fix_applied"] = append_fix(
            second.get("fix_applied"), "split_round_transition_drop_short_first"
        )
        split_detail["dropped_fragment"] = "first"
        return [second], scan_results, split_detail

    if (
        second_duration <= MIN_SPLIT_FRAGMENT_SEC
        and first_duration > MIN_SPLIT_FRAGMENT_SEC
    ):
        first["fix_applied"] = append_fix(
            first.get("fix_applied"), "split_round_transition_drop_short_second"
        )
        split_detail["dropped_fragment"] = "second"
        return [first], scan_results, split_detail

    return [first, second], scan_results, split_detail


def merge_with_previous(prev_row, cur_row):
    merged = prev_row.copy()
    start_sec = to_float(prev_row.get("start_sec"))
    end_sec = max(to_float(prev_row.get("end_sec")), to_float(cur_row.get("end_sec")))
    merged["start_sec"] = start_sec
    merged["end_sec"] = end_sec
    merged["duration_sec"] = end_sec - start_sec
    if is_missing(merged.get("round_no_ocr_start")):
        merged["round_no_ocr_start"] = cur_row.get("round_no_ocr_start")
    if not is_missing(cur_row.get("round_no_ocr_end")):
        merged["round_no_ocr_end"] = cur_row.get("round_no_ocr_end")
    elif is_missing(merged.get("round_no_ocr_end")):
        merged["round_no_ocr_end"] = prev_row.get("round_no_ocr_end")
    merged["fix_applied"] = append_fix(merged.get("fix_applied"), "merge_with_previous")
    merged["needs_review"] = False
    return merged


def process_rows(df, video_path, reader, round_search_step, apply_drop):
    fixed_rows = []
    log_rows = []

    for input_index, row in df.iterrows():
        cur = row_to_dict(row, input_index=input_index)
        action = str(cur.get("action_candidate", "")).strip()
        status = str(cur.get("sequence_status", "")).strip()

        if action == "merge_with_previous" and fixed_rows:
            prev = fixed_rows.pop()
            merged = merge_with_previous(prev, cur)
            fixed_rows.append(merged)
            detail = {
                "prev_index": prev.get("_input_index"),
                "cur_index": cur.get("_input_index"),
                "prev_round_no_ocr_start": prev.get("round_no_ocr_start"),
                "prev_round_no_ocr_end": prev.get("round_no_ocr_end"),
                "cur_round_no_ocr_start": cur.get("round_no_ocr_start"),
                "cur_round_no_ocr_end": cur.get("round_no_ocr_end"),
                "prev_gap_from_prev": prev.get("gap_from_prev"),
                "cur_gap_from_prev": cur.get("gap_from_prev"),
            }
            log_rows.append(
                make_log(cur, "merge_with_previous", dumps_json(detail), prev=prev, cur=cur)
            )
            continue

        if action == "drop_previous" and apply_drop and fixed_rows:
            prev = fixed_rows[-1]
            prev_duration = to_float(prev.get("duration_sec"))
            cur_duration = to_float(cur.get("duration_sec"))
            drop_prev = pd.isna(cur_duration) or (
                not pd.isna(prev_duration) and prev_duration < cur_duration
            )
            if drop_prev:
                dropped = fixed_rows.pop()
                cur["fix_applied"] = append_fix(cur.get("fix_applied"), "drop_shorter_previous")
                cur["needs_review"] = False
                fixed_rows.append(cur)
                fix_applied = "drop_shorter_previous"
                dropped_index = dropped.get("_input_index")
            else:
                fixed_rows[-1]["fix_applied"] = append_fix(
                    fixed_rows[-1].get("fix_applied"), "drop_shorter_current"
                )
                fixed_rows[-1]["needs_review"] = False
                fix_applied = "drop_shorter_current"
                dropped_index = cur.get("_input_index")
            detail = {
                "prev_duration_sec": prev_duration,
                "cur_duration_sec": cur_duration,
                "prev_round_no_ocr_start": prev.get("round_no_ocr_start"),
                "cur_round_no_ocr_start": cur.get("round_no_ocr_start"),
            }
            log_rows.append(
                make_log(
                    cur,
                    fix_applied,
                    dumps_json(detail),
                    prev=prev,
                    cur=cur,
                    dropped_index=dropped_index,
                )
            )
            continue

        if video_path is not None and reader is not None and row_has_round_transition(cur):
            split_rows, split_scan_results, split_detail = split_row_on_round_transition(
                cur, video_path, reader, round_search_step
            )
            if split_rows is not None:
                for split_row in split_rows:
                    fixed_rows.append(split_row)
                detail = {
                    "split_from_index": cur.get("_input_index"),
                    "from_round": to_int(cur.get("round_no_ocr_start")),
                    "to_round": to_int(cur.get("round_no_ocr_end")),
                }
                if isinstance(split_detail, dict):
                    detail.update(split_detail)
                log_rows.append(
                    make_log(
                        cur,
                        "split_round_transition",
                        dumps_json(detail),
                        cur=cur,
                        split_from_index=cur.get("_input_index"),
                        boundary_sec=(
                            split_detail.get("boundary_sec")
                            if isinstance(split_detail, dict)
                            else np.nan
                        ),
                        ocr_scan_results=dumps_json(split_scan_results),
                    )
                )
                continue
            cur["fix_applied"] = append_fix(
                cur.get("fix_applied"), "split_round_transition_not_resolved"
            )
            cur["needs_review"] = True
            detail = {
                "split_from_index": cur.get("_input_index"),
                "from_round": to_int(cur.get("round_no_ocr_start")),
                "to_round": to_int(cur.get("round_no_ocr_end")),
                "reason": "No OCR transition boundary found",
            }
            log_rows.append(
                make_log(
                    cur,
                    "split_round_transition_not_resolved",
                    dumps_json(detail),
                    cur=cur,
                    split_from_index=cur.get("_input_index"),
                    ocr_scan_results=dumps_json(split_scan_results),
                )
            )
            fixed_rows.append(cur)
            continue

        should_inspect_between = (
            action == "inspect_between" or status == "missing_round_between"
        )
        if should_inspect_between and fixed_rows and video_path is not None and reader is not None:
            prev = fixed_rows[-1]
            inserted_rows, scan_results, missing_rounds, unresolved_rounds = scan_missing_rounds(
                video_path, prev, cur, reader, round_search_step
            )
            scan_json = dumps_json(scan_results)
            for inserted in inserted_rows:
                fixed_rows.append(inserted)
                log_rows.append(
                    make_log(
                        cur,
                        "insert_missing_round_between",
                        "Inserted missing round found by OCR scan",
                        prev=prev,
                        cur=cur,
                        inserted_round_no=inserted.get("round_no_ocr_start"),
                        inserted_start_sec=inserted.get("start_sec"),
                        inserted_end_sec=inserted.get("end_sec"),
                        ocr_scan_results=scan_json,
                    )
                )
            if inserted_rows and not unresolved_rounds:
                cur["needs_review"] = False
            if unresolved_rounds or (missing_rounds and not inserted_rows):
                cur["fix_applied"] = append_fix(
                    cur.get("fix_applied"), "missing_round_between_not_resolved"
                )
                cur["needs_review"] = True
                detail = {
                    "missing_rounds": missing_rounds,
                    "inserted_rounds": [
                        row.get("round_no_ocr_start") for row in inserted_rows
                    ],
                    "unresolved_rounds": unresolved_rounds,
                }
                log_rows.append(
                    make_log(
                        cur,
                        "missing_round_between_not_resolved",
                        dumps_json(detail),
                        prev=prev,
                        cur=cur,
                        ocr_scan_results=scan_json,
                    )
                )

        if action == "manual_check" or status == "map_change_or_ocr_error":
            cur["needs_review"] = True
            cur["fix_applied"] = append_fix(cur.get("fix_applied"), "manual_review_required")
            log_rows.append(
                make_log(cur, cur["fix_applied"], "Kept row for manual review", cur=cur)
            )
        elif action == "keep" or status == "first":
            cur["needs_review"] = bool(cur.get("needs_review", False))

        fixed_rows.append(cur)

    return fixed_rows, log_rows


def add_map_ocr(fixed_rows, video_path, reader):
    if video_path is None or reader is None:
        for row in fixed_rows:
            row.setdefault("map", np.nan)
        return []

    map_logs = []
    for row in fixed_rows:
        if row.get("sequence_status") == "inserted_missing_round":
            row.setdefault("map", row.get("map", np.nan))
            continue
        current_map, samples = read_map_for_round(video_path, row.get("start_sec"), reader)
        row["map"] = current_map if current_map is not None else np.nan
        if current_map is None:
            row["needs_review"] = True
        map_logs.append(
            make_log(
                row,
                "map_ocr",
                f"current_map={current_map}",
                cur=row,
                ocr_scan_results=dumps_json(samples),
            )
        )
    return map_logs


def is_split_series(df):
    action = df.get("action_candidate", pd.Series(index=df.index, dtype=object))
    status = df.get("sequence_status", pd.Series(index=df.index, dtype=object))
    return action.eq("split") | status.eq("split_round_transition")


def refresh_split_scores(fixed_df, video_path=None, reader=None):
    fixed_df = fixed_df.copy()
    for col in [
        "left_score",
        "right_score",
        "left_score_candidates",
        "right_score_candidates",
    ]:
        if col not in fixed_df.columns:
            fixed_df[col] = np.nan
    fixed_df["left_score_candidates"] = fixed_df["left_score_candidates"].astype(object)
    fixed_df["right_score_candidates"] = fixed_df["right_score_candidates"].astype(object)

    split_mask = is_split_series(fixed_df)
    if not split_mask.any():
        return fixed_df

    score_cols = [
        "left_score",
        "right_score",
        "left_score_candidates",
        "right_score_candidates",
    ]
    fixed_df.loc[split_mask, score_cols] = np.nan

    if video_path is None or reader is None:
        return fixed_df

    for idx, row in fixed_df[split_mask].iterrows():
        start_sec = to_float(row.get("start_sec"))
        if pd.isna(start_sec):
            continue
        t_list = [start_sec + 1.0, start_sec + 2.0, start_sec + 3.0]
        try:
            score_result = read_score_multi_sec(video_path, t_list, reader)
        except Exception:
            continue

        left_score = score_result.get("left_score")
        right_score = score_result.get("right_score")
        if not is_missing(left_score):
            fixed_df.at[idx, "left_score"] = left_score
        if not is_missing(right_score):
            fixed_df.at[idx, "right_score"] = right_score
        fixed_df.at[idx, "left_score_candidates"] = dumps_json(
            score_result.get("left_candidates", [])
        )
        fixed_df.at[idx, "right_score_candidates"] = dumps_json(
            score_result.get("right_candidates", [])
        )

    return fixed_df


def recalculate_columns(fixed_df, video_path=None, reader=None):
    fixed_df = fixed_df.copy().sort_values("start_sec").reset_index(drop=True)
    fixed_df["duration_sec"] = fixed_df["end_sec"] - fixed_df["start_sec"]
    fixed_df["t_sec"] = fixed_df["start_sec"]
    fixed_df["t_min"] = fixed_df["start_sec"] / 60.0

    fixed_df = refresh_split_scores(fixed_df, video_path=video_path, reader=reader)

    if "map" not in fixed_df.columns:
        fixed_df["map"] = np.nan

    new_map_nos = []
    current_map_no = None
    current_map = None
    last_existing_map_no = None
    for _, row in fixed_df.iterrows():
        row_map = none_if_missing(row.get("map"))
        existing_map_no = to_int(row.get("map_no"))
        if existing_map_no is not None:
            last_existing_map_no = existing_map_no

        if row_map is not None:
            if current_map_no is None:
                current_map_no = existing_map_no or last_existing_map_no or 1
            elif current_map is not None and row_map != current_map:
                current_map_no += 1
            current_map = row_map
        elif current_map_no is None:
            current_map_no = existing_map_no or last_existing_map_no or 1

        new_map_nos.append(current_map_no)
    fixed_df["map_no"] = new_map_nos

    fixed_df["round_no"] = (
        fixed_df.groupby("map_no", dropna=False).cumcount() + 1
    )

    left = pd.to_numeric(fixed_df.get("left_score"), errors="coerce")
    right = pd.to_numeric(fixed_df.get("right_score"), errors="coerce")
    fixed_df["score_total"] = left + right
    fixed_df["round_no_from_score"] = fixed_df["score_total"] + 1

    if "_input_index" in fixed_df.columns:
        fixed_df = fixed_df.drop(columns=["_input_index"])
    return fixed_df


def print_consistency_checks(df):
    start = pd.to_numeric(df.get("start_sec"), errors="coerce")
    t_sec = pd.to_numeric(df.get("t_sec"), errors="coerce")
    t_mismatch = ((start - t_sec).abs() > 1e-6).fillna(False).sum()

    round_no = pd.to_numeric(df.get("round_no"), errors="coerce")
    round_no_from_score = pd.to_numeric(df.get("round_no_from_score"), errors="coerce")
    comparable = round_no.notna() & round_no_from_score.notna()
    round_mismatch = (round_no[comparable] != round_no_from_score[comparable]).sum()

    split_mask = is_split_series(df)
    left = pd.to_numeric(df.get("left_score"), errors="coerce")
    right = pd.to_numeric(df.get("right_score"), errors="coerce")
    split_score_nan = (split_mask & (left.isna() | right.isna())).sum()
    duration = pd.to_numeric(df.get("duration_sec"), errors="coerce")
    short_duration_rows = (duration <= MIN_SPLIT_FRAGMENT_SEC).fillna(False).sum()

    print("Consistency checks:")
    print(f"t_sec != start_sec rows: {int(t_mismatch)}")
    print(f"round_no_from_score != round_no rows: {int(round_mismatch)}")
    print(f"split rows with NaN left_score/right_score: {int(split_score_nan)}")
    print(f"duration_sec <= {MIN_SPLIT_FRAGMENT_SEC:g} rows: {int(short_duration_rows)}")


def print_summary(label, df):
    print(f"{label} rows: {len(df)}")
    if "action_candidate" in df.columns:
        print(f"{label} action_candidate counts:")
        print(df["action_candidate"].value_counts(dropna=False).to_string())
    if "needs_review" in df.columns:
        print(f"{label} needs_review counts:")
        print(df["needs_review"].value_counts(dropna=False).to_string())


def recalculate_left_win_label(df):
    out = df.copy()
    if "left_win_label" not in out.columns:
        out["left_win_label"] = pd.NA

    left_win = pd.Series(pd.NA, index=out.index, dtype="Int64")
    work = out.copy()
    work["_left_score_num"] = pd.to_numeric(work.get("left_score"), errors="coerce")
    work["_right_score_num"] = pd.to_numeric(work.get("right_score"), errors="coerce")
    work["_sort_round_no"] = pd.to_numeric(work.get("round_no"), errors="coerce")
    work["_sort_start_sec"] = pd.to_numeric(work.get("start_sec"), errors="coerce")
    work = work.sort_values(
        ["map_no", "_sort_round_no", "_sort_start_sec"],
        na_position="last",
    )

    for _, group in work.groupby("map_no", dropna=False, sort=False):
        indices = group.index.tolist()
        for pos, idx in enumerate(indices):
            cur = group.loc[idx]
            cur_left = cur["_left_score_num"]
            cur_right = cur["_right_score_num"]
            if pd.isna(cur_left) or pd.isna(cur_right):
                continue

            if pos + 1 < len(indices):
                nxt = group.loc[indices[pos + 1]]
                next_left = nxt["_left_score_num"]
                next_right = nxt["_right_score_num"]
                if pd.isna(next_left) or pd.isna(next_right):
                    continue
                if next_left > cur_left and next_right == cur_right:
                    left_win.at[idx] = 1
                elif next_right > cur_right and next_left == cur_left:
                    left_win.at[idx] = 0
                continue

            if cur_left != cur_right:
                left_win.at[idx] = 1 if cur_left > cur_right else 0

    out["left_win_label"] = left_win
    return out


def is_left_attacker(round_no: int) -> bool:
    if 1 <= round_no <= 12:
        return False
    if 13 <= round_no <= 24:
        return True
    if round_no >= 25:
        return (round_no - 25) % 2 == 1
    raise ValueError(f"invalid round_no: {round_no}")

def make_attacker_win(row):
    left_win = row["left_win_label"]
    round_no = row.get("round_no")
    if pd.isna(left_win) or pd.isna(round_no):
        return pd.NA
    left_win = int(left_win)
    left_attacker = is_left_attacker(int(round_no))
    if left_attacker:
        return left_win
    else:
        return 1 - left_win


def add_attack_defense_win_columns(df):
    out = df.copy()
    attacker_win = out.apply(make_attacker_win, axis=1)
    out["attacker_win"] = pd.Series(attacker_win, index=out.index).astype("Int64")
    out["defender_win"] = (1 - out["attacker_win"]).astype("Int64")
    return out


def print_win_debug(df):
    print("left_win_label value counts:")
    print(df["left_win_label"].value_counts(dropna=False).to_string())
    print("attacker_win value counts:")
    print(df["attacker_win"].value_counts(dropna=False).to_string())
    print(f"left_win_label NaN rows: {int(df['left_win_label'].isna().sum())}")
    print(f"attacker_win NaN rows: {int(df['attacker_win'].isna().sum())}")

    sort_cols = ["map_no", "round_no", "start_sec"]
    existing_sort_cols = [col for col in sort_cols if col in df.columns]
    last_rows = (
        df.sort_values(existing_sort_cols)
        .groupby("map_no", dropna=False, sort=False)
        .tail(1)
    )
    debug_cols = [
        "map_no",
        "round_no",
        "left_score",
        "right_score",
        "left_win_label",
        "attacker_win",
    ]
    print("Last row by map:")
    print(last_rows[debug_cols].to_string(index=False))


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    default_fixed, default_log = default_output_paths(input_path)
    output_path = Path(args.output) if args.output else default_fixed
    log_output_path = Path(args.log_output) if args.log_output else default_log
    if input_path.resolve() in {output_path.resolve(), log_output_path.resolve()}:
        raise ValueError("Refusing to overwrite the input checked CSV")

    checked = pd.read_csv(input_path)
    print_summary("Before", checked)

    video_path = Path(args.video) if args.video else None
    reader = None
    needs_video_ocr = video_path is not None
    if needs_video_ocr:
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        import easyocr

        reader = easyocr.Reader(["en"], gpu=args.gpu)
    else:
        print("No --video was provided; missing-round scans and map OCR will be skipped.")

    fixed_rows, log_rows = process_rows(
        checked,
        video_path=video_path,
        reader=reader,
        round_search_step=args.round_search_step,
        apply_drop=args.apply_drop,
    )
    log_rows.extend(add_map_ocr(fixed_rows, video_path=video_path, reader=reader))

    fixed_df = pd.DataFrame(fixed_rows)
    fixed_df = recalculate_columns(fixed_df, video_path=video_path, reader=reader)
    fixed_df = add_start_timer_offset_columns(fixed_df, video_path=video_path, reader=reader)
    fixed_df = recalculate_left_win_label(fixed_df)
    fixed_df = add_attack_defense_win_columns(fixed_df)
    log_df = pd.DataFrame(log_rows, columns=LOG_COLUMNS)

    print_summary("After", fixed_df)
    print_consistency_checks(fixed_df)
    print_win_debug(fixed_df)
    print(f"Fix log rows: {len(log_df)}")

    if args.dry_run:
        print("Dry run: CSV files were not written.")
        if not log_df.empty:
            print("Log preview:")
            print(log_df.head(20).to_string(index=False))
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_output_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_df.to_csv(output_path, index=False)
    log_df.to_csv(log_output_path, index=False)
    print(f"Wrote fixed CSV: {output_path}")
    print(f"Wrote fix log CSV: {log_output_path}")


if __name__ == "__main__":
    main()
