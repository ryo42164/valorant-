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

try:
    from label_rounds_from_ui_diff import (
        get_frame_at_sec,
        majority_vote,
        read_round_number_at_sec,
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


def make_inserted_missing_round(
    missing_round_no,
    times,
    prev_row,
    cur_row,
    prev_round_no,
    previous_boundary_sec,
    step,
):
    missing_start_sec = min(times)
    missing_end_sec = max(times)
    inserted = {
        "start_sec": missing_start_sec,
        "end_sec": missing_end_sec,
        "duration_sec": missing_end_sec - missing_start_sec,
        "round_no_ocr_start": missing_round_no,
        "round_no_ocr_end": missing_round_no,
        "prev_round_no_ocr": prev_round_no,
        "gap_from_prev": missing_start_sec - previous_boundary_sec,
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
    wanted = set(missing_rounds)
    scan_results = []
    found_by_round = {round_no: [] for round_no in wanted}
    t_sec = prev_end
    while t_sec <= cur_start:
        try:
            round_no = read_round_number_at_sec(video_path, t_sec, reader)
            error = None
        except Exception as exc:
            round_no = None
            error = str(exc)
        scan_results.append({"t_sec": round(t_sec, 3), "round_no": round_no, "error": error})
        if round_no in found_by_round:
            found_by_round[round_no].append(t_sec)
        t_sec += step

    inserted_rows = []
    previous_boundary_sec = prev_end
    prev_round_no = prev_round
    unresolved_rounds = []
    for missing_round_no in missing_rounds:
        times = found_by_round[missing_round_no]
        if not times:
            unresolved_rounds.append(missing_round_no)
            continue
        inserted = make_inserted_missing_round(
            missing_round_no,
            times,
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


def recalculate_columns(fixed_df):
    fixed_df = fixed_df.copy().sort_values("start_sec").reset_index(drop=True)
    fixed_df["duration_sec"] = fixed_df["end_sec"] - fixed_df["start_sec"]

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


def print_summary(label, df):
    print(f"{label} rows: {len(df)}")
    if "action_candidate" in df.columns:
        print(f"{label} action_candidate counts:")
        print(df["action_candidate"].value_counts(dropna=False).to_string())
    if "needs_review" in df.columns:
        print(f"{label} needs_review counts:")
        print(df["needs_review"].value_counts(dropna=False).to_string())


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
    fixed_df = recalculate_columns(fixed_df)
    log_df = pd.DataFrame(log_rows, columns=LOG_COLUMNS)

    print_summary("After", fixed_df)
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
