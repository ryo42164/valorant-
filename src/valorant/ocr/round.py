from __future__ import annotations

import pandas as pd

from valorant.ocr.preprocess import preprocess_round_crop
from valorant.regions import crop_round_region
from valorant.utils import majority_vote_raw
from valorant.video import get_frame_at_sec


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
    return majority_vote_raw([sample["round_no"] for sample in samples])


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
