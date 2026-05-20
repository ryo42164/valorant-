from __future__ import annotations

import re

from valorant.ocr.preprocess import preprocess_score_crop
from valorant.regions import crop_left_right_scores, crop_score_banner
from valorant.utils import majority_vote_raw
from valorant.video import get_frame_at_sec


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
        "left_score": majority_vote_raw(left_scores),
        "right_score": majority_vote_raw(right_scores),
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
