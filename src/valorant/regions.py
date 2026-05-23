from __future__ import annotations


def crop_ui_region(frame):
    """Crop the top-center round UI region using the existing ratio values.

    These coordinates are ratio-based values from the current notebooks/scripts.
    Do not change them in this refactor; changing them can alter crop shape and
    OCR/label results.
    """
    h, w = frame.shape[:2]
    x1 = int(w * 0.35)
    x2 = int(w * 0.65)
    y1 = int(h * 0.0)
    y2 = int(h * 0.06)
    return frame[y1:y2, x1:x2]


def crop_score_banner(frame):
    """Crop the score banner with the existing ratio values."""
    h, w = frame.shape[:2]
    y1 = int(h * 0.005)
    y2 = int(h * 0.06)
    x1 = int(w * 0.31)
    x2 = int(w * 0.69)
    return frame[y1:y2, x1:x2]


def crop_left_right_scores(banner):
    """Crop left/right score digits from the existing score banner crop."""
    h, w = banner.shape[:2]
    left_crop = banner[int(h * 0.12) : int(h * 0.88), int(w * 0.25) : int(w * 0.38)]
    right_crop = banner[int(h * 0.12) : int(h * 0.88), int(w * 0.62) : int(w * 0.75)]
    return left_crop, right_crop


def crop_round_region(frame):
    """Crop the round-number region with the existing ratio values."""
    h, w = frame.shape[:2]
    x1 = int(w * 0.51)
    x2 = int(w * 0.527)
    y1 = int(h * 0.01)
    y2 = int(h * 0.025)
    return frame[y1:y2, x1:x2]