from __future__ import annotations

import cv2


def preprocess_score_crop(crop, scale=4):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    return th


def preprocess_round_crop(crop, scale=3):
    """Preprocess round-number crops using the existing multi-threshold flow."""
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
