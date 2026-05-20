from __future__ import annotations

from collections import Counter

import pandas as pd


def majority_vote_raw(values):
    """Return the most common value, ignoring only None.

    This preserves scripts/label_rounds_from_ui_diff.py behavior. It does not
    ignore pandas NaN/pd.NA values.
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def majority_vote_ignore_na(values):
    """Return the most common value, ignoring None and pandas missing values.

    This preserves the fallback behavior embedded in
    scripts/fix_round_labels_from_actions.py. It is intentionally separate from
    majority_vote_raw because unifying them could change label results.
    """
    vals = [v for v in values if v is not None and not pd.isna(v)]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]
