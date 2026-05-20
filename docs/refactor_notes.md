# Refactor Notes

This refactor is intentionally behavior-preserving. The goal is to move reusable
helpers under `src/valorant/` without changing labeling, OCR, crop, or pandas
missing-value behavior.

## Preserved Risky Behavior

- `majority_vote_raw` ignores only `None`, matching
  `scripts/label_rounds_from_ui_diff.py`.
- `majority_vote_ignore_na` ignores `None` and pandas missing values, matching
  the fallback helper inside `scripts/fix_round_labels_from_actions.py`.
- These two behaviors are intentionally not unified.

## TODOs For A Separate Branch

- Decide whether missing-value handling should be unified across scripts.
- Decide whether fallback imports in `scripts/fix_round_labels_from_actions.py`
  should be replaced by direct `src/valorant/` imports after CLI compatibility
  tests are expanded.
- Review notebook-only functions manually before moving them. Some notebooks have
  invalid JSON or mojibake and should not be mechanically converted.
- Document the exact source video resolution assumptions for each crop region.
  Current crop functions keep the existing ratio values and must not change crop
  shapes in this refactor.
- Split and move map/timer post-processing helpers from
  `scripts/fix_round_labels_from_actions.py` only after confirming the existing
  fallback behavior remains available.
