"""
baseline.py
-----------
Simple baseline score generators that return M non-negative scores for the
CURRENT candidate buffer. The env will convert scores -> integer allocations.

Policies
- SJF: smaller man_days_remaining => higher score
- LJF: larger man_days_remaining  => higher score
- RANDOM: random scores on valid candidates
"""

from __future__ import annotations
import numpy as np
from housegymrl import HousegymRLENV

def make_baseline_scores(env: HousegymRLENV, policy: str = "SJF") -> np.ndarray:
    c = env.last_candidate_view()
    remain, mask = c["remain"], c["mask"]
    M = remain.shape[0]
    scores = np.zeros(M, dtype=np.float32)

    valid = mask > 0.0
    if not valid.any():
        return scores

    if policy == "SJF":
        r = remain.copy()
        r[~valid] = r[valid].max() + 1.0
        scores = (r.max() + 1e-6) - r
    elif policy == "LJF":
        r = remain.copy()
        r[~valid] = -1.0
        scores = r - r.min()
    else:  # RANDOM
        rng = np.random.default_rng()
        scores = rng.random(M, dtype=np.float32)
        scores[~valid] = 0.0

    return np.clip(np.nan_to_num(scores, nan=0.0), 0.0, None)
