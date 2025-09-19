
# Why two-stage allocation?
# --------------
# If we allocate only within the M-sized candidate buffer, the daily throughput
# is upper-bounded by sum(cmax over candidates), i.e., ~ M * average(cmax),
# regardless of how large the global capacity K is. In practice M<<|tasks|,
# so this becomes an artificial bottleneck.
# The spillover stage removes that bottleneck by allocating any remaining capacity
# to eligible non-candidate tasks (e.g., SJF-style), so the effective throughput
# approaches min(K, sum(cmax over all eligible tasks)).
#

#
# Observation and action:
# -----------------------
# - Action is a continuous score in [0,1]^M over *candidates only* (fixed M),
#   which is stable for SB3 and scales to large problem sizes.
# - Stage 1: convert scores -> integer allocations under (global K, per-task cmax).
# - Stage 2 (optional, on by default): allocate remaining capacity K_rem to
#   eligible non-candidate tasks via a fast SJF-like rule.
#
# Reward:
# ---------------------------------
# info["completion"] = completed_households / total_households, and use its daily increment as reward.
# with evaluation plots/RMSE that are defined on *household completion*, not man-days.

from __future__ import annotations
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Dict, Tuple, Optional


M_CANDIDATES = 96      # candidate buffer size
MAX_STEPS    = 1200    # episode day cap
OBS_G        = 6       # global feature dims
OBS_F        = 4       # per-candidate feature dims


#   Vectorized residual assignment:
#       - normalize on valid entries (mask==1)
#       - expected = w * K
#       - base = floor(expected)
#       - cap by cmax; then assign +1 to the top-k fractional parts (k = remaining)
def allocate_workers(scores: np.ndarray,
                     K: int,
                     cmax: np.ndarray,
                     mask: np.ndarray) -> Tuple[np.ndarray, int]:
    scores = np.nan_to_num(scores, nan=0.0)
    scores = np.clip(scores, 0.0, 1.0).astype(np.float32, copy=False)
    m = mask.astype(np.float32, copy=False)

    # normalize weights on valid candidates
    w = scores * m
    if w.sum() <= 1e-8:
        w = m  # fallback: uniform over valid ones
    ws = w.sum()
    if ws > 0:
        w = w / ws

    expected = w * float(K)                       # continuous shares
    base = np.floor(expected)
    # cap by cmax
    alloc = np.minimum(base, cmax).astype(np.int32, copy=False)

    rem = int(K - int(alloc.sum()))
    if rem > 0:
        # fractional parts; block invalid or capped rows
        frac = (expected - base).astype(np.float32, copy=False)
        blocked = (m <= 0.0) | (alloc >= cmax)
        # Use -inf to exclude blocked rows from top-k
        frac = np.where(blocked, -np.inf, frac)
        # pick top-rem indices in O(M)
        valid_cnt = int(np.sum(~blocked))
        if valid_cnt > 0:
            k = min(rem, valid_cnt)
            top_idx = np.argpartition(-frac, k - 1)[:k]
            alloc[top_idx] += 1
            rem -= k

    idle = max(rem, 0)
    assert alloc.sum() <= K
    return alloc, idle


    # Stage 1 (candidate-first):
    #   - Only within the candidate buffer (size M), convert scores to integer
    #     allocations under per-task cmax and global capacity K.

    # Stage 2 (spillover to non-candidates, optional):
    #   - If there is remaining capacity K_rem after Stage 1, allocate it to
    #     eligible non-candidate tasks using an SJF-like vectorized rule.
    #   - This removes the artificial throughput cap at ~M*avg(cmax).

    # Notes:
    #   - info["completion"] is *household* completion ratio, which aligns with
    #     evaluation plots and RMSE in this project.
    #   - Ramp function k_ramp(day) can scale effective K day-by-day.
class HousegymRLENV(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        tasks_df: Optional[pd.DataFrame] = None,
        resources: Optional[Dict] = None,
        scenario_sampler: Optional[Callable[[], Tuple[pd.DataFrame, Dict, Dict]]] = None,
        M: int = M_CANDIDATES,
        max_steps: int = MAX_STEPS,
        seed: Optional[int] = None,
        # logging controls
        tb_every: int = 200,
        print_every: int = 500,
        # new knobs
        fill_non_candidates: bool = True,      # Stage 2 spillover switch 
        k_ramp: Optional[Callable[[int], float]] = None,  # apacity ramp scaler
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.M = int(M)
        self.max_steps = int(max_steps)
        self.scenario_sampler = scenario_sampler

        # ablation & realism knobs
        self.fill_non_candidates = bool(fill_non_candidates)
        self.k_ramp = k_ramp  # function: day -> [0,1] scale for K

        if tasks_df is None or resources is None:
            if scenario_sampler is None:
                raise ValueError("Provide (tasks_df, resources) or a scenario_sampler().")
            tasks_df, resources, _ = scenario_sampler()

        self._load_scenario(tasks_df, resources)

        obs_dim = OBS_G + self.M * OBS_F
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.M,), dtype=np.float32)

        self._last_candidates = None

        # logging throttles
        self.tb_every = int(tb_every)
        self.print_every = int(print_every)
        self._global_step = 0
        self.writer = None 

 
    def _load_scenario(self, tasks_df: pd.DataFrame, resources: Dict):
    
        t = tasks_df.copy(deep=False).reset_index(drop=True)
        required = ["man_days_total", "man_days_remaining", "delay_days_remaining", "cmax_per_day", "damage_level"]
        for col in required:
            if col not in t.columns:
                raise ValueError(f"missing column: {col}")
        if "delay_days_init" not in t.columns:
            t["delay_days_init"] = t["delay_days_remaining"].astype(int)
        self.tasks = t

        self.K = int(resources["workers"])
        self.region_name = str(resources.get("region_name", "UNKNOWN"))
        self.day = 0
        self.done_flag = False
        self.idle_history = []

        self._arr_delay = self.tasks["delay_days_remaining"].to_numpy(copy=False)
        self._arr_delay_init = self.tasks["delay_days_init"].to_numpy(copy=False)
        self._arr_rem = self.tasks["man_days_remaining"].to_numpy(copy=False)
        self._arr_total = self.tasks["man_days_total"].to_numpy(copy=False)
        self._arr_cmax = self.tasks["cmax_per_day"].to_numpy(copy=False)
        self._arr_dmg = self.tasks["damage_level"].to_numpy(copy=False)
        self._total_work_scalar = float(self._arr_total.sum())

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.scenario_sampler is not None:
            tasks_df, resources, _ = self.scenario_sampler()
            self._load_scenario(tasks_df, resources)
        else:
            np.copyto(self._arr_rem, self._arr_total.astype(np.int64, copy=False))
            np.copyto(self._arr_delay, self._arr_delay_init.astype(np.int64, copy=False))
            self._total_work_scalar = float(self._arr_total.sum())

        self.day = 0
        self.done_flag = False
        self.idle_history.clear()
        self._global_step = 0

        self._refresh_candidates()
        return self._get_obs(), {}

    # ----------- step -----------
    def step(self, action: np.ndarray):
        """
        One day transition with two-stage dispatch.

        Stage 1  (candidate-first):
          - Turn scores into integer allocations under {global K, per-task cmax, mask}.
        Stage 2  (spillover, optional):
          - If capacity remains, send it to non-candidate eligible tasks (SJF-like).

        Reward: Δ( household completion ) - 0.1  (you can adjust the constant
                if you prefer a smaller time penalty or remove it for stability).
        Termination: all households completed; Truncation: day >= max_steps.
        """
        if self.done_flag:
            # when already terminated previously, keep returning a terminal transition
            return self._get_obs(), 0.0, True, False, {}

        # ----capacity ----
        K_eff = int(self.K)
        if self.k_ramp is not None:
            try:
                K_eff = int(max(0, min(self.K, round(float(self.k_ramp(self.day)) * self.K))))
            except Exception:
                K_eff = int(self.K)

        # ---- Before-state stats ----
        total_work = self._total_work_scalar
        remain_before = float(self._arr_rem.sum())
        completion_md_before = 1.0 - (remain_before / (total_work + 1e-8))

        total_houses = int(self._arr_rem.shape[0])
        completed_before = int(np.sum(self._arr_rem <= 0))
        completion_hh_before = completed_before / max(1, total_houses)

        # ---- Stage 1: candidate-only allocation ----
        cand = self._last_candidates
        scores = np.asarray(action, dtype=np.float32).reshape(-1)[:self.M]
        alloc_cand, _idle_dummy = allocate_workers(scores, K_eff, cand["cmax"], cand["mask"])

        sel = cand["idx"]
        valid_rows = sel >= 0
        used_in_cand = int(alloc_cand.sum())

        if valid_rows.any():
            ar = alloc_cand[valid_rows]
            work_mask = ar > 0
            if work_mask.any():
                rows = sel[valid_rows][work_mask]
                delta = ar[work_mask].astype(np.int64, copy=False)
                rows = rows[(rows >= 0) & (rows < self._arr_rem.shape[0])]
                if rows.size > 0:
                    self._arr_rem[rows] = np.maximum(0, self._arr_rem[rows] - delta[:rows.size])

        # ---- Stage 2: spillover to non-candidates ----
        remK = max(0, K_eff - used_in_cand)
        if self.fill_non_candidates and remK > 0:
            # eligible
            eligible_all = (self._arr_delay <= 0) & (self._arr_rem > 0)
            extra_mask = np.array(eligible_all, copy=True)

            # exclude current candidates 
            if valid_rows.any():
                in_cand = sel[valid_rows]
                extra_mask[in_cand] = False

            extra_idx = np.nonzero(extra_mask)[0]
            if extra_idx.size > 0:
                # SJF order on remaining man-days
                rem_extra = self._arr_rem[extra_idx]
                order = np.argsort(rem_extra, kind="stable")
                extra_idx = extra_idx[order]

                # today's per-task cap = min(cmax, remaining), avoid overshoot
                cap_extra = np.minimum(
                    self._arr_cmax[extra_idx].astype(np.int32, copy=False),
                    self._arr_rem[extra_idx].astype(np.int32, copy=False)
                )

                N = int(extra_idx.size)
                if N > 0 and remK > 0:
                    base = int(remK // N)
                    if base > 0:
                        give = np.minimum(cap_extra, base).astype(np.int32, copy=False)
                        self._arr_rem[extra_idx] = np.maximum(0, self._arr_rem[extra_idx] - give)
                        remK -= int(give.sum())

                    if remK > 0:
                        # residual +1 sweep over those that still have room (SJF front)
                        room = cap_extra - (np.minimum(cap_extra, base) if base > 0 else 0)
                        able = np.nonzero(room > 0)[0]
                        if able.size > 0:
                            k = min(remK, able.size)
                            pick = able[:k]
                            self._arr_rem[extra_idx[pick]] = np.maximum(
                                0, self._arr_rem[extra_idx[pick]] - 1
                            )
                            remK -= k

        # ---- Global delay ----
        np.subtract(self._arr_delay, 1, out=self._arr_delay, casting="unsafe") # Decrease the delay array self._arr_delay for all tasks by 1 in-place. Tasks that are not yet eligible
                                                                                # (i.e., with positive delay) have their delay reduced by 1, moving them closer to becoming eligible for work.
        np.maximum(self._arr_delay, 0, out=self._arr_delay)

        # ---- After-state stats & reward ----
        remain_after = float(self._arr_rem.sum())
        completion_md_after = 1.0 - (remain_after / (total_work + 1e-8))

        completed_after = int(np.sum(self._arr_rem <= 0))
        completion_hh_after = completed_after / max(1, total_houses)

        reward = (completion_hh_after - completion_hh_before) - 0.1


        # ---- Termination----
        self.day += 1
        idle_final = int(remK) 
        self.idle_history.append(idle_final)

        terminated = (completed_after >= total_houses)
        truncated = (self.day >= self.max_steps) and not terminated
        self.done_flag = bool(terminated or truncated)

        # ---- Refresh candidates for next day; build info ----
        self._refresh_candidates()
        info = {
            "idle_workers": idle_final,
            "allocated_workers": int(K_eff - idle_final),
            "candidates_available": int(np.sum(cand["mask"] > 0)),
            "completion": float(completion_hh_after),       # household completion
            "completion_man_days": float(completion_md_after),
            "day": int(self.day),
        }

        # ---- Throttled logging ----
        self._global_step += 1
        if (self.writer is not None) and (self.tb_every > 0) and (self._global_step % self.tb_every == 0):
            self.writer.add_scalar("env/day", self.day, self._global_step)
            self.writer.add_scalar("env/completion_household", completion_hh_after, self._global_step)
            self.writer.add_scalar("env/completion_man_days", completion_md_after, self._global_step)
            self.writer.add_scalar("env/idle_workers", idle_final, self._global_step)
            self.writer.add_scalar("env/backlog_sum", remain_after, self._global_step)

        if (self.print_every > 0) and (self._global_step % self.print_every == 0):
            print(f"[{self.region_name}] step={self._global_step} day={self.day} "
                  f"comp_hh={completion_hh_after:.3f} comp_md={completion_md_after:.3f} "
                  f"avail_cand={int(np.sum(cand['mask']>0))} idle={idle_final}")

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    # ------------- candidates + obs -------------
    def _refresh_candidates(self):
        """
        Build candidate buffer arrays with minimal pandas overhead.

        Eligible = (delay<=0) & (remaining>0).
        Candidate list is composed of SJF head + LJF head + random fill, then padded to M.
        """
        eligible_mask = (self._arr_delay <= 0) & (self._arr_rem > 0)
        eligible = np.nonzero(eligible_mask)[0]

        M = self.M
        sjf_n = int(0.4 * M)
        ljf_n = int(0.4 * M)
        idx_list: list[int] = []

        if eligible.size > 0:
            rem = self._arr_rem[eligible]
            # SJF: small remaining first
            order_sjf = eligible[np.argsort(rem, kind="stable")]
            idx_list.extend(order_sjf[:min(sjf_n, order_sjf.size)].tolist())
            # LJF: large remaining first
            order_ljf = eligible[np.argsort(-rem, kind="stable")]
            idx_list.extend(order_ljf[:min(ljf_n, order_ljf.size)].tolist())

            need = M - len(idx_list)
            if need > 0:
                replace_flag = eligible.size < need
                rnd = self.rng.choice(eligible, size=need, replace=replace_flag)
                idx_list.extend(rnd.tolist())

        # pad with -1
        if len(idx_list) < M:
            idx_list.extend([-1] * (M - len(idx_list)))
        idx = np.asarray(idx_list[:M], dtype=np.int32)

        # assemble candidate feature arrays (all vectorized)
        remain = np.zeros(M, dtype=np.float32)
        delay  = np.zeros(M, dtype=np.float32)
        dmg    = np.zeros(M, dtype=np.float32)
        cmax   = np.zeros(M, dtype=np.float32)

        valid_rows = idx >= 0
        if valid_rows.any():
            sel = idx[valid_rows]
            remain[valid_rows] = self._arr_rem[sel].astype(np.float32, copy=False)
            delay[valid_rows]  = self._arr_delay[sel].astype(np.float32, copy=False)
            dmg[valid_rows]    = self._arr_dmg[sel].astype(np.float32, copy=False)
            cmax[valid_rows]   = self._arr_cmax[sel].astype(np.float32, copy=False)

        mask = ((delay <= 0) & (remain > 0) & valid_rows).astype(np.float32, copy=False)

        self._last_candidates = {
            "idx": idx,
            "remain": remain,
            "delay": delay,
            "dmg": dmg,
            "cmax": cmax,
            "mask": mask
        }

    def _get_obs(self) -> np.ndarray:
        # use cached arrays for speed
        remain = self._arr_rem
        dmg    = self._arr_dmg

        total_work = self._total_work_scalar
        backlog_cnt = int(np.sum(remain > 0))
        remain_sum  = float(remain.sum())
        lvl1 = float(remain[dmg == 1].sum())
        lvl2 = float(remain[dmg == 2].sum())

        g = np.array([
            self.day / max(1.0, self.max_steps),
            float(self.K),
            float(backlog_cnt),
            remain_sum / (total_work + 1e-8),
            lvl1 / (total_work + 1e-8),
            lvl2 / (total_work + 1e-8),
        ], dtype=np.float32)

        c = self._last_candidates
        cand = np.stack([c["remain"], c["delay"], c["dmg"], c["cmax"]], axis=1).astype(np.float32, copy=False)
        return np.concatenate([g, cand.reshape(-1)], axis=0).astype(np.float32, copy=False)

    def last_candidate_view(self) -> Dict[str, np.ndarray]:
        return self._last_candidates
