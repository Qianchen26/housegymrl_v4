# PyRebuild SAC.

---

## 1) Dependencies

Install Python 3.10+ and:

- `numpy`, `pandas`, `matplotlib`  
- `gymnasium>=0.29`  
- `stable-baselines3>=2.3.0`  
- `torch` (CPU is fine; Apple Silicon can use MPS if available)  
- `tqdm`  
- `tensorboard`

Repo files:

- `housegymrl.py` — env  
- `baseline.py` — SJF/LJF/Random heuristic allocators  
- `evaluate.py` — daily alignment, metrics, plotting, multi-seed   
- `multiseed_eval.ipynb` — multi-seed evaluation notebook  


---

## 2) Files & What They Do

### `housegymrl.py`
- **Purpose**: Environment for **continuous workforce allocation** with a candidate buffer of size **M**.  
- **State (observation)**: flat vector `[global ; M × candidate]`  
  - **Global (6 dims):**  
    1. `day / max_steps` — normalized episode progress  
    2. `K` — static crew capacity (before day-level ramping)  
    3. `backlog_cnt` — number of tasks with `remaining > 0`  
    4. `remain_sum / total_work` — fraction of remaining total work  
    5. `lvl1 / total_work` — fraction of remaining work in level 1  
    6. `lvl2 / total_work` — fraction of remaining work in level 2  
  - **Candidate (4 dims per candidate × M):**  
    1. `remain` — remaining work for that house  
    2. `delay` — days until actionable  
    3. `dmg` — damage level (categorical as int)  
    4. `cmax` — per-house daily crew cap  
  - **Dimension:** `obs_dim = 6 + 4 × M`  
- **Action**: continuous scores in `[0,1]^M` mapped to **integer crews** with per-house `cmax` and global capacity constraint.  
- **Reward**: daily **functionality/completion gain** (larger/faster recovery ⇒ larger AUC).  
- **Termination**: all tasks done **or** `max_days` reached.  
- **Helper**: `_scores_to_alloc(scores, capacity, cmax)` encapsulates score→integer allocation mapping.  

### `baseline.py`
- Heuristic policies: **SJF / LJF / Random**.  
- Core function:  
  `allocate_baseline(candidates, remaining_work, crew_capacity, cmax=1, policy="SJF", rng=None)`  
  Guarantees `sum(alloc) ≤ crew_capacity` and skips padded candidates (`mask==0`).  

### `evaluate.py`
- Evaluates a **trained run dir** (SAC) and baselines; aligns simulated daily curves to observed; computes metrics; saves plots/CSVs.  
- **Entrypoints:**  
  - `eval_run_dir(run_dir, seeds, t_max_eval=1200, alpha=1.0, save_plots=True, save_csv=True)`  
  - `eval_baselines_for_run(run_dir, seeds, ...)`  
- **Internals:**  
  - `_resolve_artifacts(run_dir)` finds checkpoints/logs/observed data  
  - `_rollout_and_score(...)` runs per-seed episodes and metrics  
- **Evaluation matrix (CSV columns):**  
  `region, strategy, days_obs, rmse, auc@200, auc@300, t80, t90, t95, final_completion, makespan`  

### `multiseed_eval.ipynb`
- Multi-seed evaluation + plots per region and overall. Useful for variance diagnostics.  

**Run order:**

- **A) Evaluate existing model**  
  1. Set `RUN_DIR` in the notebook (or in `main.py`) to your trained run directory.  
  2. Run evaluation → curves and metrics land under that run.  

- **B) Full pipeline (train + eval)**  
  1. Train SAC to produce a run dir (with checkpoints).  
  2. Evaluate via the notebook or `main.py`.  

---

## 3) State–Action–Reward Summary

| Element | Definition | Shape / Range | Notes |
|---|---|---|---|
| **State (observation)** | Concatenation of **6 global** and **M × 4 candidate** features. | `(6 + 4·M,)` | Global: `day/max_steps, K, backlog_cnt, remain_sum/total_work, lvl1 share, lvl2 share`. Candidate: `remain, delay, dmg, cmax`. |
| **Action** | Continuous scores per candidate, mapped to integer crews. | `[0,1]^M` → integer `(M,)` | Mapping respects per-house `cmax` and total ≤ `K_t` (today’s available crews). |
| **Reward** | Daily functionality/completion gain. | scalar | Aligned with resilience AUC: faster recovery ⇒ larger area under the curve. |
| **Termination** | All tasks complete or `max_days` reached. | — | Distinguishes success vs timeout. |

---

## 4) Scenario Settings

### Mobilization Toggle

| Setting | Meaning | Daily crews `K_t` | Typical Use |
|---|---|---|---|
| `USE_RAMP = True` | Realistic mobilization (delayed start + ramp + ceiling). | `K_t = ⌊K_static · k_ramp(day)⌋`, `k_ramp ∈ [0, cap]`. | Realistic runs (slower start, gradual scale-up). |
| `USE_RAMP = False` | No mobilization. Full crews from day 0. | `K_t = K_static`. | Idealized upper bound (fastest possible build-out). |

### Crew

- `K_static = cfg["num_contractors"]` 


### Day-Level Supply (time variation)

- `k_ramp = linear_ramp(warmup_days, rise_days, cap)`  
- `warmup_days` — zero output before mobilization kicks in  
- `rise_days` — days to reach the ceiling  
- `cap` — long-run ceiling (≤1.0)  
- Optional `alpha` — global scenario scalar  

**Formula:**

```
K_t = floor(K_static * k_ramp(day))   # when USE_RAMP=True
K_t = K_static                        # when USE_RAMP=False
```

### Step-Level Allocation

- `M` — candidate buffer size (action dimension)  
- `cmax` — per-house cap (by level, defaults `{0:2, 1:4, 2:6}`)  
- `mask` — filter for valid vs padded candidates  
- Optional spillover — redirect leftover crews to non-buffer tasks  

### Synthetic Data Generation

- `H = sum(damage_dist)` — total houses  
- `props = Dirichlet([1.2, 1.0, 0.9])` — level proportions  
- `pert_ranges` — man-day PERT params by level  
- `delay_ranges` — reporting delays by level  
- `cmax_by_lvl` — per-house daily caps  
- `clusters = max(8, int(H**0.5))` — spatial clusters  
- `seed` — RNG seed  

### Evaluation

- `t_max_eval` — horizon for metrics/plots  
- `seeds` — evaluate with multiple seeds (≥3)  
- Baselines — SJF/LJF/Random  

---

## 5) Evaluation Matrix

Each row = **region × strategy** (saved in `results/metrics_eval.csv`):


- `rmse`  
- `auc@200`, `auc@300`  
- `t80`, `t90`, `t95`  
- `final_completion`  
- `makespan`

---

## 6) Tuning Guide

- **Make it harder:** increase `warmup_days`, `rise_days`; lower `cap`; reduce `cmax`; reduce `M`; increase heterogeneity.  
- **Fast smoke tests:** `USE_RAMP=False`, small `M`, short `max_days`.  
- **SAC essentials:**  
  - `learning_rate`: 1e-4–3e-4  
  - `batch_size`: 256–512  
  - `gamma`: 0.99 (0.995 for long horizons)  
  - `tau`: 0.005–0.02  
  - `ent_coef`: `"auto"` (target entropy ≈ -action_dim)  
  - `train_steps`: ~1e5 for smoke tests, scale up later  
- **Metrics:** aim for ↑ AUC@T, ↓ t90; check observed vs simulated curves.
# housegymrl_v4
