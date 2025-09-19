from __future__ import annotations
from pathlib import Path
import pickle, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

from stable_baselines3 import SAC
from housegymrl import HousegymRLENV
from baseline import make_baseline_scores

# ------------------- Config -------------------
# Change this to your local path if needed
DATA_PATH = "/Users/qianchenyu/Documents/HouseGymRL/HouseGymEnv/data/lombok_data.pkl" # i uploaded the data to github

REGION_CONFIG = {
    'Mataram': {
        'damage_dist': [9500, 3672, 1345],  # [minor, moderate, major] 
        'num_contractors': 9917 
    },
    'West Lombok': {
        'damage_dist': [45218, 13556, 14069],  
        'num_contractors': 45208  
    },
    'North Lombok': {
        'damage_dist': [8889, 4772, 42049],  
        'num_contractors': 22996  
    },
    'Central Lombok': {
        'damage_dist': [16639, 3096, 4483],  
        'num_contractors': 15048 
    },
    'East Lombok': {
        'damage_dist': [12209, 4657, 10104],  
        'num_contractors': 15404  
    },
    'West Sumbawa': {
        'damage_dist': [13078, 3803, 1283],  
        'num_contractors': 10200  
    },
    'Sumbawa': {
        'damage_dist': [9652, 2756, 1374],  
        'num_contractors': 10360  
    }
}
M_CANDIDATES = 96
MAX_STEPS    = 1500

RESULTS_DIR = Path("results")
(RESULTS_DIR / "curves").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "figs").mkdir(parents=True, exist_ok=True)

# Mapping names
REGION_ALIASES = {
    "Mataram":        "mataram",
    "West Lombok":    "lombokbarat",
    "Central Lombok": "lomboktengah",
    "North Lombok":   "lombokutara",
    "East Lombok":    "lomboktimur",
    "West Sumbawa":   "sumbawabarat",
    "Sumbawa":        "sumbawa",
}

def linear_ramp(warmup_days: int = 150, rise_days: int = 180, cap: float = 1.0): # simulate a delay in capacity ramp-up: not completely ready at day 0. using 150 becasue checked
                                                                                # data and 150 is approx. the first day when most regions start to have non-zeroo completion
    warmup_days = int(warmup_days)
    rise_days   = max(1, int(rise_days))
    cap = float(max(0.0, min(1.0, cap)))
    def _r(d: int) -> float:
        if d < warmup_days: return 0.0
        return min(cap, (d - warmup_days) / float(rise_days))
    return _r

def infer_region_warmup_days(
    observed_map: dict[str, "pd.Series"],
    *,
    frac_threshold: float = 0.01,   # "non-zero" threshold (e.g., 1% completion)
    sustain_days: int = 7,          # how many days we require it to stay above the threshold
    clip_max: int | None = None     # optionally clip warmup to avoid extreme values
) -> dict[str, int]:
   
    warmups: dict[str, int] = {}
    for reg, s in observed_map.items():
        y = np.asarray(s.to_numpy(), dtype=float)
        D = len(y)
        w = max(1, min(int(sustain_days), D))

        idx = None
        # sustained threshold crossing
        for i in range(0, D - w + 1):
            if (y[i] >= frac_threshold) and np.all(y[i:i+w] >= frac_threshold):
                idx = i
                break

        # fallback: first strictly >0
        if idx is None:
            nz = np.where(y > 0.0)[0]
            idx = int(nz[0]) if nz.size > 0 else 0

        if clip_max is not None:
            idx = int(min(idx, int(clip_max)))

        warmups[reg] = int(max(0, idx))
    return warmups

def build_region_ramps(
    observed_map: dict[str, "pd.Series"],
    *,
    rise_days: int = 120,   # how fast to climb to full capacity after the warmup
    cap: float = 1.0,       # no reserve requested (keep 1.0)
    frac_threshold: float = 0.01,
    sustain_days: int = 7,
    clip_max: int | None = None
) -> dict[str, callable]:    # based on each region's first non-zero day
   
    warmup_by_region = infer_region_warmup_days(
        observed_map,
        frac_threshold=frac_threshold,
        sustain_days=sustain_days,
        clip_max=clip_max,
    )
    region_ramp = {
        reg: linear_ramp(warmup_days=wd, rise_days=rise_days, cap=cap)
        for reg, wd in warmup_by_region.items()
    }
    return region_ramp, warmup_by_region


# ------------------- Synthetic generator -------------------
## randomly pick a number from (10000, 100000). this covers the min and max of the real regions. and randomly split into minor/moderate/major using dirichlet distribution
## construction crew K = ratio * H, where ratio is randomly picked from (0.1, 0.25). this covers the ratio range of the real regions

_rng = np.random.default_rng()

def _sample_pert(a, m, b, n):
    lam = 4.0
    alpha = 1 + lam * (m - a) / (b - a + 1e-8)
    beta  = 1 + lam * (b - m) / (b - a + 1e-8)
    x = _rng.beta(alpha, beta, size=n)
    return a + x * (b - a)

def generate_synthetic_scenario(region_name: str, H: int, K: int, seed: int | None):
    rng = np.random.default_rng(seed)
    props = rng.dirichlet(np.array([1.2,1.0,0.9], dtype=float))
    counts = np.maximum(1, np.round(props * H).astype(int))
    counts[np.argmax(counts)] += (H - counts.sum())

    pert_ranges = {0:(8,12,18), 1:(20,30,45), 2:(60,90,140)} # by pert. construction days man_days_total
    delay_ranges= {0:(0,2,5),  1:(1,5,10),  2:(5,15,30)} # by pert. delay_days_remaining
    cmax_by_lvl = {0:2, 1:4, 2:6}
    clusters = max(8, int(H ** 0.5))

    rows = []
    for lvl, n in enumerate(counts):
        a,m,b   = pert_ranges[lvl]
        ad,md,bd= delay_ranges[lvl]
        man = _sample_pert(a,m,b,n).round().astype(int)
        dly = _sample_pert(ad,md,bd,n).round().astype(int)
        cid = rng.integers(0, clusters, size=n)
        for mdays, d, c in zip(man, dly, cid):
            rows.append((lvl, int(mdays), int(mdays), int(d), int(cmax_by_lvl[lvl]), int(c)))
    tasks = pd.DataFrame(rows, columns=[                           
        "damage_level","man_days_total","man_days_remaining",
        "delay_days_remaining","cmax_per_day","cluster_id"
    ])
    resources = {"workers": int(K), "region_name": region_name}
    info = {"region": region_name, "H": int(H), "K": int(K), "counts": counts.tolist(), "seed": seed}
    return tasks, resources, info



def make_synth_env(
    H_min: int = 10_000,
    H_max: int = 100_000,
    worker_ratio: float | tuple[float, float] = (0.1, 0.25),
    seed: int | None = None,
    verbose: bool = False,  
) -> HousegymRLENV:
 
    rng = np.random.default_rng(seed)
    H = int(rng.integers(int(H_min), int(H_max) + 1))

    
    if isinstance(worker_ratio, (tuple, list)) and len(worker_ratio) == 2:
        lo, hi = float(worker_ratio[0]), float(worker_ratio[1])
        rho = float(rng.uniform(min(lo, hi), max(lo, hi)))
    else:
        rho = float(worker_ratio)

    K = max(1, int(round(rho * H)))

    tasks_df, resources, _ = generate_synthetic_scenario(
        region_name="SYNTH", H=H, K=K, seed=seed
    )

    if verbose:
        lvl = tasks_df["damage_level"].to_numpy()
        minor    = int(np.sum(lvl == 0))
        moderate = int(np.sum(lvl == 1))
        major    = int(np.sum(lvl == 2))
        total    = minor + moderate + major
        crew     = int(resources["workers"])
        print(f"total: {total}  minor: {minor}  moderate: {moderate}  major: {major}  crew: {crew}")

    return HousegymRLENV(
        tasks_df=tasks_df,
        resources=resources,
        M=M_CANDIDATES,
        max_steps=MAX_STEPS,
        seed=seed,
        print_every=0,
        tb_every=0,
    )



def make_region_env(region_key: str, k_ramp=None) -> HousegymRLENV:
    cfg = REGION_CONFIG[region_key]
    H = int(sum(cfg["damage_dist"])) # total houses
    K = int(cfg["num_contractors"]) # total workers

    tasks_df, resources, _ = generate_synthetic_scenario(
        region_key, H, K, cfg.get("seed", None)
    )
    return HousegymRLENV(
        tasks_df=tasks_df, resources=resources, M=M_CANDIDATES,
        max_steps=MAX_STEPS, seed=cfg.get("seed", None),
        print_every=0, tb_every=0,
        k_ramp=k_ramp
    )


# avoid obs mismatch so add this function
def infer_M_from_model(model_obj) -> int:
    import housegymrl
    obs_dim = int(model_obj.observation_space.shape[0])
    M = (obs_dim - housegymrl.OBS_G) // housegymrl.OBS_F
    assert housegymrl.OBS_G + M * housegymrl.OBS_F == obs_dim, \
        f"obs_dim={obs_dim} not compatible with OBS_G={housegymrl.OBS_G}, OBS_F={housegymrl.OBS_F}"
    return int(M)

def make_region_env_with_M(region_key: str, M_override: int, k_ramp=None) -> HousegymRLENV:
    cfg = REGION_CONFIG[region_key]
    H = int(sum(cfg["damage_dist"]))
    K = int(cfg["num_contractors"])
    K = min(K, max(1, int(0.02 * H)))  # 同样的斜率修正

    tasks_df, resources, _ = generate_synthetic_scenario(
        region_key, H, K, cfg.get("seed", None)
    )
    import housegymrl
    return housegymrl.HousegymRLENV(
        tasks_df=tasks_df, resources=resources, M=M_override,
        max_steps=MAX_STEPS, seed=cfg.get("seed", None),
        print_every=0, tb_every=0, k_ramp=k_ramp
    )


# ------------------- Observed loader tailored to your DataFrame -------------------
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame: # a helper function to ensure the df has a datetime index. adding this becasue it has caused issues before

    # 1) DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date")
        else:
            raise ValueError("Observed DataFrame must have a DatetimeIndex or a 'date' column.")
    else:
        df = df.copy()

    idx = pd.to_datetime(df.index)
    try:
        idx = idx.tz_localize(None)
    except Exception:
        pass
    df.index = idx.normalize()

    # 2) duplicates
    df = df.groupby(df.index).max()

    df = df.sort_index()
    return df


def _pick_total_comp_series(df: pd.DataFrame, alias: str) -> pd.Series | None:
    cols_lower = {c.lower(): c for c in df.columns}
    key = f"{alias}_total_comp"
    if key in cols_lower:
        return pd.to_numeric(df[cols_lower[key]], errors="coerce")
    # Fallback: sum of comp_rb/rs/rr
    parts = [f"{alias}_comp_rb", f"{alias}_comp_rs", f"{alias}_comp_rr"]
    got = [cols_lower[p] for p in parts if p in cols_lower]
    if got:
        return sum(pd.to_numeric(df[g], errors="coerce") for g in got)
    return None

def _daily_align_monotone_fraction(s: pd.Series, denominator: float) -> pd.Series:
    
    s = s.copy()
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_idx)
    if pd.isna(s.iloc[0]):
        s.iloc[0] = 0.0
    s = s.ffill().fillna(0.0)
    s = s.cummax()

    denom = max(1.0, float(denominator))
    s = (s.astype(float) / denom).clip(0.0, 1.0)
    s.index = pd.RangeIndex(start=0, stop=len(s), step=1)
    s.name = "obs_completion"
    return s


def load_observed(DATA_PATH: str | Path) -> dict[str, pd.Series]:
    p = Path(DATA_PATH)
    if p.suffix == ".pkl":
        obj = pd.read_pickle(p)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, pd.DataFrame) and any(c.lower().endswith("_total_comp") for c in v.columns):
                    obj = v
                    break
        if not isinstance(obj, pd.DataFrame):
            raise ValueError(f"Expected a pickled DataFrame; got {type(obj)}")
        df = _ensure_datetime_index(obj)
    else:
        raise ValueError("DATA_PATH must be a .pkl of a DataFrame with DatetimeIndex.")

    out = {}
    for region_name in REGION_CONFIG.keys():
        alias = REGION_ALIASES.get(region_name)
        if not alias:
            continue
        s = _pick_total_comp_series(df, alias)
        if s is None:
            continue
        s.index = df.index

        target_total = _pick_target_total(df, alias)
        observed_final = float(s.dropna().iloc[-1]) if s.dropna().size > 0 else 0.0
        denominator = target_total if (target_total is not None and target_total > 0) else observed_final

        out[region_name] = _daily_align_monotone_fraction(s, denominator)
    return out


# ------------------- Metrics -------------------
def rmse(sim: np.ndarray, obs: np.ndarray) -> float:
    D = min(len(sim), len(obs))
    if D == 0: return float("nan")
    return float(np.sqrt(np.mean((sim[:D] - obs[:D]) ** 2)))

def auc_at_T(curve: np.ndarray, T: int) -> float:
    D = min(len(curve), T)
    if D <= 1: return 0.0
    x = np.arange(D)
    y = curve[:D]
    return float(np.trapz(y, x) / max(1.0, float(T)))

def t_reach(curve: np.ndarray, thr: float) -> float | float("nan"):
    idx = np.where(curve >= thr)[0]
    return float(idx[0]) if idx.size > 0 else float("nan")

def rollout(env: HousegymRLENV, model: SAC | None, policy: str, max_days: int) -> np.ndarray:
    obs, _ = env.reset()
    traj = []
    for _ in range(max_days):
        if model is not None and policy == "SAC":
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = make_baseline_scores(env, policy=policy)
        obs, r, done, trunc, info = env.step(action)
        traj.append(info["completion"])
        if done or trunc:
            break
    return np.array(traj, dtype=float)

def _pick_target_total(df: pd.DataFrame, alias: str) -> float | None:
    
    cols_lower = {c.lower(): c for c in df.columns}
    parts = [f"{alias}_target_rb", f"{alias}_target_rs", f"{alias}_target_rr"]
    got = [cols_lower[p] for p in parts if p in cols_lower]
    if not got:
        return None
    tgt = sum(pd.to_numeric(df[g], errors="coerce") for g in got)
    last = tgt.dropna().iloc[-1] if tgt.dropna().size > 0 else None
    try:
        return float(last) if last is not None and np.isfinite(last) else None
    except Exception:
        return None


# ------------------- Main -------------------
def main():
    # 1) Load observed + print summary
    observed = load_observed(DATA_PATH)
    if not observed:
        print("=== Observed summary ===")
        print("[WARN] No regions found in the provided DataFrame. "
              "Check DATA_PATH, column names (e.g., 'mataram_total_comp'), and DatetimeIndex.")
        return

    print("=== Observed summary ===")
    for reg, s in observed.items():
        print(f"[{reg}] observed days: {len(s)}, final completion: {s.iloc[-1]:.3f}")

    # 2) Try load latest SAC model (optional)
    model = None
    try:
        runs = sorted(Path("runs").glob("*/sac_model.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if runs:
            model = SAC.load(str(runs[0]))
            print(f"[Eval] Loaded SAC model: {runs[0]}")
    except Exception as e:
        print(f"[Eval] No SAC model loaded ({e}). Baselines only.")

    # 3) Per-region evaluation
    rows = []
    policies = (["SAC"] if model is not None else []) + ["SJF", "LJF", "RANDOM"]

    for reg, obs_series in observed.items():
        env = make_region_env(reg)
        obs_curve = obs_series.to_numpy()
        D_obs = len(obs_curve)

        # plot
        plt.figure(figsize=(7,4))
        plt.plot(np.arange(D_obs), obs_curve, label="Observed", linewidth=2)

        for p in policies:
            sim = rollout(env, model, policy=p, max_days=D_obs)
            # save curve
            pd.DataFrame({"day": np.arange(len(sim)), "completion": sim}).to_csv(
                RESULTS_DIR / "curves" / f"{reg}_{p}.csv", index=False
            )
            # metrics
            rows.append({
                "region": reg, "strategy": p, "days_obs": D_obs,
                "rmse": rmse(sim, obs_curve),
                "auc@200": auc_at_T(sim, 200),
                "auc@300": auc_at_T(sim, 300),
                "t80": t_reach(sim, 0.80),
                "t90": t_reach(sim, 0.90),
                "t95": t_reach(sim, 0.95),
                "final_completion": float(sim[-1]) if len(sim) > 0 else 0.0,
                "makespan": t_reach(sim, 0.99),
            })
            plt.plot(sim, label=p, alpha=0.9)

        plt.axvline(x=D_obs, linestyle="--", linewidth=1)
        plt.title(f"{reg} — Observed vs Simulated")
        plt.xlabel("Day"); plt.ylabel("Cumulative completion (0..1)")
        plt.legend(); plt.tight_layout()
        plt.savefig(RESULTS_DIR / "figs" / f"{reg}_compare.png")
        plt.close()

    # 4) Write metrics CSV (append or create)
    metrics_path = RESULTS_DIR / "metrics_eval.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False, mode=("a" if metrics_path.exists() else "w"))
    print(f"[Eval] Metrics appended to: {metrics_path}")

if __name__ == "__main__":
    main()
