# End-to-end simulation for abstract-ready results
# - Calibrates baseline prevalence to an external target (e.g., 30%)
# - Runs baseline, screening, adherence, combined scenarios
# - Computes mean and 95% CIs across iterations
# - Prints clean summary lines you can paste into an abstract
# - Saves CSVs and PNG figures in OUTDIR
#
# Dependencies: numpy, pandas, matplotlib (installed by default in most environments)

import os, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Parameters you may edit
# -----------------------------
SEED = 42
N = 100_000                    # cohort size
ITERS = 300                    # Monte Carlo iterations per scenario
OUTDIR = "abrcms_outputs"      # output folder
TARGET_PREV = 0.30             # target baseline prevalence after calibration
USE_SUBGROUP_LOOKUP = False    # if True, assigns HTN by subgroup rather than logistic model

# Screening and adherence levers (relative diagnosis boost, absolute control boost)
LEVER_SCREENING = 0.20         # +20% relative diagnosis
LEVER_ADHERENCE = 0.15         # +15 percentage points control among treated

# -----------------------------
# Synthetic cohort generator
# -----------------------------
RACES = ["White", "Black", "Hispanic", "Other"]
SEXES = ["Male", "Female"]

def set_seed(seed=SEED):
    np.random.seed(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def synthesize_cohort(N: int) -> pd.DataFrame:
    age = np.clip(np.random.normal(47, 16, N), 18, 90).round().astype(int)
    sex = np.random.choice(SEXES, size=N, p=[0.49, 0.51])
    race = np.random.choice(RACES, size=N, p=[0.60, 0.13, 0.18, 0.09])
    bmi = np.clip(np.random.normal(28, 6, N), 15, 60)
    smoker = np.random.binomial(1, 0.15, N)
    diabetes = np.random.binomial(1, 0.11, N)
    return pd.DataFrame({"age": age, "sex": sex, "race": race, "bmi": bmi, "smoker": smoker, "diabetes": diabetes})

# Optional subgroup prevalence lookup (edit if you want to enforce exogenous baselines)
PREV_BY_RACE = {"White": 0.30, "Black": 0.40, "Hispanic": 0.28, "Other": 0.30}

def assign_htn_from_lookup(df: pd.DataFrame) -> np.ndarray:
    p = df["race"].map(PREV_BY_RACE).values
    return (np.random.rand(len(df)) < p).astype(int)

# -----------------------------
# Logistic baseline for HTN with calibration
# -----------------------------
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def baseline_logit(df: pd.DataFrame, intercept: float) -> np.ndarray:
    age_c = (df["age"] - 45) / 10.0
    bmi_c = (df["bmi"] - 27) / 5.0
    smoker = df["smoker"].values
    diabetes = df["diabetes"].values
    race_lo = np.zeros(len(df))
    race_lo[df["race"] == "Black"] = 0.35
    race_lo[df["race"] == "Hispanic"] = -0.20
    race_lo[df["race"] == "Other"] = -0.05
    sex_lo = np.where(df["sex"] == "Male", 0.10, 0.0)
    log_odds = intercept + 0.45*age_c + 0.35*bmi_c + 0.25*smoker + 0.60*diabetes + race_lo + sex_lo
    return sigmoid(log_odds)

def calibrate_intercept(df: pd.DataFrame, target_prev=TARGET_PREV, tol=1e-5, max_iter=80):
    lo, hi = -5.0, 5.0
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        p = baseline_logit(df, mid).mean()
        if abs(p - target_prev) < tol:
            return mid
        if p > target_prev: hi = mid
        else: lo = mid
    return 0.5*(lo+hi)

# -----------------------------
# Care pathway simulation
# -----------------------------
def simulate_once(df: pd.DataFrame,
                  intercept: float = None,
                  diag_rel_increase: float = 0.0,
                  adher_increase: float = 0.0,
                  seed: int = None,
                  use_subgroup_lookup: bool = False) -> dict:
    if seed is not None:
        np.random.seed(seed)
    N = len(df)

    if use_subgroup_lookup:
        htn = assign_htn_from_lookup(df)
    else:
        p_htn = baseline_logit(df, intercept if intercept is not None else -0.08)
        htn = np.random.binomial(1, p_htn, N)

    # Diagnosis among hypertensive
    base_diag = 0.70
    diag_p = base_diag + 0.05*(df["age"] > 60).astype(int) - 0.08*(df["age"] < 35).astype(int) + 0.05*df["diabetes"].values
    diag_p = np.clip(diag_p * (1 + diag_rel_increase), 0, 1)
    diagnosed = (np.random.rand(N) < diag_p) & (htn == 1)

    # Treatment among diagnosed
    treat_p = 0.80 - 0.05*(df["smoker"].values) + 0.03*df["diabetes"].values
    treated = (np.random.rand(N) < np.clip(treat_p, 0, 1)) & diagnosed

    # Control among treated; adherence lever boosts control
    control_p = np.clip(0.46 + adher_increase, 0, 1)
    controlled = (np.random.rand(N) < control_p) & treated

    return {
        "prevalence": (htn.mean()),
        "undiagnosed": (((htn == 1) & (~diagnosed)).mean()),
        "uncontrolled": (((htn == 1) & diagnosed & (~controlled)).mean()),
        "treated_rate": (treated.mean()),
        "controlled_rate_treated": (controlled.sum() / max(treated.sum(), 1))
    }

def run_scenario(df: pd.DataFrame, iters: int, scenario: str, intercept: float, use_subgroup_lookup: bool):
    if scenario == "baseline":
        diag_inc, adh_inc = 0.0, 0.0
    elif scenario == "screening":
        diag_inc, adh_inc = LEVER_SCREENING, 0.0
    elif scenario == "adherence":
        diag_inc, adh_inc = 0.0, LEVER_ADHERENCE
    elif scenario == "combined":
        diag_inc, adh_inc = LEVER_SCREENING, LEVER_ADHERENCE
    else:
        raise ValueError("Unknown scenario")

    rows = []
    for t in range(iters):
        rows.append(
            simulate_once(df,
                          intercept=intercept,
                          diag_rel_increase=diag_inc,
                          adher_increase=adh_inc,
                          seed=10_000 + t,
                          use_subgroup_lookup=use_subgroup_lookup)
        )
    return pd.DataFrame(rows)

# -----------------------------
# Helpers for CI and plotting
# -----------------------------
def mean_ci(series: pd.Series, alpha=0.05):
    m = series.mean()
    s = series.std(ddof=1)
    n = series.shape[0]
    if n < 2 or s == 0:
        return m, m, m
    # Normal approx 95 percent CI
    z = 1.96
    h = z * s / np.sqrt(n)
    return m, m - h, m + h

def save_bar(figpath, labels, values, ylabel, title):
    plt.figure(figsize=(7, 4))
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=15)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(figpath, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()

# -----------------------------
# Run everything
# -----------------------------
set_seed(SEED)
ensure_dir(OUTDIR)
cohort = synthesize_cohort(N)

if USE_SUBGROUP_LOOKUP:
    CAL_INTERCEPT = None
else:
    CAL_INTERCEPT = calibrate_intercept(cohort, target_prev=TARGET_PREV)

scenarios = ["baseline", "screening", "adherence", "combined"]
results = {}
t0 = time.time()
for sc in scenarios:
    df_res = run_scenario(cohort, ITERS, sc, CAL_INTERCEPT, USE_SUBGROUP_LOOKUP)
    df_res["scenario"] = sc
    results[sc] = df_res
runtime_total = time.time() - t0

# -----------------------------
# Aggregate and export
# -----------------------------
agg_rows = []
for sc, df_res in results.items():
    m_prev, lo_prev, hi_prev = mean_ci(df_res["prevalence"])
    m_undx, lo_undx, hi_undx = mean_ci(df_res["undiagnosed"])
    m_unctl, lo_unctl, hi_unctl = mean_ci(df_res["uncontrolled"])
    m_trt, lo_trt, hi_trt = mean_ci(df_res["treated_rate"])
    m_ctrl_t, lo_ctrl_t, hi_ctrl_t = mean_ci(df_res["controlled_rate_treated"])
    agg_rows.append({
        "scenario": sc,
        "prevalence_mean": m_prev, "prevalence_lo": lo_prev, "prevalence_hi": hi_prev,
        "undiagnosed_mean": m_undx, "undiagnosed_lo": lo_undx, "undiagnosed_hi": hi_undx,
        "uncontrolled_mean": m_unctl, "uncontrolled_lo": lo_unctl, "uncontrolled_hi": hi_unctl,
        "treated_rate_mean": m_trt, "treated_rate_lo": lo_trt, "treated_rate_hi": hi_trt,
        "controlled_rate_treated_mean": m_ctrl_t, "controlled_rate_treated_lo": lo_ctrl_t, "controlled_rate_treated_hi": hi_ctrl_t
    })
agg = pd.DataFrame(agg_rows)
agg.to_csv(os.path.join(OUTDIR, "summary_with_ci.csv"), index=False)

# Save per scenario runs and simple summaries
for sc, df_res in results.items():
    df_res.to_csv(os.path.join(OUTDIR, f"runs_{sc}.csv"), index=False)
    df_res.mean(numeric_only=True).to_csv(os.path.join(OUTDIR, f"summary_{sc}.csv"))

# Build a compact table for plotting
plot_tbl = pd.DataFrame({
    "scenario": agg["scenario"],
    "prevalence": agg["prevalence_mean"],
    "undiagnosed": agg["undiagnosed_mean"],
    "uncontrolled": agg["uncontrolled_mean"],
    "treated_rate": agg["treated_rate_mean"],
    "controlled_rate_treated": agg["controlled_rate_treated_mean"]
})

# -----------------------------
# Figures
# -----------------------------
save_bar(os.path.join(OUTDIR, "fig_prevalence.png"),
         plot_tbl["scenario"].tolist(),
         plot_tbl["prevalence"].tolist(),
         ylabel="Hypertension prevalence (fraction)",
         title="Prevalence by Scenario")

save_bar(os.path.join(OUTDIR, "fig_undiagnosed.png"),
         plot_tbl["scenario"].tolist(),
         plot_tbl["undiagnosed"].tolist(),
         ylabel="Undiagnosed hypertension (fraction of population)",
         title="Undiagnosed by Scenario")

save_bar(os.path.join(OUTDIR, "fig_uncontrolled.png"),
         plot_tbl["scenario"].tolist(),
         plot_tbl["uncontrolled"].tolist(),
         ylabel="Uncontrolled hypertension (fraction of population)",
         title="Uncontrolled by Scenario")

save_bar(os.path.join(OUTDIR, "fig_controlled_treated.png"),
         plot_tbl["scenario"].tolist(),
         plot_tbl["controlled_rate_treated"].tolist(),
         ylabel="Control among treated (fraction)",
         title="Control Rate Among Treated")

# -----------------------------
# Clean print for abstract
# -----------------------------
def pct(x): return f"{100*x:.1f}%"

baseline = agg[agg["scenario"]=="baseline"].iloc[0]
screen   = agg[agg["scenario"]=="screening"].iloc[0]
adh      = agg[agg["scenario"]=="adherence"].iloc[0]
combo    = agg[agg["scenario"]=="combined"].iloc[0]

print("\nABSTRACT NUMBERS")
print("----------------")
print(f"Cohort N = {N}, iterations = {ITERS}, total wall time ~ {runtime_total:.1f} s")
if CAL_INTERCEPT is not None:
    print(f"Calibrated baseline prevalence target = {TARGET_PREV:.2f} (intercept = {CAL_INTERCEPT:.3f})")
else:
    print("Using subgroup prevalence lookup (no logistic calibration)")

print("\nBaseline:")
print(f"  Prevalence = {pct(baseline['prevalence_mean'])}  (95% CI {pct(baseline['prevalence_lo'])} to {pct(baseline['prevalence_hi'])})")
print(f"  Undiagnosed = {pct(baseline['undiagnosed_mean'])}")
print(f"  Uncontrolled = {pct(baseline['uncontrolled_mean'])}")
print(f"  Treated rate = {pct(baseline['treated_rate_mean'])}")
print(f"  Control among treated = {pct(baseline['controlled_rate_treated_mean'])}")

print("\nScreening (diagnosis +20% relative):")
print(f"  Undiagnosed = {pct(screen['undiagnosed_mean'])}  (vs {pct(baseline['undiagnosed_mean'])} baseline)")
print(f"  Treated rate = {pct(screen['treated_rate_mean'])}  (vs {pct(baseline['treated_rate_mean'])} baseline)")

print("\nAdherence (+15 points control among treated):")
print(f"  Control among treated = {pct(adh['controlled_rate_treated_mean'])}  (vs {pct(baseline['controlled_rate_treated_mean'])} baseline)")
print(f"  Uncontrolled = {pct(adh['uncontrolled_mean'])}  (vs {pct(baseline['uncontrolled_mean'])} baseline)")

print("\nCombined (screening + adherence):")
print(f"  Undiagnosed = {pct(combo['undiagnosed_mean'])}")
print(f"  Control among treated = {pct(combo['controlled_rate_treated_mean'])}")
print(f"  Uncontrolled = {pct(combo['uncontrolled_mean'])}")

# Save metadata
with open(os.path.join(OUTDIR, "run_metadata.json"), "w") as f:
    json.dump({
        "seed": SEED,
        "N": N,
        "iters": ITERS,
        "target_prevalence": TARGET_PREV,
        "use_subgroup_lookup": USE_SUBGROUP_LOOKUP,
        "levers": {"screening_rel_increase": LEVER_SCREENING, "adherence_abs_increase": LEVER_ADHERENCE},
        "runtime_seconds": runtime_total
    }, f, indent=2)

print(f"\nFiles written to: {OUTDIR}")
