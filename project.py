import sys
from pathlib import Path

# adjust the path below to point at your clone of EvaluationOverTime
repo_root = Path("/EvaluationOverTime")
sys.path.insert(0, str(repo_root))

# Cell 1: Clone & install the EvaluationOverTime package
# (skip if you already have it locally)
!git clone https://github.com/acmi-lab/EvaluationOverTime.git
!pip install -e EvaluationOverTime/src

# Cell 2: Imports & global settings
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the EOT experiment framework
from emdot.eot_experiment import EotExperiment
from emdot.models.LR import LR
from emdot.models.GBDT import GBDT

# Where to store synthetic CSVs and results
SYN_DIR    = Path("data/synthetic")
RESULTS_DIR = Path("results/synthetic_notebook")
SYN_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Number of rows per dataset (adjust to 100, 500_000, etc.)
N_ROWS = 100_000_000

# Cell 3: Synthetic data generators (now accept n_rows)

import numpy as np
import pandas as pd

# Number of rows per dataset
N_ROWS = 100

def gen_seer(n_rows=N_ROWS, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(20, 90, n_rows),
        "tumor_size": rng.uniform(0, 100, n_rows),
        "sex": rng.choice(["Male","Female"], n_rows, p=[0.3,0.7]),
        "race": rng.choice(["White","Black","Asian","Other"], n_rows, p=[0.6,0.15,0.15,0.1]),
        "stage": rng.choice(["I","II","III","IV"], n_rows, p=[0.3,0.4,0.2,0.1]),
        "diagnosis_year": rng.integers(1975, 2023, n_rows),
    })
    logit = -0.03*(df.age-50) -0.05*(df.tumor_size/10) + np.where(df.stage.isin(["I","II"]),1.0,-1.0)
    df["survival_5yr"] = rng.binomial(1, 1/(1+np.exp(-logit)))
    return df

def gen_cdc(n_rows=N_ROWS, seed=1):
    rng = np.random.default_rng(seed)
    start = np.datetime64("2020-01-01"); end = np.datetime64("2023-12-31")
    df = pd.DataFrame({
        "age": rng.integers(0,101,n_rows),
        "sex": rng.choice(["Male","Female"], n_rows, p=[0.49,0.51]),
        "race_eth": rng.choice(
            ["Hispanic","Non-Hispanic White","Non-Hispanic Black","Asian/PI","Other"],
            n_rows, p=[0.18,0.6,0.12,0.05,0.05]
        ),
        "hospitalized": rng.binomial(1,0.15,n_rows),
        "case_report_date": rng.integers(start.astype("int64"), end.astype("int64"), n_rows)
                              .astype("datetime64[D]")
    })
    base = 0.001 + 0.004*(df.age/20) + 0.1*df.hospitalized
    df["death_yn"] = rng.binomial(1, np.clip(base,0,1))
    return df

def gen_swpa(n_rows=N_ROWS, seed=2):
    rng = np.random.default_rng(seed)
    start = np.datetime64("2020-03-01"); end = np.datetime64("2022-06-30")
    df = pd.DataFrame({
        "age": rng.integers(20,90,n_rows),
        "sex": rng.choice(["M","F"], n_rows, p=[0.48,0.52]),
        "comorbidity_count": rng.integers(0,6,n_rows),
        "WBC": rng.normal(7,3,n_rows).clip(1,30),
        "CRP": rng.normal(50,20,n_rows).clip(0,200),
        "admit_date": rng.integers(start.astype("int64"), end.astype("int64"), n_rows)
                         .astype("datetime64[D]")
    })
    score = 0.02*(df.age-50) + 0.1*df.comorbidity_count + 0.005*df.CRP
    df["outcome_90d"] = rng.binomial(1, 1/(1+np.exp(-score/10)))
    return df

def gen_mimic_iv(n_rows=N_ROWS, seed=3):
    rng = np.random.default_rng(seed)
    start = np.datetime64("2009-01-01"); end = np.datetime64("2020-12-31")
    df = pd.DataFrame({
        "age": rng.integers(16,90,n_rows),
        "gender": rng.choice(["M","F"], n_rows, p=[0.55,0.45]),
        "heart_rate": rng.normal(80,20,n_rows).clip(30,200),
        "sys_bp": rng.normal(120,25,n_rows).clip(40,250),
        "dia_bp": rng.normal(70,15,n_rows).clip(20,150),
        "resp_rate": rng.normal(18,5,n_rows).clip(5,60),
        "lab_glucose": rng.normal(110,30,n_rows).clip(40,400),
        "lab_creatinine": rng.normal(1.0,0.5,n_rows).clip(0.2,10),
        "chart_time": rng.integers(start.astype("int64"), end.astype("int64"), n_rows)
                         .astype("datetime64[s]")
    })
    risk = 0.03*(df.age-60) + 0.01*(df.heart_rate-80) + 0.02*(df.lab_creatinine-1)
    df["icu_mortality"] = rng.binomial(1, 1/(1+np.exp(-risk/10)))
    return df

def gen_optn(n_rows=N_ROWS, seed=4):
    rng = np.random.default_rng(seed)
    start = np.datetime64("2005-01-01"); end = np.datetime64("2017-12-31")
    df = pd.DataFrame({
        "recipient_age": rng.integers(18,75,n_rows),
        "recipient_meld": rng.integers(6,41,n_rows),
        "donor_age": rng.integers(18,75,n_rows),
        "cold_ischemia_time": rng.normal(8,2,n_rows).clip(2,24),
        "transplant_date": rng.integers(start.astype("int64"), end.astype("int64"), n_rows)
                              .astype("datetime64[D]")
    })
    risk = 0.05*(df.recipient_meld-15) + 0.01*(df.cold_ischemia_time-8)
    df["mortality_180d"] = rng.binomial(1, 1/(1+np.exp(-risk/10)))
    return df

def gen_mimic_cxr(n_rows=N_ROWS, seed=5):
    rng = np.random.default_rng(seed)
    start = np.datetime64("2010-01-01"); end = np.datetime64("2018-12-31")
    labels = [
        "Cardiomegaly","Edema","Consolidation","Pneumonia","Atelectasis",
        "Pleural_Effusion","Pneumothorax","Support_Devices","Fracture",
        "Emphysema","Fibrosis","Nodule_Mass","Enlarged_Cardiomediastinum","No_Finding"
    ]
    df = pd.DataFrame({
        "patient_age": rng.integers(0,100,n_rows),
        "patient_sex": rng.choice(["M","F"], n_rows),
        "view_position": rng.choice(["PA","AP","Lateral"], n_rows, p=[0.6,0.35,0.05]),
        "study_date": rng.integers(start.astype("int64"), end.astype("int64"), n_rows)
                          .astype("datetime64[D]")
    })
    for lab in labels:
        p = 0.05 if lab!="No_Finding" else 0.7
        df[lab] = rng.binomial(1, p, n_rows)
    return df

# Helper to enforce two periods and balanced labels
def make_two_periods(df, tcol, label, seed):
    rng = np.random.default_rng(seed)
    periods = np.tile([0,1], int(np.ceil(N_ROWS/2)))[:N_ROWS]
    rng.shuffle(periods)
    df[tcol] = periods
    labels = np.tile([0,1], int(np.ceil(N_ROWS/2)))[:N_ROWS]
    rng.shuffle(labels)
    df[label] = labels
    return df

    # Cell 4: Generate & Save Synthetic CSVs

from pathlib import Path

# Directory for synthetic data
SYN_DIR = Path("data/synthetic")
SYN_DIR.mkdir(parents=True, exist_ok=True)

# Map of generators: fn, timestamp-col, label-col, seed
gens = {
    "SEER":      (gen_seer,      "diagnosis_year",  "survival_5yr",    0),
    "CDC":       (gen_cdc,       "case_report_date","death_yn",        1),
    "SWPA":      (gen_swpa,      "admit_date",      "outcome_90d",     2),
    "MIMIC_IV":  (gen_mimic_iv,  "chart_time",      "icu_mortality",   3),
    "OPTN":      (gen_optn,      "transplant_date", "mortality_180d",  4),
    "MIMIC_CXR": (gen_mimic_cxr, "study_date",      "No_Finding",      5)
}

for name, (fn, ts_col, label, seed) in gens.items():
    # 1) Generate base synthetic
    df = fn(N_ROWS, seed)
    # 2) Force exactly two periods {0,1} and balanced labels
    df = make_two_periods(df, ts_col, label, seed+10)
    # 3) Save CSV
    out_path = SYN_DIR / f"{name.lower()}.csv"
    df.to_csv(out_path, index=False)
    # 4) Verify
    print(f"→ Saved {name}: periods={sorted(df[ts_col].unique())}, labels={sorted(df[label].unique())}")

    # Cell 5 (use singular time_unit)

DATASETS = {
    "SEER": {
        "path": SYN_DIR / "seer.csv",
        "timestamp": "diagnosis_year",
        "label": "survival_5yr",
        "numerical": ["age", "tumor_size"],
        "categorical": ["sex", "race", "stage"],
        "time_unit": "year",               # singular
        "initial_train_window": (1975, 1979),
        "train_end": 2018,
        "test_end": 2022,
        "window_length": 5
    },
    "CDC": {
        "path": SYN_DIR / "cdc.csv",
        "timestamp": "case_report_date",
        "label": "death_yn",
        "numerical": ["age"],
        "categorical": ["sex", "race_eth", "hospitalized"],
        "time_unit": "year",               # singular
        "initial_train_window": (2020, 2021),
        "train_end": 2022,
        "test_end": 2023,
        "window_length": 1
    },
    "SWPA": {
        "path": SYN_DIR / "swpa.csv",
        "timestamp": "admit_date",
        "label": "outcome_90d",
        "numerical": ["age", "comorbidity_count", "WBC", "CRP"],
        "categorical": ["sex"],
        "time_unit": "month",              # singular
        "initial_train_window": (2020, 2020),
        "train_end": 2021,
        "test_end": 2022,
        "window_length": 6
    },
    "MIMIC_IV": {
        "path": SYN_DIR / "mimic_iv.csv",
        "timestamp": "chart_time",
        "label": "icu_mortality",
        "numerical": ["age", "heart_rate", "sys_bp", "dia_bp", "resp_rate", "lab_glucose", "lab_creatinine"],
        "categorical": ["gender"],
        "time_unit": "year",
        "initial_train_window": (2009, 2012),
        "train_end": 2018,
        "test_end": 2020,
        "window_length": 3
    },
    "OPTN": {
        "path": SYN_DIR / "optn.csv",
        "timestamp": "transplant_date",
        "label": "mortality_180d",
        "numerical": ["recipient_age", "recipient_meld", "donor_age", "cold_ischemia_time"],
        "categorical": [],
        "time_unit": "year",
        "initial_train_window": (2005, 2009),
        "train_end": 2014,
        "test_end": 2017,
        "window_length": 5
    },
    "MIMIC_CXR": {
        "path": SYN_DIR / "mimic_cxr.csv",
        "timestamp": "study_date",
        "label": "No_Finding",
        "numerical": ["patient_age"],
        "categorical": ["patient_sex", "view_position"],
        "time_unit": "year",
        "initial_train_window": (2010, 2012),
        "train_end": 2016,
        "test_end": 2018,
        "window_length": 2
    }
}

# Cell 6: Run all_historical experiments with LR and patched GBDT (correct get_coefs signature)

import sys
from pathlib import Path

# 0) If needed, adjust this to point at your repo root so that `import emdot` works:
# sys.path.insert(0, "EvaluationOverTime")

import pandas as pd
import numpy as np
from emdot.eot_experiment import EotExperiment
from emdot.models.LR import LR
from emdot.models.GBDT import GBDT as BaseGBDT

# ---- Patch BaseGBDT: implement get_coefs with flexible signature ----
def _gbdt_get_coefs(self, *args, **kwargs):
    # simply return empty dict (or feature_importances_ if you like)
    return {}
BaseGBDT.get_coefs = _gbdt_get_coefs
# clear abstract‐method requirements
if hasattr(BaseGBDT, "__abstractmethods__"):
    BaseGBDT.__abstractmethods__ = set()

# ---- Setup ----
MODELS = {"LR": LR, "GBDT": BaseGBDT}
HYPERPARAMS = {
    "LR":  {"C": [0.01, 0.1, 1.0],           "max_iter": [200]},
    "GBDT":{"n_estimators": [100],           "max_depth": [3], "learning_rate": [0.1]}
}
SEEDS  = [0, 1, 2]
REGIME = "all_historical"

# Ensure RESULTS_DIR exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

for ds_name, cfg in DATASETS.items():
    # 1) Load and preprocess
    df = pd.read_csv(cfg["path"])
    tcol, label = cfg["timestamp"], cfg["label"]

    # Convert time column only if not already integer
    if not pd.api.types.is_integer_dtype(df[tcol]):
        df[tcol] = pd.to_datetime(df[tcol]).dt.year

    # One-hot encode object columns except time & label
    cats = df.select_dtypes(include="object").columns.difference([tcol, label]).tolist()
    if cats:
        df = pd.get_dummies(df, columns=cats, drop_first=False)

    # 2) Build col_dict (including label_cols)
    feature_cols = [c for c in df.columns if c not in [tcol, label]]
    col_dict = {
        "numerical_features":   df[feature_cols].select_dtypes(include="number").columns.tolist(),
        "categorical_features": [],  # after one-hot
        "all_features":         feature_cols,
        "time_col":             tcol,
        "label_cols":           [label]
    }

    # 3) Two-period window bounds
    uniq = sorted(df[tcol].unique())
    if len(uniq) < 2:
        print(f"Skipping {ds_name}: only one time period {uniq}")
        continue
    init_window = (uniq[0], uniq[0])
    train_end, test_end = uniq[0], uniq[1]
    window_length = 1

    # 4) Run experiments for each model
    for mname, Mclass in MODELS.items():
        outdir = RESULTS_DIR / ds_name / mname / REGIME
        outdir.mkdir(parents=True, exist_ok=True)

        exp = EotExperiment(
            dataset_name=ds_name,
            df=df,
            col_dict=col_dict,
            label=label,
            model_class=Mclass,
            model_name=mname,
            hyperparam_grid=HYPERPARAMS[mname],
            training_regime=REGIME,
            initial_train_window=init_window,
            train_end=train_end,
            test_end=test_end,
            train_prop=0.5,
            val_prop=0.25,
            test_prop=0.25,
            window_length=window_length,
            time_unit="Year",
            model_folder=str(outdir)
        )

        print(f"→ {ds_name} | {mname} | {REGIME} | window {init_window}->{train_end}->{test_end}")
        try:
            res_df, _ = exp.run_experiment(SEEDS, eval_metric="auc")
        except Exception as e:
            print(f"   ✗ Skipped {ds_name}/{mname}: {e}")
            continue

        res_df.to_csv(outdir / "results.csv", index=False)

print("✅ Synthetic all_historical experiments complete.")

# Cell 7 (revised): Aggregate & plot AUC over time

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Adjust this to wherever your results live
RESULTS_DIR = Path("results") / "synthetic_notebook"

# 1) Gather all results.csv
records = []
for ds_dir in RESULTS_DIR.iterdir():
    if not ds_dir.is_dir(): continue
    for model_dir in ds_dir.iterdir():
        if not model_dir.is_dir(): continue
        for regime_dir in model_dir.iterdir():
            if not regime_dir.is_dir(): continue
            f = regime_dir / "results.csv"
            if not f.exists(): continue
            df = pd.read_csv(f)
            # pick out a time column
            if "time" in df.columns:
                df["time_axis"] = df["time"]
            elif "window_end" in df.columns:
                df["time_axis"] = df["window_end"]
            else:
                df["time_axis"] = range(len(df))
            # pick out an AUC column
            if "auc" in df.columns:
                df["auc_axis"] = df["auc"]
            else:
                # fallback: first column containing "auc"
                auc_cols = [c for c in df.columns if "auc" in c.lower()]
                df["auc_axis"] = df[auc_cols[0]] if auc_cols else df.iloc[:, -1]
            df["dataset"] = ds_dir.name
            df["model"]   = model_dir.name
            df["regime"]  = regime_dir.name
            records.append(df)

if not records:
    print(f"No results found under {RESULTS_DIR.resolve()}")
else:
    all_df = pd.concat(records, ignore_index=True)
    print("Columns in combined DataFrame:", all_df.columns.tolist())

    plt.figure(figsize=(8,6))
    for (ds, m, r), grp in all_df.groupby(["dataset","model","regime"]):
        # extract as 1D numpy arrays
        x = grp["time_axis"].to_numpy()
        y = grp["auc_axis"].to_numpy()
        plt.plot(x, y, marker="o", label=f"{ds}-{m}-{r}")

    plt.xlabel("Time")
    plt.ylabel("AUC")
    plt.title("Synthetic Data: AUC Over Time")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # Cell 7: Aggregate & plot AUC results

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
import numpy as np

# <<< EDIT THIS TO YOUR ACTUAL RESULTS FOLDER >>>
RESULTS_DIR = Path("EvaluationOverTime/results/synthetic_notebook")
# e.g. wherever you see .../CDC/LR/sliding_window/0/results.csv

if not RESULTS_DIR.exists():
    raise FileNotFoundError(f"Results directory not found at {RESULTS_DIR!r}")

all_results = []
baseline_auc = {}

# 1) Load all_historical and sliding_window/<step> CSVs
for ds_dir in RESULTS_DIR.iterdir():
    if not ds_dir.is_dir(): continue
    for model_dir in ds_dir.iterdir():
        if not model_dir.is_dir(): continue

        # all_historical
        f_ah = model_dir / "all_historical" / "results.csv"
        if f_ah.exists():
            df = pd.read_csv(f_ah)
            all_results.append((ds_dir.name, model_dir.name, "all_historical", df))
            baseline_auc[ds_dir.name] = df["auc"].mean()

        # sliding_window/<step>
        sw_root = model_dir / "sliding_window"
        if sw_root.exists():
            for step_dir in sorted(sw_root.iterdir(), key=lambda d: int(d.name) if d.name.isdigit() else -1):
                f_sw = step_dir / "results.csv"
                if f_sw.exists():
                    df = pd.read_csv(f_sw)
                    all_results.append((ds_dir.name, model_dir.name, "sliding_window", df))

print(f"Loaded {len(all_results)} result‐sheets.")
print("Baseline (all_historical) AUCs:", baseline_auc)

# 2) Helper to pick the “time” column
def get_time_val(df):
    # try common names
    for col in ["time","window_end","test_end","period"]:
        if col in df.columns:
            return df[col].iloc[0]
    # fallback: any column containing “time”
    for col in df.columns:
        if "time" in col.lower():
            return df[col].iloc[0]
    # last fallback: first numeric column
    num_cols = df.select_dtypes(include="number").columns
    return df[num_cols[0]].iloc[0] if len(num_cols) else 0

# 3) Setup for plotting
datasets   = ["SEER","CDC","SWPA","MIMIC_IV","OPTN","MIMIC_CXR"]
sw_entries = [df for (_,_,r,df) in all_results if r=="sliding_window"]
n_steps    = max((len(df) for df in sw_entries), default=1)

cmap = plt.get_cmap("viridis", n_steps)
norm = mcolors.Normalize(vmin=0, vmax=n_steps-1)

fig = plt.figure(figsize=(14, 8))
gs  = GridSpec(3, 4, height_ratios=[1,2,2], hspace=0.4, wspace=0.3)

# 4) Top row: table of all_historical AUCs
ax_tab = fig.add_subplot(gs[0, :])
ax_tab.axis("off")
tbl = (
    pd.DataFrame([
        (model, ds, df["auc"].mean())
        for ds, model, reg, df in all_results if reg=="all_historical"
    ], columns=["Model","Dataset","AUC"])
    .pivot(index="Model", columns="Dataset", values="AUC")
    .round(3)
)
ax_tab.table(
    cellText=tbl.values,
    rowLabels=tbl.index,
    colLabels=tbl.columns,
    loc="center"
)

# 5) Panels for each dataset
for idx, ds in enumerate(datasets):
    ax = fig.add_subplot(gs[1 + idx//4, idx % 4])
    ax.set_title(ds)
    ax.set_ylim(0.6, 1.0)
    ax.grid(alpha=0.3)

    # gather slides
    slides = [(d,m,df) for (d,m,reg,df) in all_results if d==ds and reg=="sliding_window"]
    slides.sort(key=lambda t: get_time_val(t[2]))

    for step, (_, _, df) in enumerate(slides):
        # pick x-axis column
        for col in ["time","window_end","test_end","period"] + [c for c in df.columns if "time" in c.lower()]:
            if col in df.columns:
                x = df[col].to_numpy()
                break
        else:
            x = np.arange(len(df))
        y = df["auc"].to_numpy()
        ax.plot(x, y, color=cmap(norm(step)), lw=2)

    # plot baseline
    if ds in baseline_auc:
        xmin, xmax = ax.get_xlim()
        ax.hlines(baseline_auc[ds], xmin, xmax, linestyles="--", color="red", label="All-period")

    # deployment marker
    if slides:
        last = slides[-1][2]
        # pick last x
        for col in ["time","window_end","test_end","period"] + [c for c in last.columns if "time" in c.lower()]:
            if col in last.columns:
                td = last[col].iloc[-1]
                break
        else:
            td = len(last) - 1
        ad = last["auc"].iloc[-1]
        ci = last.get("ci", pd.Series([0]*len(last))).iloc[-1]
        ax.errorbar([td], [ad], yerr=ci, fmt="o", color="black", label="Deploy")

# 6) Colorbar
cax = fig.add_subplot(gs[2, :])
sm  = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
cbar.set_label("Training window age (earliest → latest)")

# 7) Shared legend
lines = [
    plt.Line2D([], [], color="red", linestyle="--"),
    plt.Line2D([], [], color="black", marker="o", linestyle="None")
]
fig.legend(lines, ["All-period baseline","Deployment point"],
           loc="lower center", ncol=2)

plt.tight_layout(rect=[0,0.03,1,0.95])
plt.show()