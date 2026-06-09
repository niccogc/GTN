#!/usr/bin/env python3
import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import statistics

import yaml

DATASET_INFO = {
    "adult": ("AD", "Adult", "Classification"),
    "bank": ("BA", "Bank Marketing", "Classification"),
    "mushrooms": ("MU", "Mushrooms", "Classification"),
    "winequalityc": ("WQ", "Wine Quality", "Classification"),
    "student_dropout": ("SD", "Students' Dropout", "Classification"),
    "car_evaluation": ("CE", "Car Evaluation", "Classification"),
    "breast": ("BR", "Breast Cancer Wisconsin", "Classification"),
    "hearth": ("HE", "Heart Disease", "Classification"),
    "wine": ("WI", "Wine", "Classification"),
    "iris": ("IR", "Iris", "Classification"),
    "popularity": ("PO", "Online News Popularity", "Regression"),
    "appliances": ("AP", "Appliances Energy Prediction", "Regression"),
    "bike": ("BK", "Bike Sharing", "Regression"),
    "ai4i": ("AI", "AI4I", "Regression"),
    "seoulBike": ("SB", "Seoul Bike Sharing", "Regression"),
    "abalone": ("AB", "Abalone", "Regression"),
    "obesity": ("OB", "Obesity Levels", "Regression"),
    "concrete": ("CO", "Concrete Compressive Strength", "Regression"),
    "energy_efficiency": ("EE", "Energy Efficiency", "Regression"),
    "student_perf": ("SP", "Student Performance", "Regression"),
    "realstate": ("RE", "Real Estate Valuation", "Regression"),
}

MODEL_GROUPS = {
    "MPO2": ["MPO2", "MPO2TypeI"],
    "LMPO2": ["LMPO2", "LMPO2TypeI"],
    "MMPO2": ["MMPO2", "MMPO2TypeI"],
    "CPDA": ["CPDA", "CPDATypeI"],
    "CPD-S": ["CPD-S", "CPD-S-TypeI"],
    "TNML_P": ["TNML_P"],
    "TNML_F": ["TNML_F"],
    "Ring": ["BosonMPS"],
    # External baseline models (from CSV files)
    "MLP": ["MLP"],
    "XGBoost": ["XGBoost"],
    "GP": ["GP"],
}

MODEL_ORDER = ["MPO2", "LMPO2", "MMPO2", "Ring", "CPDA", "CPD-S", "TNML_P", "TNML_F"]

# External baseline models from CSV files
EXTERNAL_MODEL_ORDER = ["MLP", "XGBoost", "GP"]

MODEL_LATEX_NAMES = {
    "MPO2": r"\textbf{(MPO)}$\bm{^2}$",
    "LMPO2": r"\textbf{(LMPO)}$\bm{^2}$",
    "MMPO2": r"\textbf{(MMPO)}$\bm{^2}$",
    "CPDA": r"\textbf{CPD-A}",
    "TNML_P": r"\textbf{TNML-P}",
    "TNML_F": r"\textbf{TNML-F}",
    "Ring": r"\textbf{Ring}",
    "MLP": r"\textbf{MLP}",
    "XGBoost": r"\textbf{XGBoost}",
    "GP": r"\textbf{GP}",
    "CPD-S": r"\textbf{TEMPO}",
}

ALL_MODE_LATEX_NAMES = {
    "N-MPO2": r"\textbf{N-(MPO)}$\bm{^2}$",
    "G-MPO2": r"\textbf{G-(MPO)}$\bm{^2}$",
    "N-CPDA": r"\textbf{N-CPD-A}",
    "G-CPDA": r"\textbf{G-CPD-A}",
    "TEMPO": r"\textbf{TEMPO}",
    "N-TNML_P": r"\textbf{N-TNML-P}",
    "G-TNML_P": r"\textbf{G-TNML-P}",
    "N-TNML_F": r"\textbf{N-TNML-F}",
    "G-TNML_F": r"\textbf{G-TNML-F}",
    "G-Ring": r"\textbf{G-Ring}",
    "TNML_P": r"\textbf{TNML-P}",
    "TNML_F": r"\textbf{TNML-F}",
    "MLP": r"\textbf{MLP}",
    "XGBoost": r"\textbf{XGBoost}",
    "GP": r"\textbf{GP}",
    "Base": r"\textbf{Base}",
}


def load_tnml_best_configs(conf_dir: Path) -> dict[str, dict[str, str]]:
    """Load TNML best configs to know which trainer to use per dataset.
    
    Checks conf/best_conf/tnml/ for ntn/gtn configs, and also checks
    conf/best_conf/dmrg/ to see if dmrg has better results for any dataset.
    
    Returns: {"TNML_P": {"realstate": "ntn", "iris": "gtn", ...}, "TNML_F": {...}}
    """
    tnml_configs = {}
    tnml_dir = conf_dir / "tnml"
    dmrg_dir = conf_dir / "dmrg"
    
    for model in ["tnml_p", "tnml_f"]:
        model_key = model.upper()
        tnml_configs[model_key] = {}
        
        # Load from tnml directory (ntn/gtn configs)
        config_file = tnml_dir / f"{model}.yaml"
        if config_file.exists():
            with open(config_file) as f:
                data = yaml.safe_load(f)
            
            best_configs = data.get("_best_configs", {})
            for dataset, cfg in best_configs.items():
                trainer = cfg.get("trainer", "gtn")
                val_quality = cfg.get("avg_val_quality", float('-inf'))
                tnml_configs[model_key][dataset] = {
                    "trainer": trainer,
                    "val_quality": val_quality,
                }
        
        # Check DMRG configs and override if better
        dmrg_file = dmrg_dir / f"{model}.yaml"
        if dmrg_file.exists():
            with open(dmrg_file) as f:
                dmrg_data = yaml.safe_load(f)
            
            dmrg_configs = dmrg_data.get("_best_configs", {})
            for dataset, cfg in dmrg_configs.items():
                dmrg_val = cfg.get("avg_val_quality", float('-inf'))
                
                # Compare with existing config
                if dataset in tnml_configs[model_key]:
                    existing_val = tnml_configs[model_key][dataset].get("val_quality", float('-inf'))
                    if dmrg_val > existing_val:
                        tnml_configs[model_key][dataset] = {
                            "trainer": "dmrg",
                            "val_quality": dmrg_val,
                        }
                else:
                    # No existing config, use DMRG
                    tnml_configs[model_key][dataset] = {
                        "trainer": "dmrg",
                        "val_quality": dmrg_val,
                    }
    
    # Flatten to just trainer strings for backward compatibility
    result = {}
    for model_key, datasets in tnml_configs.items():
        result[model_key] = {ds: cfg["trainer"] for ds, cfg in datasets.items()}
    
    return result

# CLASSIFICATION_DATASETS = ["iris", "hearth", "winequalityc", "breast", "adult", 
#                            "bank", "wine", "car_evaluation", "student_dropout", "mushrooms"]

# REGRESSION_DATASETS = ["realstate", "energy_efficiency", "concrete", "student_perf",
#                        "obesity", "abalone", "seoulBike", "ai4i", "bike", "popularity"]

CLASSIFICATION_DATASETS = ["iris", "hearth", "winequalityc", "wine"]

REGRESSION_DATASETS = ["realstate", "energy_efficiency", "concrete", "abalone","ai4i"]


@dataclass
class SeedResult:
    seed: int
    best_val_quality: float
    test_quality_at_best_val: float
    best_epoch: int


@dataclass
class ModelDatasetResult:
    model: str
    dataset: str
    trainer: str
    seed_results: list[SeedResult] = field(default_factory=list)
    
    @property
    def n_seeds(self) -> int:
        return len(self.seed_results)
    
    @property
    def mean_test_quality(self) -> float:
        if not self.seed_results:
            return float('nan')
        return statistics.mean(r.test_quality_at_best_val for r in self.seed_results)
    
    @property
    def std_test_quality(self) -> float:
        if len(self.seed_results) < 2:
            return 0.0
        return statistics.stdev(r.test_quality_at_best_val for r in self.seed_results)


def parse_results_json(path: Path, use_val_loss: bool = False) -> Optional[SeedResult]:
    try:
        with open(path) as f:
            data = json.load(f)
        metrics_log = data.get("metrics_log", [])
        if not metrics_log:
            return None
        
        best_epoch = -1
        test_quality_at_best = float('nan')
        
        if use_val_loss:
            best_val = float('inf')
            for entry in metrics_log:
                val_loss = entry.get("val_loss")
                if val_loss is not None and val_loss <= best_val:
                    best_val, best_epoch = val_loss, entry.get("epoch", -1)
                    test_quality_at_best = entry.get("test_quality", float('nan'))
        else:
            best_val = float('-inf')
            for entry in metrics_log:
                val_q = entry.get("val_quality")
                if val_q is not None and val_q >= best_val:
                    best_val, best_epoch = val_q, entry.get("epoch", -1)
                    test_quality_at_best = entry.get("test_quality", float('nan'))
        
        # Fallback: if test_quality not in metrics_log, use top-level test_quality
        if math.isnan(test_quality_at_best):
            top_level_test = data.get("test_quality")
            if top_level_test is not None:
                test_quality_at_best = top_level_test
        
        # Skip results with no valid test_quality
        if math.isnan(test_quality_at_best):
            return None
        
        if best_epoch < 0: return None
        return SeedResult(seed=data.get("config", {}).get("seed", 0), 
                          best_val_quality=best_val, test_quality_at_best_val=test_quality_at_best, best_epoch=best_epoch)
    except: return None


def collect_results(test_outputs_dir: Path, trainer: str, use_val_loss: bool = False) -> dict[tuple[str, str], ModelDatasetResult]:
    trainer_dir = test_outputs_dir / trainer
    if not trainer_dir.exists(): return {}
    results = {}
    for model_dir in trainer_dir.iterdir():
        if not model_dir.is_dir(): continue
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir(): continue
            key = (model_dir.name, dataset_dir.name)
            if key not in results:
                results[key] = ModelDatasetResult(model=model_dir.name, dataset=dataset_dir.name, trainer=trainer)
            for seed_dir in dataset_dir.iterdir():
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"): continue
                res_file = seed_dir / "results.json"
                if res_file.exists():
                    seed_res = parse_results_json(res_file, use_val_loss)
                    if seed_res: results[key].seed_results.append(seed_res)
    return results


def get_baseline_results(test_outputs_dir: Path, datasets: list[str]) -> dict[str, float]:
    baselines = {}
    for ds in datasets:
        path = test_outputs_dir / "mean_baseline" / ds / "results.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                raw_val = data.get("test_quality", float('nan'))
                baselines[ds] = raw_val
    return baselines


def load_external_csv_results(csv_dir: Path) -> dict[tuple[str, str], ModelDatasetResult]:
    results = {}
    
    csv_configs = {
        "MLP": {
            "file": "test_results_mlp.csv",
            "cols": ["timestamp", "model_type", "dataset", "num_layers", "num_channels", 
                     "test_rmse", "test_r2", "test_accuracy", "num_params", "converged_epoch"]
        },
        "XGBoost": {
            "file": "test_results_xgboost.csv",
            "cols": ["timestamp", "model_type", "dataset", "n_estimators", "max_depth",
                     "test_rmse", "test_r2", "test_accuracy", "num_params", "converged_epoch"]
        },
        "GP": {
            "file": "test_results_gp.csv",
            "cols": ["timestamp", "model_type", "dataset", "best_kernel_name", "best_alpha",
                     "test_rmse", "test_r2", "test_accuracy", "num_params", "converged_epoch"]
        },
    }
    
    for model_name, config in csv_configs.items():
        csv_path = csv_dir / config["file"]
        if not csv_path.exists():
            continue
        
        dataset_runs: dict[str, list[float]] = {}
        
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f, fieldnames=config["cols"])
            for row in reader:
                dataset = row["dataset"]
                test_acc_str = row["test_accuracy"]
                test_r2_str = row["test_r2"]
                
                test_acc = None if test_acc_str in ("nan", "", None) else float(test_acc_str)
                test_r2 = None if test_r2_str in ("nan", "", None) else float(test_r2_str)
                
                if test_acc is not None and not math.isnan(test_acc):
                    metric = test_acc
                elif test_r2 is not None and not math.isnan(test_r2):
                    metric = test_r2
                else:
                    continue
                
                if dataset not in dataset_runs:
                    dataset_runs[dataset] = []
                dataset_runs[dataset].append(metric)
        
        for dataset, metrics in dataset_runs.items():
            if not metrics:
                continue
            seed_results = [
                SeedResult(seed=i, best_val_quality=0.0, test_quality_at_best_val=m, best_epoch=0)
                for i, m in enumerate(metrics)
            ]
            key = (model_name, dataset)
            results[key] = ModelDatasetResult(
                model=model_name, dataset=dataset, trainer="external", seed_results=seed_results
            )
    
    _load_matlab_cpd_results(csv_dir, results)
    
    return results


def _load_matlab_cpd_results(csv_dir: Path, results: dict[tuple[str, str], ModelDatasetResult]):
    matlab_files = [
        ("CPD-S-TypeI", "matlab_results_20250915_104118.csv"),
        ("CPD-S", "matlab_results_type2_20250915_111937.csv"),
    ]
    
    for model_name, filename in matlab_files:
        csv_path = csv_dir / filename
        if not csv_path.exists():
            continue
        
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset = row["Dataset"]
                
                mean_acc = row.get("Mean_Test_Accuracy", "")
                mean_r2 = row.get("Mean_Test_R2", "")
                sem_acc = row.get("SEM_Test_Accuracy", "")
                sem_r2 = row.get("SEM_Test_R2", "")
                
                mean_acc = None if mean_acc in ("NaN", "", None) else float(mean_acc)
                mean_r2 = None if mean_r2 in ("NaN", "", None) else float(mean_r2)
                sem_acc = None if sem_acc in ("NaN", "", None) else float(sem_acc)
                sem_r2 = None if sem_r2 in ("NaN", "", None) else float(sem_r2)
                
                if mean_acc is not None and not math.isnan(mean_acc):
                    mean_val = mean_acc / 100.0
                    sem_val = (sem_acc / 100.0) if sem_acc and not math.isnan(sem_acc) else 0.0
                elif mean_r2 is not None and not math.isnan(mean_r2):
                    mean_val = mean_r2
                    sem_val = sem_r2 if sem_r2 and not math.isnan(sem_r2) else 0.0
                else:
                    continue
                
                n_assumed = 5
                std_val = sem_val * math.sqrt(n_assumed)
                
                seed_results = [
                    SeedResult(seed=0, best_val_quality=0.0, test_quality_at_best_val=mean_val + std_val, best_epoch=0),
                    SeedResult(seed=1, best_val_quality=0.0, test_quality_at_best_val=mean_val - std_val, best_epoch=0),
                ]
                
                key = (model_name, dataset)
                results[key] = ModelDatasetResult(
                    model=model_name, dataset=dataset, trainer="external", seed_results=seed_results
                )


def get_best_variant_result(results, unified_model, dataset):
    variants = MODEL_GROUPS.get(unified_model, [unified_model])
    best_res, best_mean = None, float('-inf')
    for var in variants:
        key = (var, dataset)
        if key in results and results[key].n_seeds > 0:
            if results[key].mean_test_quality > best_mean:
                best_mean, best_res = results[key].mean_test_quality, results[key]
    return best_res


def get_mpo2_datasets(results: dict, dataset_list: list[str]) -> list[str]:
    mpo2_variants = MODEL_GROUPS.get("MPO2", ["MPO2"])
    available = set()
    for var in mpo2_variants:
        for key in results:
            if key[0] == var and key[1] in dataset_list:
                available.add(key[1])
    return [d for d in dataset_list if d in available]


def get_filtered_model_order(trainer, include_external=False):
    order = [m for m in MODEL_ORDER if m != "Ring"]
    if trainer == "gtn": order.append("Ring")
    if include_external:
        order.extend(EXTERNAL_MODEL_ORDER)
    return order


def find_best_per_column(results, datasets, trainer, include_external=False):
    best_per_col = {}
    order = get_filtered_model_order(trainer, include_external)
    for ds in datasets:
        vals = [get_best_variant_result(results, m, ds).mean_test_quality 
                for m in order if get_best_variant_result(results, m, ds)]
        best_per_col[ds] = max(vals) if vals else float('-inf')
    return best_per_col


def find_best_tn_per_column(results, datasets, trainer):
    best_tn = {}
    tn_order = get_filtered_model_order(trainer, include_external=False)
    for ds in datasets:
        vals = [get_best_variant_result(results, m, ds).mean_test_quality 
                for m in tn_order if get_best_variant_result(results, m, ds)]
        best_tn[ds] = max(vals) if vals else float('-inf')
    return best_tn


def format_mean_value(mean, is_best_overall, is_best_tn):
    val_str = f"{mean * 100:.2f}"
    if is_best_overall:
        return f"\\textbf{{{val_str}}}"
    if is_best_tn:
        return f"\\underline{{{val_str}}}"
    return val_str


def generate_table(results, datasets, task, trainer, test_outputs_dir, include_external=False, show_avg=False):
    order = get_filtered_model_order(trainer, include_external)
    datasets = get_mpo2_datasets(results, datasets)
    if not datasets: return ""
    
    n_cols = len(datasets) + (1 if show_avg else 0)
    dataset_codes = [DATASET_INFO[d][0] for d in datasets]
    best_overall = find_best_per_column(results, datasets, trainer, include_external)
    best_tn = find_best_tn_per_column(results, datasets, trainer)
    
    row_avgs = {m: statistics.mean([get_best_variant_result(results, m, d).mean_test_quality 
                                    for d in datasets if get_best_variant_result(results, m, d)]) 
                for m in order if any(get_best_variant_result(results, m, d) for d in datasets)}
    
    tn_order = get_filtered_model_order(trainer, include_external=False)
    
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\scriptsize", 
             r"\begin{tabular}{l" + "r" * n_cols + "}", r"\toprule"]
    header = " & ".join(dataset_codes)
    if show_avg:
        header += r" & Avg"
    lines.append(header + r" \\")
    lines.append(r"\midrule")

    for m in order:
        model_latex = MODEL_LATEX_NAMES.get(m, f"\\textbf{{{m}}}")
        means, stds = [], []
        is_tn_model = m in tn_order
        for d in datasets:
            res = get_best_variant_result(results, m, d)
            if not res: 
                means.append("--"); stds.append("--")
            else:
                is_best_overall = abs(res.mean_test_quality - best_overall[d]) < 1e-9
                is_best_tn = is_tn_model and abs(res.mean_test_quality - best_tn[d]) < 1e-9
                means.append(format_mean_value(res.mean_test_quality, is_best_overall, is_best_tn))
                stds.append(f"$\\pm${res.std_test_quality*100:.2f}")
        
        if show_avg:
            avg = row_avgs.get(m, float('nan'))
            means.append(f"{avg*100:.2f}" if avg == avg else "--")
        
        lines.append(f"{model_latex} & " + " & ".join(means) + r" \\")
        has_nonzero_std = any(s not in ("--", "", "$\\pm$0.00") for s in stds)
        if has_nonzero_std:
            stds_extra = stds + ([""] if show_avg else [])
            lines.append(" & " + " & ".join(stds_extra) + r" \\")
        lines.append(r"\midrule")

    bs = get_baseline_results(test_outputs_dir, datasets)
    base_row = [f"{bs[d]*100:.2f}" if d in bs else "--" for d in datasets]
    if show_avg:
        avg_b = statistics.mean([bs[d] for d in datasets if d in bs]) if bs else 0.0
        base_row.append(f"{avg_b*100:.2f}")
    lines.append(r"\textbf{Mean} & " + " & ".join(base_row) + r" \\")
    
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_combined_table(results, reg_ds, class_ds, trainer, test_outputs_dir, include_external=False, show_avg=False):
    order = get_filtered_model_order(trainer, include_external)
    tn_order = get_filtered_model_order(trainer, include_external=False)
    reg_ds = get_mpo2_datasets(results, reg_ds)
    class_ds = get_mpo2_datasets(results, class_ds)
    if not reg_ds and not class_ds: return ""

    def n_cols(ds):
        return len(ds) + (1 if show_avg else 0)
    col_spec = "l" + ("r" * n_cols(reg_ds) if reg_ds else "") + ("||" if reg_ds and class_ds else "") + ("r" * n_cols(class_ds) if class_ds else "")
    lines = [r"\begin{table*}[!htbp]", r"\centering", r"\scriptsize", r"\begin{tabular}{" + col_spec + "}", r"\toprule"]
    
    h = []
    for ds in [reg_ds, class_ds]:
        if not ds: continue
        parts = [DATASET_INFO[d][0] for d in ds]
        if show_avg:
            parts.append("Avg")
        h.append(" & ".join(parts))
    lines.append(" & " + " & ".join(h) + r" \\")
    lines.append(r"\midrule")

    best_overall_r = find_best_per_column(results, reg_ds, trainer, include_external)
    best_tn_r = find_best_tn_per_column(results, reg_ds, trainer)
    best_overall_c = find_best_per_column(results, class_ds, trainer, include_external)
    best_tn_c = find_best_tn_per_column(results, class_ds, trainer)
    
    for m in order:
        model_latex = MODEL_LATEX_NAMES.get(m, f"\\textbf{{{m}}}")
        is_tn_model = m in tn_order
        means, stds = [], []
        for ds, best_overall, best_tn in [(reg_ds, best_overall_r, best_tn_r), (class_ds, best_overall_c, best_tn_c)]:
            if not ds: continue
            cur_vals = []
            for d in ds:
                res = get_best_variant_result(results, m, d)
                if not res: 
                    means.append("--"); stds.append("--")
                else:
                    is_best_overall = abs(res.mean_test_quality - best_overall[d]) < 1e-9
                    is_best_tn = is_tn_model and abs(res.mean_test_quality - best_tn[d]) < 1e-9
                    means.append(format_mean_value(res.mean_test_quality, is_best_overall, is_best_tn))
                    stds.append(f"$\\pm${res.std_test_quality*100:.2f}"); cur_vals.append(res.mean_test_quality)
            
            if show_avg:
                avg = statistics.mean(cur_vals) if cur_vals else float('nan')
                means.append(f"{avg*100:.2f}" if avg==avg else "--")
                stds.append("")
        
        lines.append(f"{model_latex} & " + " & ".join(means) + r" \\")
        has_nonzero_std = any(s not in ("--", "", "$\\pm$0.00") for s in stds)
        if has_nonzero_std:
            lines.append(" & " + " & ".join(stds) + r" \\")
        lines.append(r"\midrule")

    base_vals = []
    for ds in [reg_ds, class_ds]:
        if not ds: continue
        bs = get_baseline_results(test_outputs_dir, ds)
        base_vals.extend([f"{bs[d]*100:.2f}" if d in bs else "--" for d in ds])
        if show_avg:
            avg_b = statistics.mean([bs[d] for d in ds if d in bs]) if bs else 0.0
            base_vals.append(f"{avg_b*100:.2f}")
    lines.append(r"\textbf{Mean} & " + " & ".join(base_vals) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


ALL_MODE_MODEL_ORDER = ["N-MPO2", "G-MPO2", "G-Ring", "N-CPDA", "G-CPDA", "TEMPO", 
                        "N-TNML_P", "G-TNML_P", "N-TNML_F", "G-TNML_F"]

# New order with TNML unified (best across trainers)
UNIFIED_MODEL_ORDER = ["N-MPO2", "G-MPO2", "G-Ring", "N-CPDA", "G-CPDA", "TEMPO", 
                       "TNML_P", "TNML_F"]

ALL_MODE_MPO2_VARIANTS = ["MPO2", "LMPO2", "MMPO2", "MPO2TypeI", "LMPO2TypeI", "MMPO2TypeI"]


def get_all_mode_model_order(include_external=False):
    order = list(ALL_MODE_MODEL_ORDER)
    if include_external:
        order.extend(EXTERNAL_MODEL_ORDER)
    return order


def generate_all_table(ntn_results, gtn_results, reg_ds, class_ds, test_outputs_dir, include_external=False, show_avg=False):
    combined_results = {}
    for key, val in ntn_results.items():
        if key[0] in EXTERNAL_MODEL_ORDER:
            combined_results[key] = val
        else:
            new_key = (f"N-{key[0]}", key[1])
            combined_results[new_key] = val
    for key, val in gtn_results.items():
        if key[0] in EXTERNAL_MODEL_ORDER:
            combined_results[key] = val
        else:
            new_key = (f"G-{key[0]}", key[1])
            combined_results[new_key] = val
    
    reg_ds = get_mpo2_datasets(gtn_results, reg_ds)
    class_ds = get_mpo2_datasets(gtn_results, class_ds)
    if not reg_ds and not class_ds:
        return ""
    
    order = get_all_mode_model_order(include_external)
    
    def n_cols(ds):
        return len(ds) + (1 if show_avg else 0)
    col_spec = "l" + ("r" * n_cols(reg_ds) if reg_ds else "") + ("||" if reg_ds and class_ds else "") + ("r" * n_cols(class_ds) if class_ds else "")
    lines = [r"\begin{table*}[!htbp]", r"\centering", r"\scriptsize", r"\begin{tabular}{" + col_spec + "}", r"\toprule"]
    
    h = []
    for ds in [reg_ds, class_ds]:
        if not ds: continue
        parts = [DATASET_INFO[d][0] for d in ds]
        if show_avg:
            parts.append("Avg")
        h.append(" & ".join(parts))
    lines.append(" & " + " & ".join(h) + r" \\")
    lines.append(r"\midrule")
    
    def get_result_for_all_mode(model_key, dataset):
        if model_key == "TEMPO":
            best_res, best_mean = None, float('-inf')
            for var in ["CPD-S", "CPD-S-TypeI"]:
                for prefix in ["N-", "G-"]:
                    key = (f"{prefix}{var}", dataset)
                    if key in combined_results and combined_results[key].n_seeds > 0:
                        if combined_results[key].mean_test_quality > best_mean:
                            best_mean = combined_results[key].mean_test_quality
                            best_res = combined_results[key]
            return best_res
        
        if model_key.startswith("N-") or model_key.startswith("G-"):
            prefix = model_key[:2]
            base_model = model_key[2:]
            
            if base_model == "MPO2":
                variants = ALL_MODE_MPO2_VARIANTS
            else:
                variants = MODEL_GROUPS.get(base_model, [base_model])
            
            best_res, best_mean = None, float('-inf')
            for var in variants:
                key = (f"{prefix}{var}", dataset)
                if key in combined_results and combined_results[key].n_seeds > 0:
                    if combined_results[key].mean_test_quality > best_mean:
                        best_mean = combined_results[key].mean_test_quality
                        best_res = combined_results[key]
            return best_res
        
        key = (model_key, dataset)
        if key in combined_results and combined_results[key].n_seeds > 0:
            return combined_results[key]
        return None
    
    tn_order = get_all_mode_model_order(include_external=False)
    
    def find_best_overall(datasets):
        best = {}
        for ds in datasets:
            vals = [get_result_for_all_mode(m, ds).mean_test_quality 
                    for m in order if get_result_for_all_mode(m, ds)]
            best[ds] = max(vals) if vals else float('-inf')
        return best
    
    def find_best_tn(datasets):
        best = {}
        for ds in datasets:
            vals = [get_result_for_all_mode(m, ds).mean_test_quality 
                    for m in tn_order if get_result_for_all_mode(m, ds)]
            best[ds] = max(vals) if vals else float('-inf')
        return best
    
    best_overall_r = find_best_overall(reg_ds)
    best_tn_r = find_best_tn(reg_ds)
    best_overall_c = find_best_overall(class_ds)
    best_tn_c = find_best_tn(class_ds)
    
    for m in order:
        model_latex = ALL_MODE_LATEX_NAMES.get(m, f"\\textbf{{{m}}}")
        is_tn_model = m in tn_order
        means, stds = [], []
        for ds, best_overall, best_tn in [(reg_ds, best_overall_r, best_tn_r), (class_ds, best_overall_c, best_tn_c)]:
            if not ds:
                continue
            cur_vals = []
            for d in ds:
                res = get_result_for_all_mode(m, d)
                if not res:
                    means.append("--")
                    stds.append("--")
                else:
                    is_best_overall = abs(res.mean_test_quality - best_overall[d]) < 1e-9
                    is_best_tn = is_tn_model and abs(res.mean_test_quality - best_tn[d]) < 1e-9
                    means.append(format_mean_value(res.mean_test_quality, is_best_overall, is_best_tn))
                    stds.append(f"$\\pm${res.std_test_quality*100:.2f}")
                    cur_vals.append(res.mean_test_quality)
            
            if show_avg:
                avg = statistics.mean(cur_vals) if cur_vals else float('nan')
                means.append(f"{avg*100:.2f}" if avg == avg else "--")
                stds.append("")
        
        lines.append(f"{model_latex} & " + " & ".join(means) + r" \\")
        has_nonzero_std = any(s not in ("--", "", "$\\pm$0.00") for s in stds)
        if has_nonzero_std:
            lines.append(" & " + " & ".join(stds) + r" \\")
        lines.append(r"\midrule")
    
    base_vals = []
    for ds in [reg_ds, class_ds]:
        if not ds:
            continue
        bs = get_baseline_results(test_outputs_dir, ds)
        base_vals.extend([f"{bs[d]*100:.2f}" if d in bs else "--" for d in ds])
        if show_avg:
            avg_b = statistics.mean([bs[d] for d in ds if d in bs]) if bs else 0.0
            base_vals.append(f"{avg_b*100:.2f}")
    lines.append(r"\textbf{Mean} & " + " & ".join(base_vals) + r" \\")
    
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def generate_unified_table(
    ntn_results: dict,
    gtn_results: dict,
    dmrg_results: dict,
    reg_ds: list[str],
    class_ds: list[str],
    test_outputs_dir: Path,
    tnml_best_configs: dict[str, dict[str, str]],
    include_external: bool = False,
) -> str:
    """Generate table with TNML unified (using best trainer per dataset).
    
    Format matches the exact LaTeX structure requested with:
    - Regression columns on left, Classification on right
    - cmidrule separators between models
    - Two rows per model (mean and std)
    - TNML models use best trainer per dataset from tnml_best_configs
    - Double cmidrule for section breaks
    """
    # Combine results with prefixes
    combined_results = {}
    for key, val in ntn_results.items():
        if key[0] in EXTERNAL_MODEL_ORDER:
            combined_results[key] = val
        else:
            new_key = (f"N-{key[0]}", key[1])
            combined_results[new_key] = val
    for key, val in gtn_results.items():
        if key[0] in EXTERNAL_MODEL_ORDER:
            combined_results[key] = val
        else:
            new_key = (f"G-{key[0]}", key[1])
            combined_results[new_key] = val
    # Add DMRG results with D- prefix
    for key, val in dmrg_results.items():
        new_key = (f"D-{key[0]}", key[1])
        combined_results[new_key] = val
    
    reg_ds = get_mpo2_datasets(gtn_results, reg_ds)
    class_ds = get_mpo2_datasets(gtn_results, class_ds)
    if not reg_ds and not class_ds:
        return ""
    
    # Define model sections:
    # Section 1: MPO2 variants and Ring (proposed models)
    # Section 2: CPD-A and TEMPO (other TN methods)
    # Section 3: TNML models
    # Section 4: External baselines (MLP, XGBoost, GP, Base)
    section1 = ["N-MPO2", "G-MPO2", "G-Ring"]
    section2 = ["N-CPDA", "G-CPDA", "TEMPO"]
    section3 = ["TNML_P", "TNML_F"]
    section4 = ["MLP", "XGBoost", "GP", "Base"] if include_external else ["Base"]
    
    all_sections = [section1, section2, section3, section4]
    
    n_reg = len(reg_ds)
    n_class = len(class_ds)
    
    # Build column spec
    col_spec = "l" + "c" * n_reg + " " + "c" * n_class
    
    # Start table
    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]
    
    # Header with Regression / Classification
    lines.append(f"& \\multicolumn{{{n_reg}}}{{c}}{{Regression}}")
    lines.append(f"& \\multicolumn{{{n_class}}}{{c}}{{Classification}} \\\\")
    
    # cmidrule for header sections
    cmidrule_left = f"\\cmidrule(r{{0.7em}}){{1-{n_reg + 1}}}"
    cmidrule_right = f"\\cmidrule(l{{0.7em}}){{{n_reg + 2}-{n_reg + n_class + 1}}}"
    lines.append(cmidrule_left)
    lines.append(cmidrule_right)
    
    # Dataset codes header
    reg_codes = [DATASET_INFO[d][0] for d in reg_ds]
    class_codes = [DATASET_INFO[d][0] for d in class_ds]
    lines.append("& " + " & ".join(reg_codes) + " & " + " & ".join(class_codes) + r" \\")
    
    def get_result_for_unified(model_key: str, dataset: str):
        """Get result for a model, handling TNML special case."""
        # Handle TNML models - use best trainer from config
        if model_key in ("TNML_P", "TNML_F"):
            best_trainer = tnml_best_configs.get(model_key, {}).get(dataset)
            if best_trainer:
                prefix = {"ntn": "N-", "gtn": "G-", "dmrg": "D-"}.get(best_trainer, "G-")
                key = (f"{prefix}{model_key}", dataset)
                if key in combined_results and combined_results[key].n_seeds > 0:
                    return combined_results[key]
            # Fallback: try all trainers and pick best
            best_res, best_mean = None, float('-inf')
            for prefix in ["N-", "G-", "D-"]:
                key = (f"{prefix}{model_key}", dataset)
                if key in combined_results and combined_results[key].n_seeds > 0:
                    if combined_results[key].mean_test_quality > best_mean:
                        best_mean = combined_results[key].mean_test_quality
                        best_res = combined_results[key]
            return best_res
        
        # Handle TEMPO (CPD-S)
        if model_key == "TEMPO":
            best_res, best_mean = None, float('-inf')
            for var in ["CPD-S", "CPD-S-TypeI"]:
                for prefix in ["N-", "G-"]:
                    key = (f"{prefix}{var}", dataset)
                    if key in combined_results and combined_results[key].n_seeds > 0:
                        if combined_results[key].mean_test_quality > best_mean:
                            best_mean = combined_results[key].mean_test_quality
                            best_res = combined_results[key]
            return best_res
        
        # Handle prefixed models (N-MPO2, G-MPO2, etc.)
        if model_key.startswith(("N-", "G-")):
            prefix = model_key[:2]
            base_model = model_key[2:]
            
            if base_model == "MPO2":
                variants = ALL_MODE_MPO2_VARIANTS
            else:
                variants = MODEL_GROUPS.get(base_model, [base_model])
            
            best_res, best_mean = None, float('-inf')
            for var in variants:
                key = (f"{prefix}{var}", dataset)
                if key in combined_results and combined_results[key].n_seeds > 0:
                    if combined_results[key].mean_test_quality > best_mean:
                        best_mean = combined_results[key].mean_test_quality
                        best_res = combined_results[key]
            return best_res
        
        # Direct lookup for external models
        key = (model_key, dataset)
        if key in combined_results and combined_results[key].n_seeds > 0:
            return combined_results[key]
        return None
    
    # All TN models for highlighting
    all_models = [m for section in all_sections for m in section]
    tn_models = [m for m in all_models if m not in EXTERNAL_MODEL_ORDER and m != "Base"]
    
    def find_best_overall(datasets):
        best = {}
        for ds in datasets:
            vals = [get_result_for_unified(m, ds).mean_test_quality 
                    for m in all_models if m != "Base" and get_result_for_unified(m, ds)]
            best[ds] = max(vals) if vals else float('-inf')
        return best
    
    def find_best_tn(datasets):
        best = {}
        for ds in datasets:
            vals = [get_result_for_unified(m, ds).mean_test_quality 
                    for m in tn_models if get_result_for_unified(m, ds)]
            best[ds] = max(vals) if vals else float('-inf')
        return best
    
    best_overall_r = find_best_overall(reg_ds)
    best_tn_r = find_best_tn(reg_ds)
    best_overall_c = find_best_overall(class_ds)
    best_tn_c = find_best_tn(class_ds)
    
    def add_model_row(m: str):
        """Add rows for a single model."""
        if m == "Base":
            # Handle baseline separately
            bs = get_baseline_results(test_outputs_dir, reg_ds + class_ds)
            base_vals = []
            for d in reg_ds + class_ds:
                if d in bs:
                    base_vals.append(f"{bs[d]*100:.2f}")
                else:
                    base_vals.append("--")
            lines.append(r"\textbf{Base} & " + " & ".join(base_vals) + r" \\")
            return
        
        model_latex = ALL_MODE_LATEX_NAMES.get(m, f"\\textbf{{{m}}}")
        is_tn_model = m in tn_models
        
        means = []
        stds = []
        has_any_result = False
        
        # Process regression datasets
        for d in reg_ds:
            res = get_result_for_unified(m, d)
            if not res:
                means.append("--")
                stds.append("--")
            else:
                has_any_result = True
                is_best_overall = abs(res.mean_test_quality - best_overall_r[d]) < 1e-9
                is_best_tn = is_tn_model and abs(res.mean_test_quality - best_tn_r[d]) < 1e-9
                means.append(format_mean_value(res.mean_test_quality, is_best_overall, is_best_tn))
                stds.append(f"$\\pm${res.std_test_quality*100:.2f}")
        
        # Process classification datasets
        for d in class_ds:
            res = get_result_for_unified(m, d)
            if not res:
                means.append("--")
                stds.append("--")
            else:
                has_any_result = True
                is_best_overall = abs(res.mean_test_quality - best_overall_c[d]) < 1e-9
                is_best_tn = is_tn_model and abs(res.mean_test_quality - best_tn_c[d]) < 1e-9
                means.append(format_mean_value(res.mean_test_quality, is_best_overall, is_best_tn))
                stds.append(f"$\\pm${res.std_test_quality*100:.2f}")
        
        # Mean row
        lines.append(f"{model_latex} & " + " & ".join(means) + r" \\")
        
        # Std row (only if has non-zero stds and has results)
        has_nonzero_std = any(s not in ("--", "", "$\\pm$0.00") for s in stds)
        if has_nonzero_std and has_any_result:
            lines.append(" & " + " & ".join(stds) + r" \\")
    
    # Generate rows for each section
    for section_idx, section in enumerate(all_sections):
        for model_idx, m in enumerate(section):
            add_model_row(m)
            
            # Add cmidrule after each model (except last in section)
            is_last_in_section = (model_idx == len(section) - 1)
            is_last_section = (section_idx == len(all_sections) - 1)
            
            if not (is_last_section and is_last_in_section):
                lines.append(f"{cmidrule_left} {cmidrule_right}")
            
            # Add double cmidrule between sections
            if is_last_in_section and not is_last_section:
                lines.append(f"{cmidrule_left} {cmidrule_right}")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-outputs-dir", type=Path, default=Path("test_outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper_scripts/tables"))
    parser.add_argument("--conf-dir", type=Path, default=Path("conf/best_conf"),
                        help="Directory containing best config YAML files")
    parser.add_argument("--trainer", choices=["ntn", "gtn", "both"], default="both")
    parser.add_argument("--combined", action="store_true")
    parser.add_argument("--all", action="store_true",
                        help="Generate table with both NTN and GTN, prefixed as N-/G-")
    parser.add_argument("--unified", action="store_true",
                        help="Generate unified table with TNML using best trainer per dataset")
    parser.add_argument("--use-val-loss", action="store_true")
    parser.add_argument("--external-csv-dir", type=Path, default=Path("oldResults"),
                        help="Directory containing external baseline CSVs (test_results_mlp.csv, etc.)")
    parser.add_argument("--exclude-external", action="store_true",
                        help="Exclude external baseline models (MLP, XGBoost, GP) from the tables")
    parser.add_argument("--average-column", action="store_true",
                        help="Add an Average column to the table")
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    external_results = {}
    cpd_s_results = {}
    if args.external_csv_dir:
        all_external = load_external_csv_results(args.external_csv_dir)
        for key, val in all_external.items():
            if key[0] in ("CPD-S", "CPD-S-TypeI"):
                cpd_s_results[key] = val
            else:
                external_results[key] = val
    
    include_external = not args.exclude_external
    
    # Handle unified table generation
    if args.unified:
        ntn_res = collect_results(args.test_outputs_dir, "ntn", args.use_val_loss)
        ntn_res.update(cpd_s_results)
        gtn_res = collect_results(args.test_outputs_dir, "gtn", args.use_val_loss)
        gtn_res.update(cpd_s_results)
        dmrg_res = collect_results(args.test_outputs_dir, "dmrg", args.use_val_loss)
        
        if include_external:
            ntn_res.update(external_results)
            gtn_res.update(external_results)
        
        # Load TNML best configs
        tnml_best_configs = load_tnml_best_configs(args.conf_dir)
        
        tbl = generate_unified_table(
            ntn_res, gtn_res, dmrg_res,
            REGRESSION_DATASETS, CLASSIFICATION_DATASETS,
            args.test_outputs_dir, tnml_best_configs, include_external
        )
        with open(args.output_dir / "unified_table.tex", "w") as f:
            f.write(tbl)
        print(f"Generated: {args.output_dir / 'unified_table.tex'}")
        return
    
    if getattr(args, 'all'):
        ntn_res = collect_results(args.test_outputs_dir, "ntn", args.use_val_loss)
        ntn_res.update(cpd_s_results)
        gtn_res = collect_results(args.test_outputs_dir, "gtn", args.use_val_loss)
        gtn_res.update(cpd_s_results)
        if include_external:
            ntn_res.update(external_results)
            gtn_res.update(external_results)
        tbl = generate_all_table(ntn_res, gtn_res, REGRESSION_DATASETS, CLASSIFICATION_DATASETS, 
                                  args.test_outputs_dir, include_external, show_avg=args.average_column)
        with open(args.output_dir / "combined_all.tex", "w") as f:
            f.write(tbl)
        return
    
    trainers = ["ntn", "gtn"] if args.trainer == "both" else [args.trainer]
    
    for tr in trainers:
        res = collect_results(args.test_outputs_dir, tr, args.use_val_loss)
        res.update(cpd_s_results)
        if include_external:
            res.update(external_results)
        if args.combined:
            tbl = generate_combined_table(res, REGRESSION_DATASETS, CLASSIFICATION_DATASETS, tr, args.test_outputs_dir, include_external, show_avg=args.average_column)
            with open(args.output_dir / f"combined_{tr}.tex", "w") as f: f.write(tbl)
        else:
            for task, ds_list in [("classification", CLASSIFICATION_DATASETS), ("regression", REGRESSION_DATASETS)]:
                tbl = generate_table(res, ds_list, task, tr, args.test_outputs_dir, include_external, show_avg=args.average_column)
                if tbl:
                    with open(args.output_dir / f"{task}_{tr}.tex", "w") as f: f.write(tbl)

if __name__ == "__main__":
    main()
