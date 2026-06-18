#!/usr/bin/env python3
"""
Parse ablation study results from outputs/ folder and generate LaTeX tables.
Reports best val_quality across all hyperparameter configurations for each model.
Each model (including TypeI/TypeII variants) gets its own row.
"""
import argparse
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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

# All models for ablation (each variant is its own row)
# TypeI = Type I, default (no TypeI suffix) = Type II
ABLATION_MODELS = [
    "MPO2",
    "MPO2TypeI", 
    "LMPO2",
    "LMPO2TypeI",
    "MMPO2",
    "MMPO2TypeI",
    "CPDA",
    "CPDATypeI",
    "TNML_P",
    "TNML_F",
    "BosonMPS",
]

# Model display names for LaTeX
MODEL_LATEX_NAMES = {
    "MPO2": r"\textbf{(MPO)}$\bm{^2}$ \textbf{II}",
    "MPO2TypeI": r"\textbf{(MPO)}$\bm{^2}$ \textbf{I}",
    "LMPO2": r"\textbf{(LMPO)}$\bm{^2}$ \textbf{II}",
    "LMPO2TypeI": r"\textbf{(LMPO)}$\bm{^2}$ \textbf{I}",
    "MMPO2": r"\textbf{(MMPO)}$\bm{^2}$ \textbf{II}",
    "MMPO2TypeI": r"\textbf{(MMPO)}$\bm{^2}$ \textbf{I}",
    "CPDA": r"\textbf{CPD-A} \textbf{II}",
    "CPDATypeI": r"\textbf{CPD-A} \textbf{I}",
    "TNML_P": r"\textbf{TNML-P}",
    "TNML_F": r"\textbf{TNML-F}",
    "BosonMPS": r"\textbf{Ring}",
}

# N-/G- prefixed names for combined table
ALL_MODE_LATEX_NAMES = {
    "N-MPO2": r"\textbf{N-(MPO)}$\bm{^2}$ \textbf{II}",
    "G-MPO2": r"\textbf{G-(MPO)}$\bm{^2}$ \textbf{II}",
    "N-MPO2TypeI": r"\textbf{N-(MPO)}$\bm{^2}$ \textbf{I}",
    "G-MPO2TypeI": r"\textbf{G-(MPO)}$\bm{^2}$ \textbf{I}",
    "N-LMPO2": r"\textbf{N-(LMPO)}$\bm{^2}$ \textbf{II}",
    "G-LMPO2": r"\textbf{G-(LMPO)}$\bm{^2}$ \textbf{II}",
    "N-LMPO2TypeI": r"\textbf{N-(LMPO)}$\bm{^2}$ \textbf{I}",
    "G-LMPO2TypeI": r"\textbf{G-(LMPO)}$\bm{^2}$ \textbf{I}",
    "N-MMPO2": r"\textbf{N-(MMPO)}$\bm{^2}$ \textbf{II}",
    "G-MMPO2": r"\textbf{G-(MMPO)}$\bm{^2}$ \textbf{II}",
    "N-MMPO2TypeI": r"\textbf{N-(MMPO)}$\bm{^2}$ \textbf{I}",
    "G-MMPO2TypeI": r"\textbf{G-(MMPO)}$\bm{^2}$ \textbf{I}",
    "N-CPDA": r"\textbf{N-CPD-A} \textbf{II}",
    "G-CPDA": r"\textbf{G-CPD-A} \textbf{II}",
    "N-CPDATypeI": r"\textbf{N-CPD-A} \textbf{I}",
    "G-CPDATypeI": r"\textbf{G-CPD-A} \textbf{I}",
    "N-TNML_P": r"\textbf{N-TNML-P}",
    "G-TNML_P": r"\textbf{G-TNML-P}",
    "N-TNML_F": r"\textbf{N-TNML-F}",
    "G-TNML_F": r"\textbf{G-TNML-F}",
    "G-BosonMPS": r"\textbf{G-Ring}",
}

# Same datasets as test_results script
CLASSIFICATION_DATASETS = ["iris", "hearth", "winequalityc", "breast", "adult",
                           "bank", "wine", "car_evaluation", "student_dropout", "mushrooms"]
REGRESSION_DATASETS = ["realstate", "energy_efficiency", "concrete", "student_perf",
                       "obesity", "abalone", "seoulBike", "ai4i", "bike", "popularity"]


@dataclass
class RunResult:
    """Result from a single run (one hyperparameter configuration)."""
    best_val_quality: float
    best_epoch: int
    config_name: str  # e.g., "L3_bd4_seed42"


@dataclass 
class ModelDatasetResult:
    """Aggregated results for a model-dataset pair across all seeds."""
    model: str
    dataset: str
    trainer: str
    run_results: list[RunResult] = field(default_factory=list)
    
    @property
    def n_runs(self) -> int:
        return len(self.run_results)
    
    @property
    def mean_val_quality(self) -> float:
        if not self.run_results:
            return float('nan')
        return statistics.mean(r.best_val_quality for r in self.run_results)
    
    @property
    def std_val_quality(self) -> float:
        if len(self.run_results) < 2:
            return 0.0
        return statistics.stdev(r.best_val_quality for r in self.run_results)
    
    @property
    def best_val_quality(self) -> float:
        """Return the best val_quality across all runs."""
        if not self.run_results:
            return float('nan')
        return max(r.best_val_quality for r in self.run_results)


def parse_results_json(path: Path) -> Optional[RunResult]:
    """Parse a results.json file and extract best val_quality."""
    try:
        with open(path) as f:
            data = json.load(f)
        
        val_quality = data.get("val_quality")
        best_epoch = data.get("best_epoch", -1)
        
        if val_quality is None:
            return None
            
        return RunResult(
            best_val_quality=val_quality,
            best_epoch=best_epoch,
            config_name=path.parent.name
        )
    except Exception:
        return None


def extract_seed_from_config(config_name: str) -> Optional[int]:
    """Extract seed number from config name like 'L3_bd4_seed42'."""
    parts = config_name.split("_")
    for part in parts:
        if part.startswith("seed"):
            try:
                return int(part[4:])
            except ValueError:
                pass
    return None


def collect_results(outputs_dir: Path, trainer: str) -> dict[tuple[str, str], ModelDatasetResult]:
    """
    Collect results from outputs/{trainer}/{dataset}/{model_*}/{config}/results.json
    
    Groups by seed - for each seed, takes the best val_quality across all hyperparameters.
    """
    trainer_dir = outputs_dir / trainer
    if not trainer_dir.exists():
        return {}
    
    # Intermediate storage: (model, dataset) -> {seed: [RunResult]}
    seed_results: dict[tuple[str, str], dict[int, list[RunResult]]] = {}
    
    for dataset_dir in trainer_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        
        for model_config_dir in dataset_dir.iterdir():
            if not model_config_dir.is_dir():
                continue
            
            # Extract model name from dir like "MPO2_rg0.005_init0.1"
            model = model_config_dir.name.split("_rg")[0]
            if model not in ABLATION_MODELS:
                continue
                
            key = (model, dataset)
            if key not in seed_results:
                seed_results[key] = {}
            
            # Iterate over hyperparameter configurations
            for config_dir in model_config_dir.iterdir():
                if not config_dir.is_dir():
                    continue
                    
                res_file = config_dir / "results.json"
                if not res_file.exists():
                    continue
                    
                run_result = parse_results_json(res_file)
                if run_result is None:
                    continue
                
                seed = extract_seed_from_config(config_dir.name)
                if seed is None:
                    continue
                    
                if seed not in seed_results[key]:
                    seed_results[key][seed] = []
                seed_results[key][seed].append(run_result)
    
    # Now aggregate: for each seed, take the best val_quality
    results: dict[tuple[str, str], ModelDatasetResult] = {}
    
    for (model, dataset), seeds_dict in seed_results.items():
        key = (model, dataset)
        results[key] = ModelDatasetResult(model=model, dataset=dataset, trainer=trainer)
        
        for seed, runs in seeds_dict.items():
            if not runs:
                continue
            # Take the best run for this seed
            best_run = max(runs, key=lambda r: r.best_val_quality)
            results[key].run_results.append(best_run)
    
    return results


def get_baseline_results(outputs_dir: Path, datasets: list[str]) -> dict[str, float]:
    """Get mean baseline results (if available)."""
    # The ablation outputs don't have mean_baseline, return empty
    return {}


def get_available_datasets(results: dict, dataset_list: list[str]) -> list[str]:
    """Get datasets that have at least one model with results."""
    available = set()
    for (model, dataset) in results.keys():
        if dataset in dataset_list and results[(model, dataset)].n_runs > 0:
            available.add(dataset)
    return [d for d in dataset_list if d in available]


def find_best_per_column(results: dict, datasets: list[str], models: list[str]) -> dict[str, float]:
    """Find best mean val_quality per dataset across all models."""
    best_per_col = {}
    for ds in datasets:
        vals = []
        for m in models:
            key = (m, ds)
            if key in results and results[key].n_runs > 0:
                vals.append(results[key].mean_val_quality)
        best_per_col[ds] = max(vals) if vals else float('-inf')
    return best_per_col


def format_mean_value(mean: float, is_best: bool, is_second_best: bool = False) -> str:
    """Format a mean value, with bold for best and underline for second best."""
    val_str = f"{mean * 100:.2f}"
    if is_best:
        return f"\\textbf{{{val_str}}}"
    if is_second_best:
        return f"\\underline{{{val_str}}}"
    return val_str


def get_model_order(trainer: str) -> list[str]:
    """Get model order based on trainer (BosonMPS only for GTN)."""
    order = [m for m in ABLATION_MODELS if m != "BosonMPS"]
    if trainer == "gtn":
        order.append("BosonMPS")
    return order


def generate_table(results: dict, datasets: list[str], trainer: str, show_avg: bool = False) -> str:
    """Generate LaTeX table for single trainer."""
    models = get_model_order(trainer)
    datasets = get_available_datasets(results, datasets)
    if not datasets:
        return ""
    
    n_cols = len(datasets) + (1 if show_avg else 0)
    dataset_codes = [DATASET_INFO[d][0] for d in datasets]
    best_overall = find_best_per_column(results, datasets, models)
    
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{l" + "r" * n_cols + "}",
        r"\toprule"
    ]
    
    header = " & ".join(dataset_codes)
    if show_avg:
        header += r" & Avg"
    lines.append(header + r" \\")
    lines.append(r"\midrule")
    
    for m in models:
        model_latex = MODEL_LATEX_NAMES.get(m, f"\\textbf{{{m}}}")
        means, stds = [], []
        cur_vals = []
        
        for d in datasets:
            key = (m, d)
            if key not in results or results[key].n_runs == 0:
                means.append("--")
                stds.append("--")
            else:
                res = results[key]
                is_best = abs(res.mean_val_quality - best_overall[d]) < 1e-9
                means.append(format_mean_value(res.mean_val_quality, is_best))
                stds.append(f"$\\pm${res.std_val_quality*100:.2f}")
                cur_vals.append(res.mean_val_quality)
        
        if show_avg and cur_vals:
            avg = statistics.mean(cur_vals)
            means.append(f"{avg*100:.2f}")
        elif show_avg:
            means.append("--")
        
        lines.append(f"{model_latex} & " + " & ".join(means) + r" \\")
        has_nonzero_std = any(s not in ("--", "", "$\\pm$0.00") for s in stds)
        if has_nonzero_std:
            stds_extra = stds + ([""] if show_avg else [])
            lines.append(" & " + " & ".join(stds_extra) + r" \\")
        lines.append(r"\midrule")
    
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_combined_table(results: dict, reg_ds: list[str], class_ds: list[str], 
                           trainer: str, show_avg: bool = False) -> str:
    """Generate combined table with regression and classification datasets."""
    models = get_model_order(trainer)
    reg_ds = get_available_datasets(results, reg_ds)
    class_ds = get_available_datasets(results, class_ds)
    if not reg_ds and not class_ds:
        return ""
    
    def n_cols(ds):
        return len(ds) + (1 if show_avg else 0)
    
    col_spec = "l"
    if reg_ds:
        col_spec += "r" * n_cols(reg_ds)
    if reg_ds and class_ds:
        col_spec += "||"
    if class_ds:
        col_spec += "r" * n_cols(class_ds)
    
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering", 
        r"\scriptsize",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule"
    ]
    
    h = []
    for ds in [reg_ds, class_ds]:
        if not ds:
            continue
        parts = [DATASET_INFO[d][0] for d in ds]
        if show_avg:
            parts.append("Avg")
        h.append(" & ".join(parts))
    lines.append(" & " + " & ".join(h) + r" \\")
    lines.append(r"\midrule")
    
    best_overall_r = find_best_per_column(results, reg_ds, models)
    best_overall_c = find_best_per_column(results, class_ds, models)
    
    for m in models:
        model_latex = MODEL_LATEX_NAMES.get(m, f"\\textbf{{{m}}}")
        means, stds = [], []
        
        for ds, best_overall in [(reg_ds, best_overall_r), (class_ds, best_overall_c)]:
            if not ds:
                continue
            cur_vals = []
            for d in ds:
                key = (m, d)
                if key not in results or results[key].n_runs == 0:
                    means.append("--")
                    stds.append("--")
                else:
                    res = results[key]
                    is_best = abs(res.mean_val_quality - best_overall[d]) < 1e-9
                    means.append(format_mean_value(res.mean_val_quality, is_best))
                    stds.append(f"$\\pm${res.std_val_quality*100:.2f}")
                    cur_vals.append(res.mean_val_quality)
            
            if show_avg and cur_vals:
                avg = statistics.mean(cur_vals)
                means.append(f"{avg*100:.2f}")
                stds.append("")
            elif show_avg:
                means.append("--")
                stds.append("")
        
        lines.append(f"{model_latex} & " + " & ".join(means) + r" \\")
        has_nonzero_std = any(s not in ("--", "", "$\\pm$0.00") for s in stds)
        if has_nonzero_std:
            lines.append(" & " + " & ".join(stds) + r" \\")
        lines.append(r"\midrule")
    
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def get_all_mode_model_order() -> list[str]:
    """Get model order for combined N-/G- table."""
    order = []
    for m in ABLATION_MODELS:
        if m == "BosonMPS":
            # BosonMPS only for GTN
            order.append(f"G-{m}")
        else:
            order.append(f"N-{m}")
            order.append(f"G-{m}")
    return order


def generate_all_table(ntn_results: dict, gtn_results: dict, 
                       reg_ds: list[str], class_ds: list[str],
                       show_avg: bool = False) -> str:
    """Generate table with both NTN and GTN results, prefixed as N-/G-."""
    # Combine results with prefixes
    combined_results = {}
    for key, val in ntn_results.items():
        new_key = (f"N-{key[0]}", key[1])
        combined_results[new_key] = val
    for key, val in gtn_results.items():
        new_key = (f"G-{key[0]}", key[1])
        combined_results[new_key] = val
    
    # Filter datasets based on GTN availability (more complete)
    reg_ds = get_available_datasets(gtn_results, reg_ds)
    class_ds = get_available_datasets(gtn_results, class_ds)
    if not reg_ds and not class_ds:
        return ""
    
    models = get_all_mode_model_order()
    
    def n_cols(ds):
        return len(ds) + (1 if show_avg else 0)
    
    col_spec = "l"
    if reg_ds:
        col_spec += "r" * n_cols(reg_ds)
    if reg_ds and class_ds:
        col_spec += "||"
    if class_ds:
        col_spec += "r" * n_cols(class_ds)
    
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        r"\scriptsize", 
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule"
    ]
    
    h = []
    for ds in [reg_ds, class_ds]:
        if not ds:
            continue
        parts = [DATASET_INFO[d][0] for d in ds]
        if show_avg:
            parts.append("Avg")
        h.append(" & ".join(parts))
    lines.append(" & " + " & ".join(h) + r" \\")
    lines.append(r"\midrule")
    
    def find_best(datasets):
        best = {}
        for ds in datasets:
            vals = []
            for m in models:
                key = (m, ds)
                if key in combined_results and combined_results[key].n_runs > 0:
                    vals.append(combined_results[key].mean_val_quality)
            best[ds] = max(vals) if vals else float('-inf')
        return best
    
    best_overall_r = find_best(reg_ds)
    best_overall_c = find_best(class_ds)
    
    for m in models:
        model_latex = ALL_MODE_LATEX_NAMES.get(m, f"\\textbf{{{m}}}")
        means, stds = [], []
        
        for ds, best_overall in [(reg_ds, best_overall_r), (class_ds, best_overall_c)]:
            if not ds:
                continue
            cur_vals = []
            for d in ds:
                key = (m, d)
                if key not in combined_results or combined_results[key].n_runs == 0:
                    means.append("--")
                    stds.append("--")
                else:
                    res = combined_results[key]
                    is_best = abs(res.mean_val_quality - best_overall[d]) < 1e-9
                    means.append(format_mean_value(res.mean_val_quality, is_best))
                    stds.append(f"$\\pm${res.std_val_quality*100:.2f}")
                    cur_vals.append(res.mean_val_quality)
            
            if show_avg and cur_vals:
                avg = statistics.mean(cur_vals)
                means.append(f"{avg*100:.2f}")
                stds.append("")
            elif show_avg:
                means.append("--")
                stds.append("")
        
        lines.append(f"{model_latex} & " + " & ".join(means) + r" \\")
        has_nonzero_std = any(s not in ("--", "", "$\\pm$0.00") for s in stds)
        if has_nonzero_std:
            lines.append(" & " + " & ".join(stds) + r" \\")
        lines.append(r"\midrule")
    
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def _fmt_oob_pair(mean_val: float, std_val: float) -> str:
    """Format a (mean ± std) pair sharing the mean's exponent."""
    if mean_val == 0:
        return f"$(0.00 \\pm {std_val:.2e})$"
    exp = int(math.floor(math.log10(abs(mean_val))))
    scale = 10.0 ** exp
    scaled_mean = mean_val / scale
    scaled_std = std_val / scale
    return f"$({scaled_mean:.2f} \\pm {scaled_std:.2f})\\times 10^{{{exp}}}$"


def generate_unified_ablation_table(
    ntn_results: dict,
    gtn_results: dict,
    datasets: list[str],
    task_type: str,
) -> str:
    """Generate unified ablation table following same style as test --unified.

    Creates one table for either classification or regression with:
    - All model variants (I/II) shown as separate N-/G- rows
    - Section breaks with double midrules
    - Bold = best overall, underline = best among TNs
    """
    # Combine results with prefixes
    combined_results = {}
    for key, val in ntn_results.items():
        new_key = (f"N-{key[0]}", key[1])
        combined_results[new_key] = val
    for key, val in gtn_results.items():
        new_key = (f"G-{key[0]}", key[1])
        combined_results[new_key] = val

    datasets = get_available_datasets(gtn_results, datasets)
    if not datasets:
        return "", []

    oob_entries: list = []

    # Define model sections matching test --unified style
    section1 = [
        "N-MPO2", "G-MPO2", "N-MPO2TypeI", "G-MPO2TypeI",
        "N-LMPO2", "G-LMPO2", "N-LMPO2TypeI", "G-LMPO2TypeI",
        "N-MMPO2", "G-MMPO2", "N-MMPO2TypeI", "G-MMPO2TypeI",
        "G-BosonMPS",
    ]
    section2 = [
        "N-CPDA", "G-CPDA", "N-CPDATypeI", "G-CPDATypeI",
    ]
    section3 = [
        "N-TNML_P", "G-TNML_P", "N-TNML_F", "G-TNML_F",
    ]

    all_sections = [section1, section2, section3]

    n_cols = len(datasets)
    col_spec = "l" + "c" * n_cols

    task_title = "Classification" if task_type == "classification" else "Regression"
    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\small",
        f"\\caption{{Ablation study -- {task_title} (validation).}}",
        f"\\label{{tab:ablation_{task_type}}}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
    ]

    codes = [DATASET_INFO[d][0] for d in datasets]
    lines.append("& " + " & ".join(codes) + r" \\")
    lines.append(r"\midrule")

    def get_result(model_key: str, dataset: str):
        """Direct lookup for prefixed models."""
        key = (model_key, dataset)
        if key in combined_results and combined_results[key].n_runs > 0:
            return combined_results[key]
        return None

    all_models = [m for section in all_sections for m in section]
    # All models in ablation are TN models
    tn_models = list(all_models)

    def find_best_overall(ds_list):
        best = {}
        for ds in ds_list:
            vals = [get_result(m, ds).mean_val_quality
                    for m in all_models if get_result(m, ds)]
            best[ds] = max(vals) if vals else float('-inf')
        return best

    def find_best_tn(ds_list):
        best = {}
        for ds in ds_list:
            vals = [get_result(m, ds).mean_val_quality
                    for m in tn_models if get_result(m, ds)]
            best[ds] = max(vals) if vals else float('-inf')
        return best

    best_overall = find_best_overall(datasets)
    best_tn = find_best_tn(datasets)

    cmidrule = r"\midrule"

    def add_model_row(m: str):
        model_latex = ALL_MODE_LATEX_NAMES.get(m, f"\\textbf{{{m}}}")

        means = []
        stds = []
        has_any_result = False
        any_nonzero_std = False

        for d in datasets:
            res = get_result(m, d)
            if not res:
                means.append("--")
                stds.append("")
            else:
                has_any_result = True
                mean_scaled = res.mean_val_quality * 100
                std_scaled = res.std_val_quality * 100
                is_oob = abs(mean_scaled) > 100

                if is_oob:
                    oob_entries.append((m, d, mean_scaled, std_scaled))
                    means.append("F")
                    stds.append("")
                else:
                    is_best_overall = abs(res.mean_val_quality - best_overall[d]) < 1e-9
                    is_best_tn = abs(res.mean_val_quality - best_tn[d]) < 1e-9

                    val_str = f"{mean_scaled:.2f}"
                    if is_best_overall:
                        val_str = f"\\textbf{{{val_str}}}"
                    elif is_best_tn:
                        val_str = f"\\underline{{{val_str}}}"
                    means.append(val_str)

                    std_str = f"$\\pm${std_scaled:.2f}"
                    stds.append(std_str)
                    if std_str != "$\\pm$0.00":
                        any_nonzero_std = True

        lines.append(f"{model_latex} & " + " & ".join(means) + r" \\")

        if any_nonzero_std and has_any_result:
            lines.append(" & " + " & ".join(stds) + r" \\")

    for section_idx, section in enumerate(all_sections):
        for model_idx, m in enumerate(section):
            add_model_row(m)

            is_last_in_section = (model_idx == len(section) - 1)
            is_last_section = (section_idx == len(all_sections) - 1)

            if not (is_last_section and is_last_in_section):
                lines.append(cmidrule)

            if is_last_in_section and not is_last_section:
                lines.append(cmidrule)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines), oob_entries


def main():
    parser = argparse.ArgumentParser(description="Parse ablation study results and generate LaTeX tables")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"),
                        help="Directory containing ablation outputs")
    parser.add_argument("--output-dir", type=Path, default=Path("paper_scripts/tables"),
                        help="Directory to write output tables")
    parser.add_argument("--trainer", choices=["ntn", "gtn", "both"], default="both",
                        help="Which trainer results to include")
    parser.add_argument("--combined", action="store_true",
                        help="Generate combined regression/classification table")
    parser.add_argument("--all", action="store_true",
                        help="Generate table with both NTN and GTN, prefixed as N-/G-")
    parser.add_argument("--average-column", action="store_true",
                        help="Add an Average column to the table")
    parser.add_argument("--unified", action="store_true",
                        help="Generate unified table following test --unified style (separate tables per task)")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.unified:
        ntn_res = collect_results(args.outputs_dir, "ntn")
        gtn_res = collect_results(args.outputs_dir, "gtn")

        all_oob: list = []
        for task_type, datasets in [("classification", CLASSIFICATION_DATASETS),
                                     ("regression", REGRESSION_DATASETS)]:
            tbl, oob_entries = generate_unified_ablation_table(ntn_res, gtn_res, datasets, task_type)
            all_oob.extend(oob_entries)
            if tbl:
                out_path = args.output_dir / f"ablation_unified_{task_type}.tex"
                with open(out_path, "w") as f:
                    f.write(tbl)
                print(f"Generated: {out_path}")

        # Generate out-of-bound table if any extreme values found
        if all_oob:
            oob_models = sorted(set(e[0] for e in all_oob))
            oob_ds_set = set(e[1] for e in all_oob)
            oob_datasets = [d for d in REGRESSION_DATASETS + CLASSIFICATION_DATASETS if d in oob_ds_set]
            oob_lookup = {(m, d): (mv, sv) for m, d, mv, sv in all_oob}

            oob_lines = [
                r"\begin{table}[ht]",
                r"\centering",
                r"\small",
                r"\caption{Ablation out-of-bound results (values where $|\text{value} \times 100| > 100$).}",
                r"\label{tab:ablation_out_of_bound}",
                r"\begin{tabular}{l" + "c" * len(oob_models) + "}",
                r"\toprule",
            ]
            model_headers = [ALL_MODE_LATEX_NAMES.get(m, f"\\textbf{{{m}}}") for m in oob_models]
            oob_lines.append("& " + " & ".join(model_headers) + r" \\")
            oob_lines.append(r"\midrule")

            for d in oob_datasets:
                ds_code = DATASET_INFO[d][0]
                cells = []
                for m in oob_models:
                    entry = oob_lookup.get((m, d))
                    if entry:
                        mv, sv = entry
                        cells.append(_fmt_oob_pair(mv, sv))
                    else:
                        cells.append("--")
                oob_lines.append(f"{ds_code} & " + " & ".join(cells) + r" \\")
                oob_lines.append(r"\midrule")

            oob_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

            oob_file = args.output_dir / "ablation_out_of_bound.tex"
            with open(oob_file, "w") as f:
                f.write("\n".join(oob_lines))
            print(f"Generated: {oob_file}")
        return

    if getattr(args, 'all'):
        ntn_res = collect_results(args.outputs_dir, "ntn")
        gtn_res = collect_results(args.outputs_dir, "gtn")
        tbl = generate_all_table(ntn_res, gtn_res, REGRESSION_DATASETS, CLASSIFICATION_DATASETS,
                                  show_avg=args.average_column)
        with open(args.output_dir / "ablation_all.tex", "w") as f:
            f.write(tbl)
        print(f"Generated: {args.output_dir / 'ablation_all.tex'}")
        return

    trainers = ["ntn", "gtn"] if args.trainer == "both" else [args.trainer]

    for tr in trainers:
        res = collect_results(args.outputs_dir, tr)
        if args.combined:
            tbl = generate_combined_table(res, REGRESSION_DATASETS, CLASSIFICATION_DATASETS,
                                         tr, show_avg=args.average_column)
            out_path = args.output_dir / f"ablation_combined_{tr}.tex"
            with open(out_path, "w") as f:
                f.write(tbl)
            print(f"Generated: {out_path}")
        else:
            for task, ds_list in [("classification", CLASSIFICATION_DATASETS),
                                  ("regression", REGRESSION_DATASETS)]:
                tbl = generate_table(res, ds_list, tr, show_avg=args.average_column)
                if tbl:
                    out_path = args.output_dir / f"ablation_{task}_{tr}.tex"
                    with open(out_path, "w") as f:
                        f.write(tbl)
                    print(f"Generated: {out_path}")


if __name__ == "__main__":
    main()
