#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import statistics

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
    "TNML_P": ["TNML_P"],
    "TNML_F": ["TNML_F"],
    "Ring": ["BosonMPS"],
}

MODEL_ORDER = ["MPO2", "LMPO2", "MMPO2", "CPDA", "TNML_P", "TNML_F", "Ring"]

MODEL_LATEX_NAMES = {
    "MPO2": r"\textbf{(MPO)}$\bm{^2}$",
    "LMPO2": r"\textbf{(LMPO)}$\bm{^2}$",
    "MMPO2": r"\textbf{(MMPO)}$\bm{^2}$",
    "CPDA": r"\textbf{CPD-A}",
    "TNML_P": r"\textbf{TNML-P}",
    "TNML_F": r"\textbf{TNML-F}",
    "Ring": r"\textbf{Ring}",
}

CLASSIFICATION_DATASETS = ["iris", "hearth", "winequalityc", "breast", "adult", 
                           "bank", "wine", "car_evaluation", "student_dropout", "mushrooms"]

REGRESSION_DATASETS = ["realstate", "energy_efficiency", "concrete", "student_perf",
                       "obesity", "abalone", "seoulBike", "ai4i", "bike", "popularity"]


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
            try:
                with open(path) as f:
                    data = json.load(f)
                    raw_val = data.get("test_quality", float('nan'))
                    is_class = DATASET_INFO.get(ds, ["", "", ""])[2] == "Classification"
                    # Rule: Classification baseline already in %, Regression needs *100
                    baselines[ds] = raw_val if is_class else raw_val * 100
            except: continue
    return baselines


def get_best_variant_result(results, unified_model, dataset):
    variants = MODEL_GROUPS.get(unified_model, [unified_model])
    best_res, best_mean = None, float('-inf')
    for var in variants:
        key = (var, dataset)
        if key in results and results[key].n_seeds > 0:
            if results[key].mean_test_quality > best_mean:
                best_mean, best_res = results[key].mean_test_quality, results[key]
    return best_res


def get_filtered_model_order(trainer):
    order = [m for m in MODEL_ORDER if m != "Ring"]
    if trainer == "gtn": order.append("Ring")
    return order


def find_best_per_column(results, datasets, trainer):
    best_per_col = {}
    order = get_filtered_model_order(trainer)
    for ds in datasets:
        vals = [get_best_variant_result(results, m, ds).mean_test_quality 
                for m in order if get_best_variant_result(results, m, ds)]
        best_per_col[ds] = max(vals) if vals else float('-inf')
    return best_per_col


def find_second_best_per_column(results, datasets, best_per_col, trainer):
    second_best = {}
    order = get_filtered_model_order(trainer)
    for ds in datasets:
        vals = [get_best_variant_result(results, m, ds).mean_test_quality 
                for m in order if get_best_variant_result(results, m, ds)]
        below = [v for v in vals if v < best_per_col[ds] - 1e-9]
        second_best[ds] = max(below) if below else float('-inf')
    return second_best


def format_mean_value(mean, is_best, is_second):
    val_str = f"{mean * 100:.2f}"
    if is_best: return f"\\underline{{\\textbf{{{val_str}}}}}"
    if is_second: return f"\\underline{{{val_str}}}"
    return val_str


def generate_table(results, datasets, task, trainer, test_outputs_dir):
    order = get_filtered_model_order(trainer)
    datasets = [d for d in datasets if any(get_best_variant_result(results, m, d) for m in order)]
    if not datasets: return ""
    
    dataset_codes = [DATASET_INFO[d][0] for d in datasets]
    best_per_col = find_best_per_column(results, datasets, trainer)
    sec_per_col = find_second_best_per_column(results, datasets, best_per_col, trainer)
    
    row_avgs = {m: statistics.mean([get_best_variant_result(results, m, d).mean_test_quality 
                                    for d in datasets if get_best_variant_result(results, m, d)]) 
                for m in order if any(get_best_variant_result(results, m, d) for d in datasets)}
    
    v_avgs = [v for v in row_avgs.values() if v == v]
    b_avg = max(v_avgs) if v_avgs else float('-inf')
    s_avg = max([v for v in v_avgs if v < b_avg - 1e-9], default=float('-inf'))

    lines = [r"\begin{table}[!htbp]", r"\centering", r"\scriptsize", 
             r"\begin{tabular}{l" + "r" * (len(datasets) + 1) + "}", r"\toprule",
             " & " + " & ".join(dataset_codes) + r" & Avg \\", r"\midrule"]

    for m in order:
        model_latex = MODEL_LATEX_NAMES.get(m, f"\\textbf{{{m}}}")
        means, stds = [], []
        for d in datasets:
            res = get_best_variant_result(results, m, d)
            if not res: 
                means.append("--"); stds.append("--")
            else:
                means.append(format_mean_value(res.mean_test_quality, abs(res.mean_test_quality-best_per_col[d])<1e-9, 
                                               abs(res.mean_test_quality-sec_per_col[d])<1e-9))
                stds.append(f"$\\pm${res.std_test_quality*100:.2f}")
        
        avg = row_avgs.get(m, float('nan'))
        means.append(format_mean_value(avg, abs(avg-b_avg)<1e-9, abs(avg-s_avg)<1e-9) if avg==avg else "--")
        lines.append(f"{model_latex} & " + " & ".join(means) + r" \\")
        lines.append(" & " + " & ".join(stds + [""]) + r" \\")
        lines.append(r"\midrule")

    # Baseline Row
    bs = get_baseline_results(test_outputs_dir, datasets)
    base_row = [f"{bs[d]:.2f}" if d in bs else "--" for d in datasets]
    avg_b = statistics.mean([bs[d] for d in datasets if d in bs]) if bs else 0.0
    lines.append(r"\textbf{Mean} & " + " & ".join(base_row + [f"{avg_b:.2f}"]) + r" \\")
    
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_combined_table(results, reg_ds, class_ds, trainer, test_outputs_dir):
    order = get_filtered_model_order(trainer)
    reg_ds = [d for d in reg_ds if any(get_best_variant_result(results, m, d) for m in order)]
    class_ds = [d for d in class_ds if any(get_best_variant_result(results, m, d) for m in order)]
    if not reg_ds and not class_ds: return ""

    col_spec = "l" + ("r" * (len(reg_ds)+1) if reg_ds else "") + ("||" if reg_ds and class_ds else "") + ("r" * (len(class_ds)+1) if class_ds else "")
    lines = [r"\begin{table*}[!htbp]", r"\centering", r"\scriptsize", r"\begin{tabular}{" + col_spec + "}", r"\toprule"]
    
    h = [" & ".join([DATASET_INFO[d][0] for d in ds] + ["Avg"]) for ds in [reg_ds, class_ds] if ds]
    lines.append(" & " + " & ".join(h) + r" \\")
    lines.append(r"\midrule")

    # Metrics prep
    best_r = find_best_per_column(results, reg_ds, trainer); sec_r = find_second_best_per_column(results, reg_ds, best_r, trainer)
    best_c = find_best_per_column(results, class_ds, trainer); sec_c = find_second_best_per_column(results, class_ds, best_c, trainer)
    
    for m in order:
        model_latex = MODEL_LATEX_NAMES.get(m, f"\\textbf{{{m}}}")
        means, stds = [], []
        for ds, best, sec in [(reg_ds, best_r, sec_r), (class_ds, best_c, sec_c)]:
            if not ds: continue
            cur_vals = []
            for d in ds:
                res = get_best_variant_result(results, m, d)
                if not res: means.append("--"); stds.append("--")
                else:
                    means.append(format_mean_value(res.mean_test_quality, abs(res.mean_test_quality-best[d])<1e-9, abs(res.mean_test_quality-sec[d])<1e-9))
                    stds.append(f"$\\pm${res.std_test_quality*100:.2f}"); cur_vals.append(res.mean_test_quality)
            
            avg = statistics.mean(cur_vals) if cur_vals else float('nan')
            means.append(f"{avg*100:.2f}" if avg==avg else "--"); stds.append("") # Simplified avg bolding for combined space
        
        lines.append(f"{model_latex} & " + " & ".join(means) + r" \\")
        lines.append(" & " + " & ".join(stds) + r" \\")
        lines.append(r"\midrule")

    # Baseline Row
    base_vals = []
    for ds in [reg_ds, class_ds]:
        if not ds: continue
        bs = get_baseline_results(test_outputs_dir, ds)
        base_vals.extend([f"{bs[d]:.2f}" if d in bs else "--" for d in ds])
        avg_b = statistics.mean([bs[d] for d in ds if d in bs]) if bs else 0.0
        base_vals.append(f"{avg_b:.2f}")
    lines.append(r"\textbf{Mean} & " + " & ".join(base_vals) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-outputs-dir", type=Path, default=Path("test_outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper_scripts/tables"))
    parser.add_argument("--trainer", choices=["ntn", "gtn", "both"], default="both")
    parser.add_argument("--combined", action="store_true")
    parser.add_argument("--use-val-loss", action="store_true")
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainers = ["ntn", "gtn"] if args.trainer == "both" else [args.trainer]
    
    for tr in trainers:
        res = collect_results(args.test_outputs_dir, tr, args.use_val_loss)
        if args.combined:
            tbl = generate_combined_table(res, REGRESSION_DATASETS, CLASSIFICATION_DATASETS, tr, args.test_outputs_dir)
            with open(args.output_dir / f"combined_{tr}.tex", "w") as f: f.write(tbl)
        else:
            for task, ds_list in [("classification", CLASSIFICATION_DATASETS), ("regression", REGRESSION_DATASETS)]:
                tbl = generate_table(res, ds_list, task, tr, args.test_outputs_dir)
                if tbl:
                    with open(args.output_dir / f"{task}_{tr}.tex", "w") as f: f.write(tbl)

if __name__ == "__main__":
    main()
