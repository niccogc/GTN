#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import Optional
import math


def is_valid_number(val) -> bool:
    if val is None:
        return False
    if isinstance(val, float):
        return not (math.isinf(val) or math.isnan(val))
    return True


def find_best_epoch(metrics_log: list) -> Optional[dict]:
    if not metrics_log:
        return None

    best = None
    for entry in metrics_log:
        vq = entry.get("val_quality")
        if not is_valid_number(vq):
            continue
        if best is None or vq > best.get("val_quality"):
            best = entry

    return best


def check_result(data: dict) -> dict:
    metrics_log = data.get("metrics_log", [])

    if not metrics_log:
        return {"ok": True}

    max_epoch = max(e.get("epoch", 0) for e in metrics_log)
    if max_epoch == 0:
        return {"ok": True}

    best = find_best_epoch(metrics_log)
    if best is None:
        return {"ok": True}

    current_vq = data.get("val_quality")
    best_vq = best["val_quality"]

    current_valid = is_valid_number(current_vq)

    if current_valid:
        if current_vq >= best_vq:
            return {"ok": True}

    issue_type = "null/inf" if not current_valid else "worse_than_log"

    return {
        "ok": False,
        "issue": f"[{issue_type}] val_quality={current_vq} but best in log is {best_vq} at epoch {best['epoch']}",
        "fix": {
            "train_loss": best.get("train_loss"),
            "train_quality": best.get("train_quality"),
            "val_loss": best.get("val_loss"),
            "val_quality": best.get("val_quality"),
            "best_epoch": best.get("epoch"),
        },
    }


def process_file(path: Path, fix: bool = False, verbose: bool = False) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        return {"path": str(path), "error": str(e)}

    result = check_result(data)

    if result["ok"]:
        return {"path": str(path), "status": "ok"}

    status = {
        "path": str(path),
        "status": "mismatch",
        "issue": result["issue"],
    }

    if fix:
        for k, v in result["fix"].items():
            data[k] = v

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        status["fixed"] = True

    return status


def main():
    parser = argparse.ArgumentParser(description="Verify/fix results.json val_quality")
    parser.add_argument("--outputs", default="outputs", help="Outputs directory")
    parser.add_argument("--fix", action="store_true", help="Apply fixes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all files")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs)
    if not outputs_dir.exists():
        print(f"ERROR: {outputs_dir} not exist")
        return 1

    files = list(outputs_dir.rglob("results.json"))
    print(f"Found {len(files)} results.json files")

    ok_count = 0
    mismatch_count = 0
    error_count = 0
    fixed_count = 0

    for path in files:
        result = process_file(path, fix=args.fix, verbose=args.verbose)

        if "error" in result:
            error_count += 1
            print(f"ERROR: {result['path']}: {result['error']}")
        elif result["status"] == "ok":
            ok_count += 1
            if args.verbose:
                print(f"OK: {result['path']}")
        else:
            mismatch_count += 1
            print(f"MISMATCH: {result['path']}")
            print(f"  {result['issue']}")
            if result.get("fixed"):
                fixed_count += 1
                print(f"  FIXED!")

    print()
    print("=" * 50)
    print(f"Total:     {len(files)}")
    print(f"OK:        {ok_count}")
    print(f"Mismatch:  {mismatch_count}")
    print(f"Errors:    {error_count}")
    if args.fix:
        print(f"Fixed:     {fixed_count}")

    return 0 if mismatch_count == 0 else 1


if __name__ == "__main__":
    exit(main())
