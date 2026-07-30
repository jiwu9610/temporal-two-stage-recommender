"""Cross-category summary of the Evaluation & Calibration results.

Reads results/calibration/{cat}_{variant}_test.json for every category that
has one and emits a compact markdown table + results/calibration/summary.json.

    python -m scripts.evaluation.summarize_calibration [--variant C2_k500_recent]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "calibration"
CATEGORIES = ("All_Beauty", "Video_Games", "Books", "Electronics")


def summarize(variant: str, results_dir: Path = RESULTS_DIR) -> dict:
    rows = []
    for cat in CATEGORIES:
        path = results_dir / f"{cat}_{variant}_test.json"
        if not path.exists():
            continue
        with open(path) as f:
            rep = json.load(f)
        cal = rep["calibration_test"]
        met = rep["metrics_test"]
        frozen = rep["frozen"]
        raw_o = met["raw"]["overall"]
        fin_o = met["final_with_thresholds"]["overall"]
        rows.append({
            "category": cat,
            "chosen_calibrator": frozen["chosen_calibrator"],
            "platt_a": frozen["platt"]["a"],
            "platt_b": frozen["platt"]["b"],
            "cohort_thresholds": frozen["cohort_thresholds"],
            "ece_raw": cal["raw"]["ece"],
            "ece_prior": cal["prior"]["ece"],
            "ece_platt": cal["platt"]["ece"],
            "ece_final": cal["final_with_thresholds"]["ece"],
            "ece_by_cohort_raw": cal["raw"]["by_user_cohort"],
            "ece_by_cohort_final":
                cal["final_with_thresholds"]["by_user_cohort"],
            "gauc": raw_o["gauc"],
            "auc": raw_o["auc"],
            "global_auc": raw_o["global_auc"],
            "mrr": raw_o["mrr"],
            "ndcg@10": raw_o["ndcg@10"],
            "recall@100_overall_raw": raw_o["recall@100_overall"],
            "recall@100_overall_final": fin_o["recall@100_overall"],
            "recall@10_overall_raw": raw_o["recall@10_overall"],
            "recall@10_overall_final": fin_o["recall@10_overall"],
            "success": rep["success_criteria"],
            "anchor": rep.get("consistency_anchor"),
        })
    return {"variant": variant, "categories": rows}


def to_markdown(summary: dict) -> str:
    lines = [
        "| 类目 | 校准器 | ECE raw→final | gAUC | AUC | MRR | nDCG@10 "
        "| R@100 raw→final | 判据 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["categories"]:
        s = r["success"]
        ok = ("PASS" if s["overall_ece_down"]
              and s["no_regression_recall@100"] and s["no_regression_gauc"]
              else "check")
        lines.append(
            f"| {r['category']} | {r['chosen_calibrator']} "
            f"| {r['ece_raw']:.4g}→{r['ece_final']:.4g} "
            f"| {r['gauc']:.4f} | {r['auc']:.4f} | {r['mrr']:.4f} "
            f"| {r['ndcg@10']:.4f} "
            f"| {r['recall@100_overall_raw']:.4f}→"
            f"{r['recall@100_overall_final']:.4f} | {ok} |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="C2_k500_recent")
    args = p.parse_args(argv)
    summary = summarize(args.variant)
    out = RESULTS_DIR / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(to_markdown(summary))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
