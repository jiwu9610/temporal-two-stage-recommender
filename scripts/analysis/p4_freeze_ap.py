"""P4.4 joint freeze, regenerated from the ALL-POSITIVE Stage B reports only.

Pure read: no training, no test read. For every category, reads every
results/phase3a/{cat}_stageB_{V}_ap.json present (V in A0..A5), applies the
PRE-DECLARED rule and writes results/p4_freeze/frozen_config_ap.json with
full provenance (source paths, sha256 of each report, code_commit and
label_mode as recorded inside each report, every metric the rule used).

Decision rule (pre-declared):
  reference = A0
  eligible  = candidates V != A0 whose model_selection R@10 >= A0's R@10
              (no-regression guardrail, zero tolerance)
  winner    = eligible V with the highest model_selection R@100 macro,
              provided it exceeds A0's R@100; otherwise A0.
Ties on R@100 (within `tie_tol`) are broken by R@10, then by the smaller
candidate table (rows) -- recorded in the output when they fire.

    python -m scripts.analysis.p4_freeze_ap [--tie-tol 0.0005] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STAGEB = REPO / "results" / "phase3a"
STAGEA = REPO / "results" / "phase2_temporal"     # builder reports: n_rows per snapshot
OUT = REPO / "results" / "p4_freeze" / "frozen_config_ap.json"
CFG_SRC = REPO / "configs" / "p4_stageA"
CFG_OUT = REPO / "configs" / "p4_frozen_ap"
CATS = ("Books", "Electronics", "Video_Games", "All_Beauty")
CONFIGS = ("A0", "A1", "A2", "A3", "A4", "A5")
RULE = ("primary = model_selection Recall@100 macro (all_positive frame); "
        "guardrail = Recall@10 >= A0 (zero tolerance); winner must beat A0 on "
        "Recall@100 else A0; ties within tie_tol on R@100 -> higher R@10 -> fewer rows")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rows(cat: str, v: str):
    """Candidate rows (ranker_train + model_selection) from the all-positive
    Stage A builder report -- the third tie-breaker (smaller table wins)."""
    p = STAGEA / f"{cat}_candidates_report_{v}_ap.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    missing = [s for s in ("ranker_train", "model_selection") if s not in r["snapshots"]]
    if missing:
        raise RuntimeError(
            f"{p}: snapshots {missing} missing -- the Stage-A report was "
            f"overwritten by a later test-only build (the builder now merges "
            f"instead of clobbering; restore the Stage-A entries before "
            f"freezing). Refusing to record n_rows=0.")
    return int(sum(r["snapshots"][s]["n_rows"] for s in ("ranker_train", "model_selection")))


def _load(cat: str) -> dict:
    arms = {}
    for v in CONFIGS:
        p = STAGEB / f"{cat}_stageB_{v}_ap.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        if r.get("label_mode") != "all_positive":
            raise RuntimeError(f"{p}: label_mode={r.get('label_mode')!r}, expected all_positive")
        if r.get("variant") != f"{v}_ap":
            raise RuntimeError(f"{p}: variant={r.get('variant')!r}, expected {v}_ap")
        m = r["model_selection_eval"]["overall"]
        arms[v] = {
            "source": str(p.relative_to(REPO)),
            "sha256": _sha256(p),
            "code_commit": r.get("code_commit"),
            "label_mode": r.get("label_mode"),
            "stage": r.get("stage"),
            "chosen": r["selection"]["chosen"],
            "R@10": float(m["Recall@10"]),
            "R@50": float(m["Recall@50"]),
            "R@100": float(m["Recall@100"]),
            "ceiling_retrieved": float(r["model_selection_eval"]["candidate_ceiling_retrieved_coverage"]),
            "n_rows": _rows(cat, v),
        }
    return arms


def decide(arms: dict, tie_tol: float) -> dict:
    if "A0" not in arms:
        raise RuntimeError("A0 reference missing")
    a0 = arms["A0"]
    rows = []
    for v, a in arms.items():
        if v == "A0":
            continue
        guard = a["R@10"] >= a0["R@10"]
        beats = a["R@100"] > a0["R@100"]
        rows.append({"config": v, "R@10": a["R@10"], "R@100": a["R@100"],
                     "n_rows": a.get("n_rows"),
                     "dR@10_vs_A0": a["R@10"] / a0["R@10"] - 1,
                     "dR@100_vs_A0": a["R@100"] / a0["R@100"] - 1,
                     "guardrail_pass": guard, "beats_A0_R@100": beats,
                     "eligible": guard and beats})
    elig = [r for r in rows if r["eligible"]]
    tie_note = None
    if not elig:
        winner = "A0"
    else:
        best = max(r["R@100"] for r in elig)
        top = [r for r in elig if best - r["R@100"] <= tie_tol]
        if len(top) > 1:
            # declared order: higher R@10, then fewer candidate rows, then name
            top.sort(key=lambda r: (-r["R@10"],
                                    r["n_rows"] if r.get("n_rows") is not None else float("inf"),
                                    r["config"]))
            tie_note = (f"R@100 tie within {tie_tol}: "
                        + ", ".join(f"{r['config']}={r['R@100']:.5f} (R@10 {r['R@10']:.5f}, rows {r.get('n_rows')})" for r in top)
                        + f" -> {top[0]['config']}")
        winner = top[0]["config"]
    return {"winner": winner, "table": rows, "tie_note": tie_note}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tie-tol", type=float, default=0.0005)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    payload = {
        "phase": "P4.4 joint freeze -- ALL-POSITIVE rebuild (supersedes the first-positive freeze now archived at archive/results_invalidated_first_positive/frozen_config_first_positive.json)",
        "freeze_history": [
            "controlling pre-test freeze recorded 2026-08-22, before the one-shot locked-test read of 2026-08-23; winners Books A3 / Electronics A0 / Video_Games A5 / All_Beauty A1",
            "later regenerations changed provenance fields only (row-count tie-breaker, frozen_ranker recipe, restored Stage-A row counts) -- decisions identical",
        ],
        "frozen_utc": datetime.now(tz=timezone.utc).isoformat(),
        "decision_rule": RULE,
        "tie_tol": args.tie_tol,
        "inputs_frame": "model_selection snapshot only; groundtruth_test never read",
        "categories": {},
    }
    for cat in CATS:
        arms = _load(cat)
        d = decide(arms, args.tie_tol)
        w = d["winner"]
        cfg_path = CFG_SRC / f"{cat}_{w}.json"
        retrieval_cfg = json.loads(cfg_path.read_text())
        payload["categories"][cat] = {
            "winner": w, "winner_variant": f"{w}_ap",
            # The EXACT Stage-B ranker recipe (arch/lr/epoch). The executed P5
            # of 2026-08-23 predated this field and re-ran the selection grid
            # on model_selection (held-out-legitimate; frozen object there =
            # retrieval config + selection procedure). Any future run must
            # consume this via train_temporal_ranker --frozen-ranker.
            "frozen_ranker": {"arch": arms[w]["chosen"].get("arch"),
                              "lr": arms[w]["chosen"].get("lr"),
                              "epoch": arms[w]["chosen"].get("best_epoch")},
            "retrieval_config_source": str(cfg_path.relative_to(REPO)),
            "retrieval_config": retrieval_cfg,
            "min_item_history": retrieval_cfg.get("min_item_history") or 5,
            "content_i2i": any(s.get("type") == "content_i2i" for s in retrieval_cfg.get("extra_sources", [])),
            "reference": arms["A0"],
            "arms": arms,
            "decision_table": d["table"],
            "tie_note": d["tie_note"],
        }
        print(f"{cat:12s} winner={w:3s} " + "  ".join(
            f"{r['config']}: R@100 {r['dR@100_vs_A0']:+.1%} R@10 {r['dR@10_vs_A0']:+.1%} "
            f"{'OK' if r['eligible'] else ('guard-fail' if not r['guardrail_pass'] else 'no-gain')}"
            for r in d["table"]) + (f"  [{d['tie_note']}]" if d["tie_note"] else ""))
    if args.dry_run:
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    CFG_OUT.mkdir(parents=True, exist_ok=True)
    for cat, c in payload["categories"].items():
        shutil.copy(REPO / c["retrieval_config_source"], CFG_OUT / f"{cat}.json")
    print(f"wrote {OUT.relative_to(REPO)} and {CFG_OUT.relative_to(REPO)}/")


if __name__ == "__main__":
    main()
