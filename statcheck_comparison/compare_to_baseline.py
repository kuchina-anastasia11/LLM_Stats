import argparse
import json
from collections import defaultdict
from pathlib import Path

STATCHECK_ENVS = {"apa", "non_apa"}  # области применимости statcheck


def load_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def index_by(records, key):
    return {r.get(key): r for r in records if r.get(key) is not None}


def normalize_test_type(t):
    if t is None:
        return None
    s = str(t).strip().lower()
    aliases = {
        "t": "t", "f": "F",
        "chi": "chi", "chi2": "chi", "chisq": "chi",
        "chi-square": "chi", "chi_square": "chi", "chi^2": "chi",
        "z": "z", "r": "r", "q": "Q",
    }
    return aliases.get(s)


def match_test(pred, gold, stat_tol=0.05):
    pt = normalize_test_type(pred.get("test_type"))
    gt = normalize_test_type(gold.get("test_type"))
    if pt is None or gt is None or pt != gt:
        return False
    pv = pred.get("statistic_value")
    gv = gold.get("statistic_value")
    if pv is None or gv is None:
        return False
    denom = max(abs(gv), 1.0)
    return abs(pv - gv) / denom <= stat_tol


def score_example(pred_tests, gold_tests):
    matched = set()
    tp = 0
    for p in pred_tests:
        for i, g in enumerate(gold_tests):
            if i in matched:
                continue
            if match_test(p, g):
                matched.add(i)
                tp += 1
                break
    fp = len(pred_tests) - tp
    fn = len(gold_tests) - len(matched)
    return tp, fp, fn, matched


def aggregate_by_env(gold_examples, pred_idx, applicable_only=False):
    by_env = defaultdict(lambda: {
        "n_examples": 0, "n_gold": 0,
        "tp": 0, "fp": 0, "fn": 0,
        "n_empty_pred_with_gold": 0,
    })
    for ex in gold_examples:
        env = ex.get("environment", "unknown")
        gold_tests = ex.get("tests") or []
        if applicable_only and env not in STATCHECK_ENVS:
            continue
        pred_rec = pred_idx.get(ex.get("example_id"))
        pred_tests = (pred_rec or {}).get("predicted_tests") or []
        tp, fp, fn, _ = score_example(pred_tests, gold_tests)
        by_env[env]["n_examples"] += 1
        by_env[env]["n_gold"] += len(gold_tests)
        by_env[env]["tp"] += tp
        by_env[env]["fp"] += fp
        by_env[env]["fn"] += fn
        if gold_tests and not pred_tests:
            by_env[env]["n_empty_pred_with_gold"] += 1
    rows = []
    for env, m in by_env.items():
        prec = m["tp"] / max(m["tp"] + m["fp"], 1)
        rec = m["tp"] / max(m["tp"] + m["fn"], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        rows.append({
            "environment": env,
            "n_examples": m["n_examples"],
            "n_gold": m["n_gold"],
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "n_empty_pred_with_gold": m["n_empty_pred_with_gold"],
            "empty_rate": round(
                m["n_empty_pred_with_gold"] / max(m["n_examples"], 1), 3
            ),
        })
    rows.sort(key=lambda r: r["environment"] or "")
    return rows


def aggregate_idr(gold_examples, pred_idx):
    by_err = defaultdict(lambda: {"n_pos": 0, "tp": 0, "n_neg": 0, "fp": 0})
    for ex in gold_examples:
        err = ex.get("error_type") or "none"
        gold_tests = ex.get("tests") or []
        pred_rec = pred_idx.get(ex.get("example_id")) or {}
        pred_tests = (pred_rec.get("verified_tests")
                      or pred_rec.get("predicted_tests")
                      or [])
        # Сработала ли проверка
        flagged = any(
            (p.get("p_consistency") == "inconsistent")
            or p.get("decision_error") is True
            or p.get("interpretation_consistency") == "inconsistent"
            for p in pred_tests
        )
        is_pos = ex.get("label_consistent") is False
        if is_pos:
            by_err[err]["n_pos"] += 1
            if flagged:
                by_err[err]["tp"] += 1
        else:
            by_err[err]["n_neg"] += 1
            if flagged:
                by_err[err]["fp"] += 1
    rows = []
    for err, m in by_err.items():
        idr = m["tp"] / max(m["n_pos"], 1) if m["n_pos"] else None
        far = m["fp"] / max(m["n_neg"], 1) if m["n_neg"] else None
        rows.append({
            "error_type": err,
            "n_pos": m["n_pos"], "tp": m["tp"], "idr": None if idr is None else round(idr, 3),
            "n_neg": m["n_neg"], "fp": m["fp"], "far": None if far is None else round(far, 3),
        })
    rows.sort(key=lambda r: r["error_type"] or "")
    return rows


def fmt_table(rows, columns):
    if not rows:
        return "(no rows)"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = "\n".join(
        "| " + " | ".join(str(r.get(c, "")) for c in columns) + " |"
        for r in rows
    )
    return "\n".join([header, sep, body])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--statcheck", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    gold = load_jsonl(args.gold)
    sc = load_jsonl(args.statcheck)
    bl = load_jsonl(args.baseline)
    sc_idx = index_by(sc, "example_id")
    bl_idx = index_by(bl, "example_id")

    print(f"Loaded: gold={len(gold)} statcheck={len(sc)} baseline={len(bl)}\n")

    sc_env_all = aggregate_by_env(gold, sc_idx)
    bl_env_all = aggregate_by_env(gold, bl_idx)
    sc_env_apa = aggregate_by_env(gold, sc_idx, applicable_only=True)
    bl_env_apa = aggregate_by_env(gold, bl_idx, applicable_only=True)

    cols_env = ["environment", "n_examples", "n_gold", "tp", "fp", "fn",
                "precision", "recall", "f1", "n_empty_pred_with_gold", "empty_rate"]
    print("\n Подмножество области применимости statcheck (apa + non_apa)\n")
    print(" statcheck:")
    print(fmt_table(sc_env_apa, cols_env))
    print("\n baseline:")
    print(fmt_table(bl_env_apa, cols_env))

    sc_idr = aggregate_idr(gold, sc_idx)
    bl_idr = aggregate_idr(gold, bl_idx)
    cols_idr = ["error_type", "n_pos", "tp", "idr", "n_neg", "fp", "far"]
    print("\nIDR / FAR по error_type\n")
    print(" statcheck:")
    print(fmt_table(sc_idr, cols_idr))
    print("\n baseline:")
    print(fmt_table(bl_idr, cols_idr))

    coverage_gap_rows = []
    for sc_row in sc_env_all:
        env = sc_row["environment"]
        bl_row = next((r for r in bl_env_all if r["environment"] == env), {})
        coverage_gap_rows.append({
            "environment": env,
            "n_gold": sc_row["n_gold"],
            "statcheck_recall": sc_row["recall"],
            "baseline_recall": bl_row.get("recall"),
            "delta_recall": (
                round(bl_row.get("recall", 0) - sc_row["recall"], 3)
                if bl_row.get("recall") is not None else None
            ),
            "statcheck_empty_rate": sc_row["empty_rate"],
        })
    print(fmt_table(coverage_gap_rows,
                    ["environment", "n_gold", "statcheck_recall",
                     "baseline_recall", "delta_recall", "statcheck_empty_rate"]))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "statcheck_by_env_all": sc_env_all,
            "baseline_by_env_all": bl_env_all,
            "statcheck_by_env_apa_only": sc_env_apa,
            "baseline_by_env_apa_only": bl_env_apa,
            "statcheck_idr": sc_idr,
            "baseline_idr": bl_idr,
            "coverage_gap": coverage_gap_rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n {args.output}")


if __name__ == "__main__":
    main()
