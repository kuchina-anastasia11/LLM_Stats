import argparse
import json
import sys
from pathlib import Path
from pipeline import eval as eval_mod

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def load(path):
    out = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") is not None:
                continue
            out.append(r)
    return out


def attach_gold(records, repo):
    # подсасываем gold tests если их нет в records
    if not records:
        return
    if "tests" in records[0] or "gold_tests" in records[0]:
        return
    gold_by_id = {}
    for ds_name in ("dev_dataset.jsonl", "test_dataset.jsonl"):
        ds_path = repo / "data" / ds_name
        if not ds_path.exists():
            continue
        with ds_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                gold_by_id[item["example_id"]] = item["tests"]
    for r in records:
        r["tests"] = gold_by_id.get(r["example_id"], [])


def aggregate(records):
    s3 = []
    s4 = []
    s5 = []
    for rec in records:
        gold = rec.get("tests", [])
        pred = rec.get("predicted_tests", [])
        merged = rec.get("merged_tests", [])
        verified = rec.get("verified_tests", [])

        m3 = eval_mod.eval_stage3(gold, pred)
        s3.append(m3)
        pairs = m3.get("pairs", [])
        s4.append(eval_mod.eval_stage4(gold, merged, pairs))
        has_err = any(t.get("consistent") is False for t in gold)
        s5.append(eval_mod.eval_stage5_example(gold, verified, pairs, example_has_errors=has_err))

    return {
        "stage3": eval_mod.aggregate_stage3(s3),
        "stage4": eval_mod.aggregate_stage4(s4),
        "stage5": eval_mod.aggregate_stage5(s5),
    }


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return format(v, ".3f")
    return str(v)


def diff(a, b):
    if a is None or b is None:
        return ""
    d = b - a
    if d >= 0:
        sign = "+"
    else:
        sign = ""
    return "(" + sign + format(d, ".3f") + ")"


METRIC_LAYOUT = [
    ("Stage 3 — обнаружение тестов", "stage3", [
        ("tp", "TP"), ("fp", "FP"), ("fn", "FN"),
        ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
        ("field_accuracy", "Field Acc"),
        ("complete_extraction_rate", "Complete Extr"),
        ("hallucination_rate", "Hallucination"),
    ]),
    ("Stage 4 — интерпретации", "stage4", [
        ("primary_direction_accuracy", "Direction Acc"),
        ("found_interpretation_rate", "Found Interp"),
        ("matched_pairs", "Matched"),
    ]),
    ("Stage 5 — самосогласованность", "stage5", [
        ("tpv", "TPv"), ("fpv", "FPv"), ("fnv", "FNv"),
        ("inconsistency_detection_rate", "IDR (p-only)"),
        ("false_alarm_rate", "FAR (p-only)"),
        ("combined_idr", "IDR combined"),
        ("combined_far", "FAR combined"),
    ]),
]


def render(label_a, label_b, agg_a, agg_b):
    print()
    name_w = 22
    val_w = 12
    head = "metric".ljust(name_w) + " " + label_a.rjust(val_w) + " " + label_b.rjust(val_w) + " " + "Δ".rjust(val_w)
    print(head)
    print("-" * len(head))
    for stage_title, stage_key, fields in METRIC_LAYOUT:
        print("\n  " + stage_title)
        for key, label in fields:
            va = agg_a[stage_key].get(key)
            vb = agg_b[stage_key].get(key)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                d = diff(va, vb)
            else:
                d = ""
            print("  " + label.ljust(name_w - 2) + " " + fmt(va).rjust(val_w) + " " +
                  fmt(vb).rjust(val_w) + " " + d.rjust(val_w))


def filter_source(records, src):
    return [r for r in records if r.get("source") == src]


def breakdown_by_error_type(records):
    # primary_direction_accuracy 
    from pipeline import eval as eval_mod

    buckets = {}
    allowed = {"significant", "not_significant", "marginal", "unclear"}

    for rec in records:
        gold = rec.get("tests", [])
        pred = rec.get("predicted_tests", [])
        merged = rec.get("merged_tests", [])
        if not gold or not pred or not merged:
            continue
        m3 = eval_mod.eval_stage3(gold, pred)
        for pi, gi in m3.get("pairs", []):
            g = gold[gi]
            p = merged[pi]
            err = g.get("error_type") or "none"
            gold_dir = g.get("interpretation_direction")
            pred_dir = p.get("primary_direction")
            if gold_dir not in allowed or gold_dir == "unclear":
                continue
            buckets.setdefault(err, {"hits": 0, "total": 0})
            buckets[err]["total"] += 1
            if pred_dir == gold_dir:
                buckets[err]["hits"] += 1

    out = {}
    for err, c in buckets.items():
        if c["total"]:
            acc = c["hits"] / c["total"]
        else:
            acc = None
        out[err] = {"hits": c["hits"], "total": c["total"], "accuracy": acc}
    return out


def render_by_error_type(label_a, label_b, br_a, br_b):
    print()
    print("  Stage 4 — primary_direction_accuracy по error_type")
    print("-" * 65)
    print("  " + "error_type".ljust(20) + " " + label_a.rjust(14) + " " +
          label_b.rjust(14) + " " + "Δ".rjust(10))
    error_types = sorted(set(br_a) | set(br_b))
    for err in error_types:
        a = br_a.get(err, {})
        b = br_b.get(err, {})
        a_acc = a.get("accuracy")
        b_acc = b.get("accuracy")
        a_str = fmt(a_acc) + " (" + str(a.get("hits", 0)) + "/" + str(a.get("total", 0)) + ")"
        b_str = fmt(b_acc) + " (" + str(b.get("hits", 0)) + "/" + str(b.get("total", 0)) + ")"
        if a_acc is not None and b_acc is not None:
            d = b_acc - a_acc
            if d >= 0:
                sign = "+"
            else:
                sign = ""
            d_str = "(" + sign + format(d, ".3f") + ")"
        else:
            d_str = ""
        print("  " + err.ljust(20) + " " + a_str.rjust(14) + " " + b_str.rjust(14) + " " + d_str.rjust(10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path_a", help="первый jsonl (baseline)")
    ap.add_argument("path_b", help="второй jsonl (эксперимент)")
    ap.add_argument("--labels", nargs=2, default=["A", "B"], metavar=("LABEL_A", "LABEL_B"))
    ap.add_argument("--by-source", action="store_true", help="разрез по synthetic/real")
    ap.add_argument("--by-error-type", action="store_true",
                    help="разрез primary_direction_accuracy по error_type")
    args = ap.parse_args()

    a = load(Path(args.path_a))
    b = load(Path(args.path_b))
    attach_gold(a, REPO)
    attach_gold(b, REPO)

    print("\n" + args.labels[0] + ": " + args.path_a + " (" + str(len(a)) + " records)")
    print(args.labels[1] + ": " + args.path_b + " (" + str(len(b)) + " records)")
    render(args.labels[0], args.labels[1], aggregate(a), aggregate(b))

    if args.by_source:
        for src in ("synthetic", "real"):
            a_src = filter_source(a, src)
            b_src = filter_source(b, src)
            if not a_src and not b_src:
                continue
            print("\n=== source = " + src + " (" + str(len(a_src)) + " / " + str(len(b_src)) + " records) ===")
            render(args.labels[0], args.labels[1], aggregate(a_src), aggregate(b_src))

    if args.by_error_type:
        render_by_error_type(args.labels[0], args.labels[1],
                             breakdown_by_error_type(a), breakdown_by_error_type(b))


if __name__ == "__main__":
    main()
