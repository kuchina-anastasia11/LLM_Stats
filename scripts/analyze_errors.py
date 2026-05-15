import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline.test_verificator import normalize_test_type


def fmt(v, spec=".3f"):
    if v is None:
        return "—"
    return format(v, spec)


def load_all_gold():
    # склеиваем dev и test gold по example_id
    gold = {}
    for name in ("dev_dataset.jsonl", "test_dataset.jsonl"):
        path = REPO / "data" / name
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                gold[r["example_id"]] = r
    return gold


def load_results(path):
    # возвращаем (model, split, ok_rows)
    with path.open() as f:
        rows = [json.loads(line) for line in f]
    ok = [r for r in rows if not r.get("error")]
    if not ok:
        return "?", "?", []
    model = ok[0].get("model", "unknown")
    if ok[0]["example_id"].startswith("dev-"):
        split = "dev"
    elif ok[0]["example_id"].startswith("test-"):
        split = "test"
    else:
        split = "mixed"
    return model, split, ok


# виды галлюцинаций
def analyze_fp_classifications(ok):
    cats = Counter()
    confusion = Counter()
    for r in ok:
        fps = (r.get("stage3") or {}).get("fp_classifications", [])
        for fp in fps:
            cats[fp["category"]] += 1
            if fp["category"] == "wrong_test_type":
                g = fp.get("closest_gold", {}) or {}
                p = fp.get("pred", {}) or {}
                confusion[(g.get("test_type"), p.get("test_type"))] += 1
    return {
        "categories": cats,
        "confusion_wrong_type": confusion,
        "total_fp": sum(cats.values()),
    }


#  F1 
def analyze_by_test_type(ok, gold):
    by_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n": 0})
    for r in ok:
        g = gold.get(r["example_id"])
        if not g or not g.get("tests"):
            continue
        types = [normalize_test_type(t.get("test_type")) for t in g["tests"]]
        if not types:
            continue
        major = Counter(types).most_common(1)[0][0]
        s3 = r.get("stage3") or {}
        b = by_type[major]
        b["tp"] += s3.get("tp", 0)
        b["fp"] += s3.get("fp", 0)
        b["fn"] += s3.get("fn", 0)
        b["n"] += 1

    rows = []
    for t, b in sorted(by_type.items(), key=lambda x: -x[1]["n"]):
        tp, fp, fn = b["tp"], b["fp"], b["fn"]
        if (tp + fp):
            p = tp / (tp + fp)
        else:
            p = None
        if (tp + fn):
            rc = tp / (tp + fn)
        else:
            rc = None
        if p and rc and (p + rc) > 0:
            f1 = 2 * p * rc / (p + rc)
        else:
            f1 = None
        row = {"test_type": t, "precision": p, "recall": rc, "f1": f1}
        row.update(b)
        rows.append(row)
    return rows


#  IDR/FAR 
def analyze_by_error_type(ok, gold):
    by_et = defaultdict(lambda: {"tpv": 0, "fpv": 0, "fnv": 0, "n": 0, "nc": 0})
    for r in ok:
        g = gold.get(r["example_id"])
        if not g:
            continue
        et = g.get("error_type") or "none"
        s5 = r.get("stage5") or {}
        b = by_et[et]
        b["tpv"] += s5.get("tpv", 0)
        b["fpv"] += s5.get("fpv", 0)
        b["fnv"] += s5.get("fnv", 0)
        b["nc"] += s5.get("not_checkable", 0)
        b["n"] += 1

    order = ["none", "rounding", "wrong_pvalue", "wrong_conclusion", "transcription"]
    rows = []
    for et in order:
        b = by_et.get(et)
        if not b or not b["n"]:
            continue
        if (b["tpv"] + b["fnv"]):
            idr = b["tpv"] / (b["tpv"] + b["fnv"])
        else:
            idr = None
        if (b["fpv"] + b["tpv"]):
            far = b["fpv"] / (b["fpv"] + b["tpv"])
        else:
            far = None
        row = {"error_type": et, "IDR": idr, "FAR": far}
        row.update(b)
        rows.append(row)
    return rows


# F1  environment
def analyze_by_environment(ok, gold):
    by_env = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n": 0})
    for r in ok:
        g = gold.get(r["example_id"])
        if not g:
            continue
        env = g.get("environment")
        if env is None:
            if r.get("source") == "real":
                env = "real"
            else:
                env = "mixed"
        s3 = r.get("stage3") or {}
        b = by_env[env]
        b["tp"] += s3.get("tp", 0)
        b["fp"] += s3.get("fp", 0)
        b["fn"] += s3.get("fn", 0)
        b["n"] += 1

    order = ["apa", "non_apa", "text", "two_apa", "two_text", "table", "no_test", "real"]
    rows = []
    for env in order:
        b = by_env.get(env)
        if not b or not b["n"]:
            continue
        tp, fp, fn = b["tp"], b["fp"], b["fn"]
        if (tp + fp):
            p = tp / (tp + fp)
        else:
            p = None
        if (tp + fn):
            rc = tp / (tp + fn)
        else:
            rc = None
        if p and rc and (p + rc) > 0:
            f1 = 2 * p * rc / (p + rc)
        else:
            f1 = None
        row = {"environment": env, "precision": p, "recall": rc, "f1": f1}
        row.update(b)
        rows.append(row)
    return rows


def print_report(model, split, n, fp_data, by_type, by_et, by_env):
    print("\n" + "=" * 70)
    print("=== " + model + " (" + split + ", n=" + str(n) + ") ===")
    print("=" * 70)

    # FP classifications
    total_fp = fp_data["total_fp"]
    print("\n  Виды галлюцинаций (всего " + str(total_fp) + " FP):")
    for cat, cnt in fp_data["categories"].most_common():
        pct = 100 * cnt / max(total_fp, 1)
        print("    " + cat.ljust(25) + str(cnt).rjust(4) + "  (" + format(pct, "4.1f") + "%)")
    if fp_data["confusion_wrong_type"]:
        print("\n  Confusion wrong_test_type (top-5):")
        for (g, p), cnt in fp_data["confusion_wrong_type"].most_common(5):
            print("    " + str(g) + " → " + str(p) + ": " + str(cnt))

    # по типу теста
    print("\n  F1 по типу теста (major в примере):")
    print("    " + "type".ljust(8) + "n".rjust(5) + "TP".rjust(5) + "FP".rjust(5) +
          "FN".rjust(5) + "Prec".rjust(7) + "Rec".rjust(7) + "F1".rjust(7))
    for row in by_type:
        print("    " + str(row["test_type"]).ljust(8) +
              str(row["n"]).rjust(5) +
              str(row["tp"]).rjust(5) +
              str(row["fp"]).rjust(5) +
              str(row["fn"]).rjust(5) +
              fmt(row["precision"], ".2f").rjust(7) +
              fmt(row["recall"], ".2f").rjust(7) +
              fmt(row["f1"], ".2f").rjust(7))

    # IDR/FAR  error_type
    print("\n  IDR / FAR по error_type:")
    print("    " + "type".ljust(20) + "n".rjust(5) + "TP".rjust(5) + "FP".rjust(5) +
          "FN".rjust(5) + "IDR".rjust(7) + "FAR".rjust(7))
    for row in by_et:
        print("    " + str(row["error_type"]).ljust(20) +
              str(row["n"]).rjust(5) +
              str(row["tpv"]).rjust(5) +
              str(row["fpv"]).rjust(5) +
              str(row["fnv"]).rjust(5) +
              fmt(row["IDR"], ".2f").rjust(7) +
              fmt(row["FAR"], ".2f").rjust(7))

    # F1 environment
    print("\n  F1 environment:")
    print("    " + "env".ljust(14) + "n".rjust(5) + "TP".rjust(5) + "FP".rjust(5) +
          "FN".rjust(5) + "F1".rjust(7))
    for row in by_env:
        print("    " + str(row["environment"]).ljust(14) +
              str(row["n"]).rjust(5) +
              str(row["tp"]).rjust(5) +
              str(row["fp"]).rjust(5) +
              str(row["fn"]).rjust(5) +
              fmt(row["f1"], ".2f").rjust(7))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="пути к results/*.jsonl")
    args = ap.parse_args()

    gold = load_all_gold()
    print("Gold loaded: " + str(len(gold)) + " records (dev + test)")

    for fpath in args.files:
        path = Path(fpath)
        model, split, ok = load_results(path)
        if not ok:
            print("\nWARN " + path.name + ": нет OK-записей")
            continue

        fp_data = analyze_fp_classifications(ok)
        by_type = analyze_by_test_type(ok, gold)
        by_et = analyze_by_error_type(ok, gold)
        by_env = analyze_by_environment(ok, gold)
        print_report(model, split, len(ok), fp_data, by_type, by_et, by_env)


if __name__ == "__main__":
    main()
