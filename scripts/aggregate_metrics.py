import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline import eval as eval_mod


def load_results(paths):
    # читаем все jsonl
    all_rows = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print("WARN: " + str(p) + " не существует - пропускаю", file=sys.stderr)
            continue
        with path.open() as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    print("WARN: " + str(p) + ":" + str(i) + " битый JSON, пропускаю", file=sys.stderr)
                    continue
                row["_source_file"] = str(path.name)
                all_rows.append(row)
    return all_rows


def fmt(v, spec=".3f", na="—"):
    if v is None:
        return na
    return format(v, spec)


def aggregate_group(rows):
    # агрегируем per-example метрики для одной группы
    stage3_per = []
    stage4_per = []
    stage5_per = []
    for r in rows:
        if r.get("stage3"):
            stage3_per.append(r["stage3"])
        if r.get("stage4"):
            stage4_per.append(r["stage4"])
        if r.get("stage5"):
            stage5_per.append(r["stage5"])

    # stage 1 - только real
    s1 = [r["stage1"] for r in rows if r.get("stage1")]
    if s1:
        total_matched = sum(x["matched"] for x in s1)
        total = sum(x["total"] for x in s1)
        if total:
            coverage = total_matched / total
        else:
            coverage = None
    else:
        coverage = None
        total_matched = 0
        total = 0

    if stage3_per:
        agg3 = eval_mod.aggregate_stage3(stage3_per)
    else:
        agg3 = {}
    if stage4_per:
        agg4 = eval_mod.aggregate_stage4(stage4_per)
    else:
        agg4 = {}
    if stage5_per:
        agg5 = eval_mod.aggregate_stage5(stage5_per)
    else:
        agg5 = {}

    out = {
        "n_examples": len(rows),
        "n_errors": sum(1 for r in rows if r.get("error")),
        "stage1_coverage": coverage,
        "stage1_matched": total_matched,
        "stage1_total": total,
    }
    for k, v in agg3.items():
        out["s3_" + k] = v
    for k, v in agg4.items():
        out["s4_" + k] = v
    for k, v in agg5.items():
        out["s5_" + k] = v
    return out


def print_markdown(groups, out_fh):
    # печать markdown-таблиц
    def w(s=""):
        print(s, file=out_fh)

    w("# Baseline evaluation — агрегированные метрики\n")

    # stage 1
    w("## Этап 1 — PDF → markdown (coverage gold raw_text)\n")
    w("| Модель × source | examples | matched / total | coverage |")
    w("| --- | --- | --- | --- |")
    for key, agg in groups.items():
        label = key[0] + " × " + key[1]
        if agg["stage1_total"]:
            matched_total = str(agg["stage1_matched"]) + " / " + str(agg["stage1_total"])
        else:
            matched_total = "—"
        w("| " + label + " | " + str(agg["n_examples"]) + " | " + matched_total +
          " | " + fmt(agg["stage1_coverage"]) + " |")
    w()

    # stage 3
    w("## Этап 3 — извлечение тестов\n")
    w("| Модель × source | TP | FP | FN | Precision | Recall | F1 | Field Acc | Complete Extr | Hallucination |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for key, agg in groups.items():
        label = key[0] + " × " + key[1]
        w("| " + label +
          " | " + str(agg.get("s3_tp", "—")) +
          " | " + str(agg.get("s3_fp", "—")) +
          " | " + str(agg.get("s3_fn", "—")) +
          " | " + fmt(agg.get("s3_precision")) +
          " | " + fmt(agg.get("s3_recall")) +
          " | " + fmt(agg.get("s3_f1")) +
          " | " + fmt(agg.get("s3_field_accuracy")) +
          " | " + fmt(agg.get("s3_complete_extraction_rate")) +
          " | " + fmt(agg.get("s3_hallucination_rate")) + " |")
    w()

    # stage 4
    w("## Этап 4 — интерпретации\n")
    w("| Модель × source | matched pairs | primary_direction hits / total | accuracy | found_interp_rate |")
    w("| --- | --- | --- | --- | --- |")
    for key, agg in groups.items():
        label = key[0] + " × " + key[1]
        hits_total = str(agg.get("s4_direction_hits", 0)) + " / " + str(agg.get("s4_direction_total", 0))
        w("| " + label + " | " + str(agg.get("s4_matched_pairs", "—")) +
          " | " + hits_total +
          " | " + fmt(agg.get("s4_primary_direction_accuracy")) +
          " | " + fmt(agg.get("s4_found_interpretation_rate")) + " |")
    w()

    # stage 5
    w("## Этап 5 — проверка самосогласованности\n")
    w("| Модель × source | TPv | FPv | FNv | nc | IDR (p) | FAR (p) | IDR comb. | FAR comb. |")
    w("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for key, agg in groups.items():
        label = key[0] + " × " + key[1]
        w("| " + label +
          " | " + str(agg.get("s5_tpv", "—")) +
          " | " + str(agg.get("s5_fpv", "—")) +
          " | " + str(agg.get("s5_fnv", "—")) +
          " | " + str(agg.get("s5_not_checkable", "—")) +
          " | " + fmt(agg.get("s5_inconsistency_detection_rate")) +
          " | " + fmt(agg.get("s5_false_alarm_rate")) +
          " | " + fmt(agg.get("s5_combined_idr")) +
          " | " + fmt(agg.get("s5_combined_far")) + " |")
    w()

    # ошибки
    total_errors = sum(agg["n_errors"] for agg in groups.values())
    if total_errors:
        w("## Ошибки пайплайна\n")
        w("| Модель × source | ошибок / всего |")
        w("| --- | --- |")
        for key, agg in groups.items():
            label = key[0] + " × " + key[1]
            w("| " + label + " | " + str(agg["n_errors"]) + " / " + str(agg["n_examples"]) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="пути к results*.jsonl")
    ap.add_argument("--output", help="путь для сохранения markdown")
    ap.add_argument("--split-source", action="store_true", default=True,
                    help="разделять по source (synthetic/real)")
    ap.add_argument("--combined", action="store_true",
                    help="дополнительно вывести combined-строку (synthetic + real)")
    args = ap.parse_args()

    all_rows = load_results(args.paths)

    # группируем
    groups = defaultdict(list)
    errors_per_group = defaultdict(int)
    for r in all_rows:
        model = r.get("model", "unknown")
        source = r.get("source", "unknown")
        key = (model, source)
        if r.get("error"):
            errors_per_group[key] += 1
        else:
            groups[key].append(r)

    aggregated = {}
    for key, rows in sorted(groups.items()):
        agg = aggregate_group(rows)
        agg["n_errors"] = errors_per_group[key]
        agg["n_examples"] = len(rows) + errors_per_group[key]
        aggregated[key] = agg

    if args.combined:
        # combined по моделям
        by_model = defaultdict(list)
        by_model_err = defaultdict(int)
        for (model, source), rows in groups.items():
            by_model[model].extend(rows)
        for (model, source), n in errors_per_group.items():
            by_model_err[model] += n
        for model, rows in sorted(by_model.items()):
            agg = aggregate_group(rows)
            agg["n_errors"] = by_model_err[model]
            agg["n_examples"] = len(rows) + by_model_err[model]
            aggregated[(model, "ALL")] = agg

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            print_markdown(aggregated, fh)
        print("Saved to " + str(out_path))
    else:
        print_markdown(aggregated, sys.stdout)


if __name__ == "__main__":
    main()
