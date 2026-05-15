import random
from pathlib import Path

import numpy as np

from _chapter4_helpers import (
    load_run, join_gold, agg_stages, tex_table, write_tex,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"
TABLES_OUT = Path(__file__).parent / "tables"

B = 1000
SEED = 42

# Прогоны на test и dev, для которых считаем CI.
RUNS = {
    "dev_baseline_ds":        (RESULTS / "dev_baseline_v2_or_deepseek.jsonl",
                               DATA / "dev_dataset.jsonl"),
    "dev_baseline_gem":       (RESULTS / "dev_baseline_v2_or_gemini.jsonl",
                               DATA / "dev_dataset.jsonl"),
    "test_baseline_ds":       (RESULTS / "test_baseline_v2_or_deepseek.jsonl",
                               DATA / "test_dataset.jsonl"),
    "test_baseline_gem":      (RESULTS / "test_baseline_v2_or_gemini.jsonl",
                               DATA / "test_dataset.jsonl"),
    "test_ts_cot_gem":        (RESULTS / "test_table_sweep_cot_or_gemini.jsonl",
                               DATA / "test_dataset.jsonl"),
    "test_ultimate_ds":       (RESULTS / "test_ultimate_or_deepseek.jsonl",
                               DATA / "test_dataset.jsonl"),
    "test_ultimate_gem":      (RESULTS / "test_ultimate_or_gemini.jsonl",
                               DATA / "test_dataset.jsonl"),
}

RUNS_DATA = {}
for label, (path, dev_path) in RUNS.items():
    if not path.exists():
        print(f"  miss: {path.name}")
        continue
    rows = load_run(path)
    rows = join_gold(rows, dev_path)
    RUNS_DATA[label] = rows
    print(f"  {label}: n={len(rows)}")

# %%
# базовые экстракторы метрик из агрегата
METRICS = {
    "f1":         lambda a: a["stage3"].get("f1"),
    "precision":  lambda a: a["stage3"].get("precision"),
    "recall":     lambda a: a["stage3"].get("recall"),
    "field_acc":  lambda a: a["stage3"].get("field_accuracy"),
    "primary":    lambda a: a["stage4"].get("primary_direction_accuracy"),
    "idr_p":      lambda a: a["stage5"].get("inconsistency_detection_rate"),
    "far_p":      lambda a: a["stage5"].get("false_alarm_rate"),
    "comb_idr":   lambda a: a["stage5"].get("combined_idr"),
    "comb_far":   lambda a: a["stage5"].get("combined_far"),
    "hall":       lambda a: a["stage3"].get("hallucination_rate"),
}


def bootstrap_ci(rows, B=B, seed=SEED, alpha=0.05):
    """95% bootstrap CI для всех метрик из METRICS на одном прогоне.

    Сэмплируем example-ы с заменой, агрегируем как обычно.
    Возвращаем dict: metric_name -> (point, lo, hi).
    """
    if not rows:
        return {}
    rng = random.Random(seed)
    n = len(rows)
    samples = {name: [] for name in METRICS}
    for _ in range(B):
        boot = [rows[rng.randrange(n)] for _ in range(n)]
        a = agg_stages(boot)
        for name, extractor in METRICS.items():
            v = extractor(a)
            if v is None:
                continue
            samples[name].append(v)
    point_agg = agg_stages(rows)
    out = {}
    lo_q = 100 * alpha / 2
    hi_q = 100 * (1 - alpha / 2)
    for name, vals in samples.items():
        if not vals:
            out[name] = (None, None, None)
            continue
        out[name] = (
            METRICS[name](point_agg),
            float(np.percentile(vals, lo_q)),
            float(np.percentile(vals, hi_q)),
        )
    return out


# %%
print("\n[1] точечные CI на test")
CI_BY_RUN = {}
for label, rows in RUNS_DATA.items():
    CI_BY_RUN[label] = bootstrap_ci(rows)
    print(f"  {label}: f1={CI_BY_RUN[label]['f1']}")


# %%
def fmt_ci(point, lo, hi, pct=False, places=3):
    """0,936 [0,918; 0,949] — русские запятые, скобки квадратные."""
    if point is None:
        return "—"
    if pct:
        return f"{point*100:.1f} [{lo*100:.1f}; {hi*100:.1f}]".replace(".", ",")
    spec = f".{places}f"
    return f"{format(point, spec)} [{format(lo, spec)}; {format(hi, spec)}]".replace(".", ",")


def row_for_run(label_ru, label_key):
    ci = CI_BY_RUN.get(label_key, {})
    rows_loc = RUNS_DATA.get(label_key, [])
    if not ci:
        return [label_ru, "—", "—", "—", "—", "—"]
    f1_p, f1_lo, f1_hi = ci.get("f1", (None, None, None))
    pa_p, pa_lo, pa_hi = ci.get("primary", (None, None, None))
    ci_p, ci_lo, ci_hi = ci.get("comb_idr", (None, None, None))
    h_p, h_lo, h_hi    = ci.get("hall", (None, None, None))
    return [
        label_ru,
        str(len(rows_loc)),
        fmt_ci(f1_p, f1_lo, f1_hi),
        fmt_ci(pa_p, pa_lo, pa_hi),
        fmt_ci(ci_p, ci_lo, ci_hi),
        fmt_ci(h_p, h_lo, h_hi, pct=True, places=1),
    ]


test_rows = [
    row_for_run("DS, baseline v2",            "test_baseline_ds"),
    row_for_run("Gem, baseline v2",           "test_baseline_gem"),
    row_for_run("Gem + table-sweep + CoT",    "test_ts_cot_gem"),
    row_for_run("DS ULTIMATE (summ+ts+CoT)",  "test_ultimate_ds"),
    row_for_run("Gem ULTIMATE (summ+ts+CoT)", "test_ultimate_gem"),
]

caption = (
    "Headline-метрики на~test-сплите с~95\\% bootstrap CI (B=1000 итераций "
    "по~примерам). ULTIMATE = stage~2 (summarisation) + stage~3 (table-sweep) "
    "+ stage~4 (CoT v2)."
)
note = (
    "Точечная оценка и~95\\% CI получены непараметрическим бутстрэпом "
    "по~примерам test-сплита (ресэмплинг с~заменой). Inference LLM "
    "детерминирован при~$T=0$, поэтому seed-вариативности нет; источник "
    "разброса --- ограниченный размер выборки."
)
write_tex(TABLES_OUT / "table_4_7_test_headline.tex", tex_table(
    label="tab:exp_test_headline",
    caption=caption,
    headers=[
        "Прогон", "n",
        "$F_1$ [95\\% CI]",
        "Primary Acc [95\\% CI]",
        "Comb.~IDR [95\\% CI]",
        "Hall., \\% [95\\% CI]",
    ],
    rows=test_rows,
    column_spec="l r r r r r",
    note=note,
))
print("\n  -> table_4_7_test_headline.tex updated")


# %%
# paired bootstrap: разность метрик между двумя прогонами на ОДНОМ test-сплите
# (сэмплируем индексы один раз, берём одни и те же example_id из обоих наборов).
def paired_diff_ci(rows_a, rows_b, metric_name, B=B, seed=SEED, alpha=0.05):
    if not rows_a or not rows_b:
        return None
    # индексируем по example_id
    by_id_a = {r["example_id"]: r for r in rows_a}
    by_id_b = {r["example_id"]: r for r in rows_b}
    common = sorted(set(by_id_a) & set(by_id_b))
    if not common:
        return None
    rng = random.Random(seed)
    n = len(common)
    diffs = []
    for _ in range(B):
        boot_ids = [common[rng.randrange(n)] for _ in range(n)]
        boot_a = [by_id_a[i] for i in boot_ids]
        boot_b = [by_id_b[i] for i in boot_ids]
        va = METRICS[metric_name](agg_stages(boot_a))
        vb = METRICS[metric_name](agg_stages(boot_b))
        if va is None or vb is None:
            continue
        diffs.append(va - vb)
    if not diffs:
        return None
    full_a = METRICS[metric_name](agg_stages([by_id_a[i] for i in common]))
    full_b = METRICS[metric_name](agg_stages([by_id_b[i] for i in common]))
    return (
        full_a - full_b,
        float(np.percentile(diffs, 100 * alpha / 2)),
        float(np.percentile(diffs, 100 * (1 - alpha / 2))),
    )


# таблица paired-сравнений: ключевые контрасты на test
print("\n[2] paired CI разностей на test")
paired_pairs = [
    ("Gem ULTIMATE --- Gem baseline", "test_ultimate_gem", "test_baseline_gem"),
    ("DS  ULTIMATE --- DS  baseline", "test_ultimate_ds",  "test_baseline_ds"),
    ("Gem ULTIMATE --- DS ULTIMATE",  "test_ultimate_gem", "test_ultimate_ds"),
    ("Gem ULTIMATE --- Gem ts+CoT",   "test_ultimate_gem", "test_ts_cot_gem"),
]

paired_rows = []
for label, a, b in paired_pairs:
    rows_a = RUNS_DATA.get(a, [])
    rows_b = RUNS_DATA.get(b, [])
    f1 = paired_diff_ci(rows_a, rows_b, "f1")
    ci = paired_diff_ci(rows_a, rows_b, "comb_idr")
    hall = paired_diff_ci(rows_a, rows_b, "hall")
    paired_rows.append([
        label,
        fmt_ci(*f1) if f1 else "—",
        fmt_ci(*ci) if ci else "—",
        fmt_ci(*hall, pct=True, places=1) if hall else "—",
    ])
    print(f"  {label}: ΔF1={f1}, ΔIDR={ci}")

write_tex(TABLES_OUT / "table_4_7_test_paired.tex", tex_table(
    label="tab:exp_test_paired",
    caption=(
        "Paired bootstrap CI для~разностей метрик на~test-сплите "
        "(B=1000, сэмплинг по~общим example\\_id). "
        "CI, не~пересекающий ноль, соответствует статистически значимой разнице "
        "на~уровне $\\alpha=0{,}05$."
    ),
    headers=[
        "Сравнение",
        "$\\Delta F_1$ [95\\% CI]",
        "$\\Delta$ Comb.~IDR [95\\% CI]",
        "$\\Delta$ Hall., \\% [95\\% CI]",
    ],
    rows=paired_rows,
    column_spec="l r r r",
))
print("\n  -> table_4_7_test_paired.tex updated")

# %%
# обновлённая table_4_1 (baseline stage3) с CI на F1, Field Acc, Hall
print("\n[3] baseline stage3 с CI")


def baseline_row_s3(label_ru, label_key):
    ci = CI_BY_RUN.get(label_key, {})
    rows_loc = RUNS_DATA.get(label_key, [])
    a = agg_stages(rows_loc) if rows_loc else {}
    s3 = a.get("stage3", {})
    n = a.get("n_examples", 0)
    tp, fp, fn = s3.get("tp", "—"), s3.get("fp", "—"), s3.get("fn", "—")
    p   = ci.get("precision",  (None, None, None))
    r   = ci.get("recall",     (None, None, None))
    f1  = ci.get("f1",         (None, None, None))
    fa  = ci.get("field_acc",  (None, None, None))
    h   = ci.get("hall",       (None, None, None))
    return [
        label_ru, str(n), str(tp), str(fp), str(fn),
        fmt_ci(*p), fmt_ci(*r), fmt_ci(*f1), fmt_ci(*fa),
        fmt_ci(*h, pct=True, places=1),
    ]


baseline_rows_s3 = [
    baseline_row_s3("DeepSeek-V3 (dev)",       "dev_baseline_ds"),
    baseline_row_s3("Gemini 2.5 Flash (dev)",  "dev_baseline_gem"),
    baseline_row_s3("DeepSeek-V3 (test)",      "test_baseline_ds"),
    baseline_row_s3("Gemini 2.5 Flash (test)", "test_baseline_gem"),
]
write_tex(TABLES_OUT / "table_4_1_baseline_stage3.tex", tex_table(
    label="tab:exp_baseline_stage3",
    caption=(
        "Baseline (v2-промптом): метрики этапа~3 (обнаружение и~извлечение) "
        "с~95\\% bootstrap CI (B=1000, ресэмплинг по~примерам сплита)."
    ),
    headers=[
        "Прогон", "n", "TP", "FP", "FN",
        "Precision [95\\% CI]",
        "Recall [95\\% CI]",
        "$F_1$ [95\\% CI]",
        "Field Acc [95\\% CI]",
        "Hall., \\% [95\\% CI]",
    ],
    rows=baseline_rows_s3,
    column_spec="l r r r r r r r r r",
))
print("  -> table_4_1_baseline_stage3.tex updated")


# stage 5 для baseline c CI
def baseline_row_s5(label_ru, label_key):
    ci = CI_BY_RUN.get(label_key, {})
    rows_loc = RUNS_DATA.get(label_key, [])
    a = agg_stages(rows_loc) if rows_loc else {}
    s5 = a.get("stage5", {})
    n = a.get("n_examples", 0)
    tpv = s5.get("tpv", "—")
    fpv = s5.get("fpv", "—")
    fnv = s5.get("fnv", "—")
    idr  = ci.get("idr_p",    (None, None, None))
    far  = ci.get("far_p",    (None, None, None))
    cidr = ci.get("comb_idr", (None, None, None))
    cfar = ci.get("comb_far", (None, None, None))
    return [
        label_ru, str(n), str(tpv), str(fpv), str(fnv),
        fmt_ci(*idr), fmt_ci(*far), fmt_ci(*cidr), fmt_ci(*cfar),
    ]


baseline_rows_s5 = [
    baseline_row_s5("DeepSeek-V3 (dev)",       "dev_baseline_ds"),
    baseline_row_s5("Gemini 2.5 Flash (dev)",  "dev_baseline_gem"),
    baseline_row_s5("DeepSeek-V3 (test)",      "test_baseline_ds"),
    baseline_row_s5("Gemini 2.5 Flash (test)", "test_baseline_gem"),
]
write_tex(TABLES_OUT / "table_4_1_baseline_stage5.tex", tex_table(
    label="tab:exp_baseline_stage5",
    caption=(
        "Baseline (v2-промптом): метрики этапа~5 (проверка самосогласованности) "
        "с~95\\% bootstrap CI (B=1000, ресэмплинг по~примерам сплита). "
        "IDR/FAR\\,(p) --- только числовая проверка, Combined --- "
        "числовая$\\lor$семантическая."
    ),
    headers=[
        "Прогон", "n", "TP$_v$", "FP$_v$", "FN$_v$",
        "IDR\\,(p) [95\\% CI]",
        "FAR\\,(p) [95\\% CI]",
        "Comb.~IDR [95\\% CI]",
        "Comb.~FAR [95\\% CI]",
    ],
    rows=baseline_rows_s5,
    column_spec="l r r r r r r r r",
))
print("  -> table_4_1_baseline_stage5.tex updated")


print("\nDone. tables written to:", TABLES_OUT)
