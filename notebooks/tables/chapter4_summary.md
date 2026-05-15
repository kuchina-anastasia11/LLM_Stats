# Глава 4 — v2-сводка результатов

Сгенерировано `notebooks/chapter4_analysis_v2.py`.


## Baseline v2 (dev)

- **DeepSeek-V3**: n=282, P=0.942, R=0.886, F1=0.913, Hall=5.8%, IDR=0.662, Comb.IDR=0.665
- **Gemini 2.5 Flash**: n=282, P=0.912, R=0.921, F1=0.916, Hall=8.8%, IDR=0.662, Comb.IDR=0.675

## CoT v2 — главный фикс: clean-C класс

- **DS, baseline** clean-C (n=30): Comb.IDR=0.033
- **DS, + CoT v2** clean-C (n=30): Comb.IDR=0.069
- **Gem, baseline** clean-C (n=30): Comb.IDR=0.033
- **Gem, + CoT v2** clean-C (n=30): Comb.IDR=0.034

## Cross-verify v2 — adversarial framing

- **DS $\to$ Gem v2**: F1=0.913, removed=14, added=31, corrected=38
- **Gem $\to$ DS v2**: F1=0.914, removed=2, added=6, corrected=3

## Эксп. 6 — Table-sweep (новый)

- **DS, baseline** real (n=2): Recall=0.358, F1=0.528
- **DS, + table-sweep** real (n=2): Recall=0.509, F1=0.562
- **Gem, baseline** real (n=2): Recall=0.604, F1=0.610
- **Gem, + table-sweep** real (n=2): Recall=0.755, F1=0.696

## Все таблицы

- `notebooks/tables/table_4_10_env_breakdown.tex`
- `notebooks/tables/table_4_11_idr_by_error.tex`
- `notebooks/tables/table_4_1_baseline_env.tex`
- `notebooks/tables/table_4_1_baseline_stage3.tex`
- `notebooks/tables/table_4_1_baseline_stage5.tex`
- `notebooks/tables/table_4_2_cleanup.tex`
- `notebooks/tables/table_4_3_cot.tex`
- `notebooks/tables/table_4_3_cot_by_error.tex`
- `notebooks/tables/table_4_4_xverify.tex`
- `notebooks/tables/table_4_4_xverify_diag.tex`
- `notebooks/tables/table_4_5_summarisation.tex`
- `notebooks/tables/table_4_6_fewshot.tex`
- `notebooks/tables/table_4_7_models.tex`
- `notebooks/tables/table_4_7_test_headline.tex`
- `notebooks/tables/table_4_7_test_paired.tex`
- `notebooks/tables/table_4_8_table_sweep.tex`
- `notebooks/tables/table_4_8_table_sweep_diag.tex`
- `notebooks/tables/table_4_9_fp_categories.tex`
