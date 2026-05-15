# прогон baseline пайплайна с двухпроходным stage 3:
#   pass A: inline-текст без markdown-таблиц → стандартный extract_tests
#   pass B: каждая таблица отдельно → table-focused LLM-вызов
# Эксп. 6 главы 4: ответ на коммент научника №3.
#
# usage:
#   export OPENROUTER_API_KEY=...
#   python3 scripts/run_table_sweep.py --dataset data/dev_dataset.jsonl \
#       --model openrouter-deepseek --output results/dev_table_sweep_or_deepseek.jsonl
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline import pdf_extractor, stats_extractor, interpritation_extractor, test_verificator
from pipeline import eval as eval_mod
from pipeline.experiments import table_sweep


def load_markdown(record, repo):
    if record["source"] == "synthetic":
        md = record.get("fragment", "")
        return md, {"n_pages": None, "n_tables": None, "markdown_len": len(md)}
    pdf_rel = record.get("pdf_path")
    if pdf_rel:
        pdf_abs = repo / pdf_rel
        if pdf_abs.exists():
            doc = pdf_extractor.extract(str(pdf_abs))
            return doc.text, {
                "n_pages": doc.n_pages,
                "n_tables": doc.n_tables,
                "markdown_len": len(doc.text),
            }
    md = record.get("fragment", "")
    return md, {"n_pages": None, "n_tables": None, "markdown_len": len(md), "fallback": True}


def process_record(record, markdown, model, api_key, verbose=False):
    def log(msg):
        if verbose:
            print("    " + msg, flush=True)

    # детектируем таблицы для diagnostics
    detected_tables = table_sweep.extract_tables(markdown)
    log("-> table_sweep: найдено " + str(len(detected_tables)) + " таблиц")

    # stage 3 — двухпроходное извлечение
    log("-> Этап 3 (table_sweep.extract_tests_two_pass)...")
    t0 = time.time()
    predicted_tests = table_sweep.extract_tests_two_pass(
        markdown, model_name=model, api_key=api_key
    )
    n_inline = sum(1 for t in predicted_tests if t.get("_source") == "inline")
    n_table = sum(1 for t in predicted_tests if t.get("_source") == "table_sweep")
    log("  Этап 3 готов: " + str(len(predicted_tests)) +
        " тестов (inline=" + str(n_inline) + ", table=" + str(n_table) +
        "; " + format(time.time() - t0, ".1f") + "s)")

    # сегментация
    sections = interpritation_extractor.split_sections(markdown)

    # stage 4 — стандартный, без изменений
    if predicted_tests:
        log("-> Этап 4 (interpritation_extractor)...")
        t0 = time.time()
        interpretations = interpritation_extractor.extract_interpretations(
            sections, predicted_tests, model_name=model, api_key=api_key
        )
        n_with_interp = sum(1 for r in interpretations if r.get("interpretations"))
        log("  Этап 4 готов: " + str(n_with_interp) + "/" + str(len(predicted_tests)) +
            " (" + format(time.time() - t0, ".1f") + "s)")
    else:
        log("-> Этап 4: пропускаю (нет тестов)")
        interpretations = []

    merged = interpritation_extractor.merge_with_tests(predicted_tests, interpretations)

    # stage 5
    verified = test_verificator.verify_all(merged)

    return {
        "predicted_tests": predicted_tests,
        "sections_found": sorted(sections.keys()),
        "merged": merged,
        "verified": verified,
        "n_tables_detected": len(detected_tables),
        "n_inline_predicted": n_inline,
        "n_table_predicted": n_table,
    }


def eval_record(record, artifacts, markdown):
    gold_tests = record["tests"]
    predicted_tests = artifacts["predicted_tests"]
    merged = artifacts["merged"]
    verified = artifacts["verified"]

    if record["source"] == "real":
        stage1 = eval_mod.raw_text_coverage(gold_tests, markdown)
    else:
        stage1 = None

    stage3 = eval_mod.eval_stage3(gold_tests, predicted_tests)
    pairs = stage3.get("pairs", [])
    stage4 = eval_mod.eval_stage4(gold_tests, merged, pairs)
    has_errors = any(t.get("consistent") is False for t in gold_tests)
    stage5 = eval_mod.eval_stage5_example(
        gold_tests, verified, pairs, example_has_errors=has_errors
    )
    return {"stage1": stage1, "stage3": stage3, "stage4": stage4, "stage5": stage5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True,
                    choices=["deepseek", "gemini", "openrouter-deepseek", "openrouter-gemini"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    env_key = stats_extractor.MODELS[args.model]["env_key"]
    api_key = os.environ.get(env_key)
    if not api_key:
        print("ERROR: " + env_key + " не задан", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.dataset).open() as f:
        records = [json.loads(l) for l in f]
    if args.limit:
        records = records[:args.limit]

    print("-> " + str(len(records)) + " примеров x table_sweep [" + args.model + "] -> " + str(output_path))

    t_start = time.time()
    ok = 0
    fail = 0
    with output_path.open("w", encoding="utf-8") as out_fh:
        for i, record in enumerate(records, 1):
            ex_id = record["example_id"]
            t0 = time.time()
            try:
                markdown, md_stats = load_markdown(record, REPO)
                artifacts = process_record(record, markdown, args.model, api_key, verbose=args.verbose)
                metrics = eval_record(record, artifacts, markdown)
                result = {
                    "example_id": ex_id,
                    "source": record["source"],
                    "source_id": record.get("source_id"),
                    "model": args.model,
                    "experiment": "table_sweep",
                    "markdown_stats": md_stats,
                    "n_gold_tests": len(record["tests"]),
                    "n_predicted_tests": len(artifacts["predicted_tests"]),
                    "n_tables_detected": artifacts["n_tables_detected"],
                    "n_inline_predicted": artifacts["n_inline_predicted"],
                    "n_table_predicted": artifacts["n_table_predicted"],
                    "sections_found": artifacts["sections_found"],
                    "predicted_tests": artifacts["predicted_tests"],
                    "merged_tests": artifacts["merged"],
                    "verified_tests": artifacts["verified"],
                    "error": None,
                }
                result.update(metrics)
                ok += 1
            except Exception as e:
                result = {
                    "example_id": ex_id,
                    "source": record["source"],
                    "model": args.model,
                    "experiment": "table_sweep",
                    "error": type(e).__name__ + ": " + str(e),
                    "traceback": traceback.format_exc().splitlines()[-3:],
                }
                fail += 1
            out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_fh.flush()
            dt = time.time() - t0
            eta_s = (time.time() - t_start) / i * (len(records) - i)
            status = "OK " if result.get("error") is None else "ERR"
            print("[" + format(i, "3d") + "/" + str(len(records)) + "] " + status + " " +
                  ex_id.ljust(20) + " " + format(dt, "5.1f") + "s  ETA " +
                  format(eta_s / 60, ".1f") + "m")

    total = time.time() - t_start
    print("\nDone: " + str(ok) + " ok, " + str(fail) + " failed, " +
          format(total / 60, ".1f") + " min -> " + str(output_path))


if __name__ == "__main__":
    main()
