# прогон baseline пайплайна (одна модель + один датасет)
#
# для каждой записи:
#   1. markdown (fragment для synthetic, pdf_extractor для real)
#   2. stage 3: stats_extractor.extract_tests
#   3. stage 4: split_sections + extract_interpretations + merge_with_tests
#   4. stage 5: test_verificator.verify_all
#   5. метрики через pipeline.eval
#
# usage:
#   export DEEPSEEK_API_KEY=...
#   python3 scripts/run_baseline.py --dataset data/test_dataset.jsonl --model deepseek --output results/test_deepseek.jsonl
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


def load_markdown(record, repo):
    # достаём markdown для записи
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
    # fallback: склеенные raw_text
    md = record.get("fragment", "")
    return md, {"n_pages": None, "n_tables": None, "markdown_len": len(md), "fallback": True}


def process_record(record, markdown, model, api_key, verbose=False):
    # прогоняем все этапы
    def log(msg):
        if verbose:
            print("    " + msg, flush=True)

    # stage 3
    log("-> Этап 3 (stats_extractor): извлекаем тесты...")
    t0 = time.time()
    predicted_tests = stats_extractor.extract_tests(markdown, model_name=model, api_key=api_key)
    log("  Этап 3 готов: " + str(len(predicted_tests)) + " тестов (" + format(time.time() - t0, ".1f") + "s)")

    # сегментация
    log("-> Сегментация (split_sections)...")
    sections = interpritation_extractor.split_sections(markdown)
    section_names = sorted(sections.keys())
    log("  секций: " + str(len(section_names)) + " " + str(section_names))

    # stage 4
    if predicted_tests:
        log("-> Этап 4 (interpritation_extractor): " + str(len(predicted_tests)) + " тестов...")
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
    log("-> Этап 5 (test_verificator)...")
    verified = test_verificator.verify_all(merged)
    inc = sum(1 for v in verified if v.get("p_consistency") == "inconsistent")
    con = sum(1 for v in verified if v.get("p_consistency") == "consistent")
    nc = sum(1 for v in verified if v.get("p_consistency") == "not_checkable")
    log("  Этап 5: " + str(con) + " consistent / " + str(inc) + " inconsistent / " + str(nc) + " not_checkable")

    return {
        "predicted_tests": predicted_tests,
        "sections_found": section_names,
        "merged": merged,
        "verified": verified,
    }


def eval_record(record, artifacts, markdown):
    # считаем per-example метрики
    gold_tests = record["tests"]
    predicted_tests = artifacts["predicted_tests"]
    merged = artifacts["merged"]
    verified = artifacts["verified"]

    # stage 1 - только real
    if record["source"] == "real":
        stage1 = eval_mod.raw_text_coverage(gold_tests, markdown)
    else:
        stage1 = None

    stage3 = eval_mod.eval_stage3(gold_tests, predicted_tests)
    pairs = stage3.get("pairs", [])

    stage4 = eval_mod.eval_stage4(gold_tests, merged, pairs)

    has_errors = any(t.get("consistent") is False for t in gold_tests)
    stage5 = eval_mod.eval_stage5_example(gold_tests, verified, pairs, example_has_errors=has_errors)

    return {"stage1": stage1, "stage3": stage3, "stage4": stage4, "stage5": stage5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model",
                    choices=["deepseek", "gemini", "openrouter-deepseek", "openrouter-gemini"],
                    required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--skip-synth", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    env_key = stats_extractor.MODELS[args.model]["env_key"]
    api_key = os.environ.get(env_key)
    if not api_key:
        print("ERROR: " + env_key + " не задан", file=sys.stderr)
        sys.exit(1)

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with dataset_path.open() as f:
        records = [json.loads(l) for l in f]

    if args.skip_real:
        records = [r for r in records if r["source"] == "synthetic"]
    if args.skip_synth:
        records = [r for r in records if r["source"] == "real"]
    if args.limit:
        records = records[:args.limit]

    print("-> " + str(len(records)) + " примеров x модель '" + args.model + "' -> " + str(output_path))

    t_start = time.time()
    ok = 0
    fail = 0
    with output_path.open("w", encoding="utf-8") as out_fh:
        for i, record in enumerate(records, 1):
            ex_id = record["example_id"]
            t0 = time.time()

            if args.verbose:
                print("\n[" + str(i) + "/" + str(len(records)) + "] " + ex_id +
                      " (source=" + record["source"] + ")")

            try:
                if args.verbose:
                    print("    -> Этап 1 (pdf_extractor)...")
                markdown, md_stats = load_markdown(record, REPO)
                if args.verbose:
                    print("      markdown готов: " + str(md_stats.get("markdown_len")) +
                          " chars, pages=" + str(md_stats.get("n_pages")) +
                          ", tables=" + str(md_stats.get("n_tables")))

                artifacts = process_record(record, markdown, args.model, api_key,
                                           verbose=args.verbose)

                if args.verbose:
                    print("    -> eval...")
                metrics = eval_record(record, artifacts, markdown)
                if args.verbose:
                    s3 = metrics["stage3"]
                    if s3["f1"] is not None:
                        print("      Stage 3: TP=" + str(s3["tp"]) + ", FP=" + str(s3["fp"]) +
                              ", FN=" + str(s3["fn"]) + ", F1=" + format(s3["f1"], ".3f"))
                    else:
                        print("      Stage 3: TP=" + str(s3["tp"]) + ", FP=" + str(s3["fp"]) +
                              ", FN=" + str(s3["fn"]))

                result = {
                    "example_id": ex_id,
                    "source": record["source"],
                    "source_id": record.get("source_id"),
                    "model": args.model,
                    "markdown_stats": md_stats,
                    "n_gold_tests": len(record["tests"]),
                    "n_predicted_tests": len(artifacts["predicted_tests"]),
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
                    "error": type(e).__name__ + ": " + str(e),
                    "traceback": traceback.format_exc().splitlines()[-3:],
                }
                fail += 1

            out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_fh.flush()

            dt = time.time() - t0
            eta_s = (time.time() - t_start) / i * (len(records) - i)
            if result.get("error") is None:
                status = "OK "
            else:
                status = "ERR"

            if args.verbose:
                if result.get("error"):
                    err_note = "  ERROR: " + str(result.get("error"))
                else:
                    err_note = ""
                print("    " + status + " " + format(dt, ".1f") + "s | ETA " +
                      format(eta_s / 60, ".1f") + "m" + err_note)
            else:
                print("[" + format(i, "3d") + "/" + str(len(records)) + "] " + status + " " +
                      ex_id.ljust(20) + " " + format(dt, "5.1f") + "s  ETA " +
                      format(eta_s / 60, ".1f") + "m")

    total = time.time() - t_start
    print("\nDone: " + str(ok) + " ok, " + str(fail) + " failed, " +
          format(total / 60, ".1f") + " min total -> " + str(output_path))


if __name__ == "__main__":
    main()
