# прогон с экспериментом 5 (few-shot для таблиц на stage 3)
#
# отличие: stage 3 использует few_shot_tables.extract_tests_few_shot
# few-shot блок добавляется только если в тексте есть markdown-таблица
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
from pipeline.experiments import few_shot_tables, summarization


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


def process_record(record, markdown, model, api_key, summarise, min_doc_len, verbose=False):
    def log(msg):
        if verbose:
            print("    " + msg, flush=True)

    if summarise:
        log("-> Этап 2 (summarization)...")
        t0 = time.time()
        markdown, sum_stats = summarization.summarize_markdown(
            markdown, model_name=model, api_key=api_key, min_doc_len=min_doc_len,
        )
        if sum_stats["applied"]:
            log("  суммаризация: " + str(sum_stats["n_calls"]) + " вызовов, " +
                str(sum_stats["len_before"]) + " -> " + str(sum_stats["len_after"]) + " chars (" +
                format(time.time() - t0, ".1f") + "s)")
        else:
            log("  суммаризация пропущена (" + str(sum_stats["skip_reason"]) + ")")
    else:
        sum_stats = {"applied": False, "skip_reason": "flag_off"}

    # stage 3 - few-shot
    fs_applied = few_shot_tables.has_markdown_table(markdown)
    log("-> Этап 3 (few-shot tables, fs_applied=" + str(fs_applied) + ")...")
    t0 = time.time()
    predicted_tests = few_shot_tables.extract_tests_few_shot(
        markdown, model_name=model, api_key=api_key,
    )
    log("  Этап 3: " + str(len(predicted_tests)) + " тестов (" + format(time.time() - t0, ".1f") + "s)")

    log("-> split_sections...")
    sections = interpritation_extractor.split_sections(markdown)
    section_names = sorted(sections.keys())
    log("  секций: " + str(len(section_names)))

    if predicted_tests:
        log("-> Этап 4: " + str(len(predicted_tests)) + " тестов...")
        t0 = time.time()
        interpretations = interpritation_extractor.extract_interpretations(
            sections, predicted_tests, model_name=model, api_key=api_key
        )
        n_with_interp = sum(1 for r in interpretations if r.get("interpretations"))
        log("  Этап 4: " + str(n_with_interp) + "/" + str(len(predicted_tests)) +
            " (" + format(time.time() - t0, ".1f") + "s)")
    else:
        log("-> Этап 4: пропускаю (нет тестов)")
        interpretations = []

    merged = interpritation_extractor.merge_with_tests(predicted_tests, interpretations)

    log("-> Этап 5 (verificator)...")
    verified = test_verificator.verify_all(merged)

    return {
        "predicted_tests": predicted_tests,
        "sections_found": section_names,
        "merged": merged,
        "verified": verified,
        "summarization_stats": sum_stats,
        "few_shot_applied": fs_applied,
    }


def eval_record(record, artifacts, original_markdown):
    gold_tests = record["tests"]
    predicted_tests = artifacts["predicted_tests"]
    merged = artifacts["merged"]
    verified = artifacts["verified"]

    if record["source"] == "real":
        stage1 = eval_mod.raw_text_coverage(gold_tests, original_markdown)
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
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--min-doc-len", type=int, default=summarization.MIN_DOC_LEN)
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

    print("-> " + str(len(records)) + " examples x model=" + args.model + " x FewShot x summarise=" +
          str(args.summarise) + " -> " + str(output_path))

    t_start = time.time()
    ok = fail = 0
    n_fs_applied = 0
    with output_path.open("w", encoding="utf-8") as out_fh:
        for i, record in enumerate(records, 1):
            ex_id = record["example_id"]
            t0 = time.time()
            if args.verbose:
                print("\n[" + str(i) + "/" + str(len(records)) + "] " + ex_id +
                      " (source=" + record["source"] + ")")
            try:
                markdown, md_stats = load_markdown(record, REPO)
                artifacts = process_record(
                    record, markdown, args.model, api_key,
                    summarise=args.summarise, min_doc_len=args.min_doc_len,
                    verbose=args.verbose,
                )
                if artifacts["few_shot_applied"]:
                    n_fs_applied += 1
                metrics = eval_record(record, artifacts, markdown)

                if args.summarise:
                    exp_label = "few_shot+summarise"
                else:
                    exp_label = "few_shot"

                result = {
                    "example_id": ex_id,
                    "source": record["source"],
                    "source_id": record.get("source_id"),
                    "model": args.model,
                    "experiment": exp_label,
                    "markdown_stats": md_stats,
                    "n_gold_tests": len(record["tests"]),
                    "n_predicted_tests": len(artifacts["predicted_tests"]),
                    "sections_found": artifacts["sections_found"],
                    "few_shot_applied": artifacts["few_shot_applied"],
                    "summarization_stats": artifacts["summarization_stats"],
                    "predicted_tests": artifacts["predicted_tests"],
                    "merged_tests": artifacts["merged"],
                    "verified_tests": artifacts["verified"],
                    "error": None,
                }
                result.update(metrics)
                ok += 1
            except Exception as e:
                if args.summarise:
                    exp_label = "few_shot+summarise"
                else:
                    exp_label = "few_shot"
                result = {
                    "example_id": ex_id,
                    "source": record["source"],
                    "model": args.model,
                    "experiment": exp_label,
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
                    err = "  ERROR: " + str(result.get("error"))
                else:
                    err = ""
                print("    " + status + " " + format(dt, ".1f") + "s | ETA " +
                      format(eta_s / 60, ".1f") + "m" + err)
            else:
                print("[" + format(i, "3d") + "/" + str(len(records)) + "] " + status + " " +
                      ex_id.ljust(25) + " " + format(dt, "5.1f") + "s ETA " +
                      format(eta_s / 60, ".1f") + "m")

    total = time.time() - t_start
    print("\nDone: " + str(ok) + " ok, " + str(fail) + " failed, few-shot applied to " +
          str(n_fs_applied) + " examples, " +
          format(total / 60, ".1f") + " min -> " + str(output_path))


if __name__ == "__main__":
    main()
