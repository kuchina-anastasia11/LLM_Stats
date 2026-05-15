# композитный раннер: --stage3 × --stage4
#
# Stage 3:
#   baseline    — stats_extractor.extract_tests
#   few-shot    — few_shot_tables.extract_tests_few_shot
#   table-sweep — table_sweep.extract_tests_two_pass
#
# Stage 4:
#   baseline    — interpritation_extractor.extract_interpretations
#   cot         — cot_stage4.extract_interpretations_cot
#
# usage:
#   python3 scripts/run_combined.py \
#       --dataset data/dev_dataset.jsonl \
#       --model openrouter-gemini \
#       --stage3 table-sweep --stage4 cot \
#       --output results/dev_table_sweep_cot_or_gemini.jsonl
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
from pipeline.experiments import few_shot_tables, table_sweep, cot_stage4, summarization


STAGE3_REGISTRY = {
    "baseline":    stats_extractor.extract_tests,
    "few-shot":    few_shot_tables.extract_tests_few_shot,
    "table-sweep": table_sweep.extract_tests_two_pass,
}

# stage 4 — пара (extract_func, merge_func)
STAGE4_REGISTRY = {
    "baseline": (
        interpritation_extractor.extract_interpretations,
        interpritation_extractor.merge_with_tests,
    ),
    "cot": (
        cot_stage4.extract_interpretations_cot,
        cot_stage4.merge_with_tests_cot,
    ),
}


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


def process_record(record, markdown, model, api_key, stage3_fn, stage4_extract, stage4_merge,
                   summarise=False, min_doc_len=summarization.MIN_DOC_LEN, verbose=False):
    def log(msg):
        if verbose:
            print("    " + msg, flush=True)

    # опциональный этап 2 — суммаризация секций
    if summarise:
        log("-> Stage 2 (summarization)...")
        t0 = time.time()
        markdown, sum_stats = summarization.summarize_markdown(
            markdown, model_name=model, api_key=api_key, min_doc_len=min_doc_len,
        )
        if sum_stats["applied"]:
            log("  summarize: " + str(sum_stats["n_sections"]) + " sections, " +
                str(sum_stats["len_before"]) + " -> " + str(sum_stats["len_after"]) + " chars (" +
                format(time.time() - t0, ".1f") + "s)")
        else:
            log("  summarize пропущено (" + str(sum_stats["skip_reason"]) + ")")
    else:
        sum_stats = {"applied": False, "skip_reason": "flag_off"}

    # stage 3
    log("-> Stage 3...")
    t0 = time.time()
    predicted_tests = stage3_fn(markdown, model_name=model, api_key=api_key)
    log("  Stage 3: " + str(len(predicted_tests)) + " тестов (" +
        format(time.time() - t0, ".1f") + "s)")

    sections = interpritation_extractor.split_sections(markdown)
    section_names = sorted(sections.keys())

    # stage 4
    if predicted_tests:
        log("-> Stage 4 (" + stage4_extract.__name__ + ")...")
        t0 = time.time()
        interpretations = stage4_extract(
            sections, predicted_tests, model_name=model, api_key=api_key,
        )
        n_with_interp = sum(1 for r in interpretations if r.get("interpretations"))
        log("  Stage 4: " + str(n_with_interp) + "/" + str(len(predicted_tests)) +
            " (" + format(time.time() - t0, ".1f") + "s)")
    else:
        interpretations = []

    merged = stage4_merge(predicted_tests, interpretations)

    # stage 5 — детерминированный, общий для всех конфигураций
    verified = test_verificator.verify_all(merged)

    n_conflict = sum(1 for m in merged if m.get("has_text_vs_numbers_conflict"))

    return {
        "predicted_tests": predicted_tests,
        "sections_found": section_names,
        "merged": merged,
        "verified": verified,
        "n_text_vs_numbers_conflict": n_conflict,
        "summarization_stats": sum_stats,
    }


def eval_record(record, artifacts, markdown):
    gold_tests = record["tests"]
    if record["source"] == "real":
        stage1 = eval_mod.raw_text_coverage(gold_tests, markdown)
    else:
        stage1 = None
    stage3 = eval_mod.eval_stage3(gold_tests, artifacts["predicted_tests"])
    pairs = stage3.get("pairs", [])
    stage4 = eval_mod.eval_stage4(gold_tests, artifacts["merged"], pairs)
    has_errors = any(t.get("consistent") is False for t in gold_tests)
    stage5 = eval_mod.eval_stage5_example(
        gold_tests, artifacts["verified"], pairs, example_has_errors=has_errors,
    )
    return {"stage1": stage1, "stage3": stage3, "stage4": stage4, "stage5": stage5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True,
                    choices=["deepseek", "gemini", "openrouter-deepseek", "openrouter-gemini"])
    ap.add_argument("--stage3", choices=list(STAGE3_REGISTRY.keys()), default="baseline")
    ap.add_argument("--stage4", choices=list(STAGE4_REGISTRY.keys()), default="baseline")
    ap.add_argument("--summarise", action="store_true",
                    help="включить stage 2 (summarization), пропускается на короткой синтетике")
    ap.add_argument("--min-doc-len", type=int, default=summarization.MIN_DOC_LEN,
                    help="порог длины документа для активации суммаризации")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    stage3_fn = STAGE3_REGISTRY[args.stage3]
    stage4_extract, stage4_merge = STAGE4_REGISTRY[args.stage4]

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

    print("-> " + str(len(records)) + " примеров x stage3=" + args.stage3 +
          " x stage4=" + args.stage4 + " x [" + args.model + "] -> " + str(output_path))

    t_start = time.time()
    ok = 0
    fail = 0
    with output_path.open("w", encoding="utf-8") as out_fh:
        for i, record in enumerate(records, 1):
            ex_id = record["example_id"]
            t0 = time.time()
            try:
                markdown, md_stats = load_markdown(record, REPO)
                artifacts = process_record(
                    record, markdown, args.model, api_key,
                    stage3_fn, stage4_extract, stage4_merge,
                    summarise=args.summarise,
                    min_doc_len=args.min_doc_len,
                    verbose=args.verbose,
                )
                metrics = eval_record(record, artifacts, markdown)
                result = {
                    "example_id": ex_id,
                    "source": record["source"],
                    "source_id": record.get("source_id"),
                    "model": args.model,
                    "experiment": (
                        "stage3=" + args.stage3 + ";stage4=" + args.stage4
                        + (";summarise" if args.summarise else "")
                    ),
                    "markdown_stats": md_stats,
                    "n_gold_tests": len(record["tests"]),
                    "n_predicted_tests": len(artifacts["predicted_tests"]),
                    "sections_found": artifacts["sections_found"],
                    "n_text_vs_numbers_conflict": artifacts["n_text_vs_numbers_conflict"],
                    "summarization_stats": artifacts["summarization_stats"],
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
                    "experiment": "stage3=" + args.stage3 + ";stage4=" + args.stage4,
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
