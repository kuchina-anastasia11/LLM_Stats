# прогон с экспериментом 3 (LLM-as-judge cross-verify)
#
# схема: extractor извлекает -> verifier проверяет против markdown ->
# финал = verified + corrections + added (missed_tests от verifier)
#
# можно bidirectional (--bidirectional) и с суммаризацией (--summarise)
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
from pipeline.experiments import cross_verify, summarization


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


def process_record_one_direction(record, markdown, extractor, verifier, api_key,
                                 summarise, min_doc_len, verbose=False):
    # A -> B: extractor извлекает, verifier сверяет с текстом
    def log(msg):
        if verbose:
            print("    " + msg, flush=True)

    if summarise:
        log("-> Этап 2 (summarization)...")
        markdown, sum_stats = summarization.summarize_markdown(
            markdown, model_name=extractor, api_key=api_key, min_doc_len=min_doc_len,
        )
        if sum_stats["applied"]:
            log("  суммаризация: " + str(sum_stats["n_calls"]) + " вызовов, " +
                str(sum_stats["len_before"]) + " -> " + str(sum_stats["len_after"]) + " chars")
    else:
        sum_stats = {"applied": False, "skip_reason": "flag_off"}

    # stage 3a
    log("-> Stage 3a: " + extractor + " extracts...")
    t0 = time.time()
    proposed = stats_extractor.extract_tests(markdown, model_name=extractor, api_key=api_key)
    log("  proposed " + str(len(proposed)) + " tests (" + format(time.time() - t0, ".1f") + "s)")

    # stage 3b - verifier
    log("-> Stage 3b: " + verifier + " verifies against text...")
    t0 = time.time()
    xv = cross_verify.cross_verify(markdown, proposed, verifier_model=verifier,
                                   api_key=api_key, clean=True)
    diag = xv["diagnostics"]
    log("  verified=" + str(diag["n_verified"]) + " (corrected=" + str(diag["n_corrected"]) + "), " +
        "removed=" + str(diag["n_removed"]) + ", added=" + str(diag["n_added"]) +
        ", final=" + str(diag["n_final"]) + " (" + format(time.time() - t0, ".1f") + "s)")

    final_tests = xv["final_tests"]

    # stage 4 - на финальном списке через verifier
    sections = interpritation_extractor.split_sections(markdown)
    if final_tests:
        log("-> Stage 4: " + str(len(final_tests)) + " tests via " + verifier + "...")
        t0 = time.time()
        interpretations = interpritation_extractor.extract_interpretations(
            sections, final_tests, model_name=verifier, api_key=api_key,
        )
        n_with = sum(1 for r in interpretations if r.get("interpretations"))
        log("  stage 4: " + str(n_with) + "/" + str(len(final_tests)) +
            " (" + format(time.time() - t0, ".1f") + "s)")
    else:
        interpretations = []

    merged = interpritation_extractor.merge_with_tests(final_tests, interpretations)
    verified_stage5 = test_verificator.verify_all(merged)

    return {
        "predicted_tests": final_tests,
        "sections_found": sorted(sections.keys()),
        "merged": merged,
        "verified": verified_stage5,
        "summarization_stats": sum_stats,
        "cross_verify_diagnostics": diag,
        "cross_verify_records": {
            "verified_records": xv["verified_records"],
            "removed_records": xv["removed_records"],
            "added_records": xv["added_records"],
            "unmentioned_ids": xv.get("unmentioned_ids", []),
        },
    }


def process_record_bidirectional(record, markdown, model_a, model_b, api_key,
                                 summarise, min_doc_len, verbose=False):
    # A <-> B: оба извлекают и взаимно проверяют
    def log(msg):
        if verbose:
            print("    " + msg, flush=True)

    if summarise:
        log("-> Этап 2 (summarization)...")
        markdown, sum_stats = summarization.summarize_markdown(
            markdown, model_name=model_a, api_key=api_key, min_doc_len=min_doc_len,
        )
    else:
        sum_stats = {"applied": False, "skip_reason": "flag_off"}

    log("-> Stage 3a: " + model_a + " extracts...")
    proposed_a = stats_extractor.extract_tests(markdown, model_name=model_a, api_key=api_key)
    log("  A proposed " + str(len(proposed_a)))

    log("-> Stage 3b: " + model_b + " extracts...")
    proposed_b = stats_extractor.extract_tests(markdown, model_name=model_b, api_key=api_key)
    log("  B proposed " + str(len(proposed_b)))

    log("-> Stage 3c: bidirectional verify...")
    xv = cross_verify.cross_verify_bidirectional(
        markdown, proposed_a, proposed_b,
        model_a=model_a, model_b=model_b,
        api_key_a=api_key, api_key_b=api_key, clean=True,
    )
    diag = xv["diagnostics"]
    log("  A->B final=" + str(diag["n_a_final"]) +
        ", B->A final=" + str(diag["n_b_final"]) +
        ", cross_confirmed=" + str(diag["n_cross_confirmed"]) +
        ", total=" + str(diag["n_total_final"]))

    final_tests = xv["final_tests"]

    sections = interpritation_extractor.split_sections(markdown)
    if final_tests:
        # для интерпретаций берём model_b
        interpretations = interpritation_extractor.extract_interpretations(
            sections, final_tests, model_name=model_b, api_key=api_key,
        )
    else:
        interpretations = []

    merged = interpritation_extractor.merge_with_tests(final_tests, interpretations)
    verified_stage5 = test_verificator.verify_all(merged)

    return {
        "predicted_tests": final_tests,
        "sections_found": sorted(sections.keys()),
        "merged": merged,
        "verified": verified_stage5,
        "summarization_stats": sum_stats,
        "cross_verify_diagnostics": diag,
        "cross_verify_records": {
            "out_a_to_b_diag": xv["out_a_to_b"]["diagnostics"],
            "out_b_to_a_diag": xv["out_b_to_a"]["diagnostics"],
        },
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
    ap.add_argument("--extractor", required=True,
                    choices=["deepseek", "gemini", "openrouter-deepseek", "openrouter-gemini"])
    ap.add_argument("--verifier", required=True,
                    choices=["deepseek", "gemini", "openrouter-deepseek", "openrouter-gemini"])
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--skip-synth", action="store_true")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--min-doc-len", type=int, default=summarization.MIN_DOC_LEN)
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    # для bidirectional оба провайдера через один env_key
    env_key = stats_extractor.MODELS[args.extractor]["env_key"]
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

    if args.bidirectional:
        mode_label = "bidirectional"
    else:
        mode_label = args.extractor + "->" + args.verifier
    print("-> " + str(len(records)) + " examples x cross-verify (" + mode_label + ") x summarise=" +
          str(args.summarise) + " -> " + str(output_path))

    t_start = time.time()
    ok = fail = 0
    with output_path.open("w", encoding="utf-8") as out_fh:
        for i, record in enumerate(records, 1):
            ex_id = record["example_id"]
            t0 = time.time()
            if args.verbose:
                print("\n[" + str(i) + "/" + str(len(records)) + "] " + ex_id +
                      " (source=" + record["source"] + ")")
            try:
                markdown, md_stats = load_markdown(record, REPO)
                if args.bidirectional:
                    artifacts = process_record_bidirectional(
                        record, markdown, args.extractor, args.verifier, api_key,
                        summarise=args.summarise, min_doc_len=args.min_doc_len,
                        verbose=args.verbose,
                    )
                else:
                    artifacts = process_record_one_direction(
                        record, markdown, args.extractor, args.verifier, api_key,
                        summarise=args.summarise, min_doc_len=args.min_doc_len,
                        verbose=args.verbose,
                    )
                metrics = eval_record(record, artifacts, markdown)

                if args.bidirectional:
                    model_label = args.extractor + "<->" + args.verifier
                    exp_label = "cross_verify_bidir"
                else:
                    model_label = args.extractor + "->" + args.verifier
                    exp_label = "cross_verify_unidir"
                if args.summarise:
                    exp_label = exp_label + "+summarise"

                result = {
                    "example_id": ex_id,
                    "source": record["source"],
                    "source_id": record.get("source_id"),
                    "extractor": args.extractor,
                    "verifier": args.verifier,
                    "model": model_label,
                    "experiment": exp_label,
                    "markdown_stats": md_stats,
                    "n_gold_tests": len(record["tests"]),
                    "n_predicted_tests": len(artifacts["predicted_tests"]),
                    "sections_found": artifacts["sections_found"],
                    "summarization_stats": artifacts["summarization_stats"],
                    "cross_verify_diagnostics": artifacts["cross_verify_diagnostics"],
                    "cross_verify_records": artifacts["cross_verify_records"],
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
                    "model": args.extractor + "->" + args.verifier,
                    "experiment": "cross_verify",
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
    print("\nDone: " + str(ok) + " ok, " + str(fail) + " failed, " +
          format(total / 60, ".1f") + " min -> " + str(output_path))


if __name__ == "__main__":
    main()
