import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from pipeline.pdf_extractor import extract_pdf

ROOT = Path(__file__).resolve().parent
R_SCRIPT = ROOT / "statcheck_runner.R"
REPO_ROOT = ROOT.parent

STATCHECK_TO_OURS = {
    "t": "t",
    "F": "F",
    "Chi2": "chi",
    "chi2": "chi",
    "Z": "z",
    "z": "z",
    "r": "r",
    "Q": "Q",
}


def run_statcheck_on_text(text: str, timeout: int = 60) -> list:
    if not text or not text.strip():
        return []
    try:
        proc = subprocess.run(
            ["Rscript", str(R_SCRIPT)],
            input=text,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=ROOT,
        )
    except subprocess.TimeoutExpired:
        sys.stderr.write("statcheck timeout\n")
        return []
    except FileNotFoundError:
        sys.stderr.write(
            "Rscript не найден. Установи R: bash install_statcheck.sh\n"
        )
        sys.exit(1)

    if proc.returncode != 0:
        sys.stderr.write(f"statcheck failed (rc={proc.returncode}): {proc.stderr}\n")
        return []

    raw = (proc.stdout or "").strip()
    if not raw:
        return []

    rows = None
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        import re as _re
        # ищем массив верхнего уровня в конце вывода
        matches = list(_re.finditer(r"\[.*?\](?=\s*$|\s*\Z)", raw, _re.DOTALL))
        for m in reversed(matches):
            try:
                rows = json.loads(m.group(0))
                break
            except json.JSONDecodeError:
                continue
        if rows is None:
            # последний шанс — "did not find any results" в любой комбинации
            if "did not find" in raw.lower() or raw.rstrip().endswith("[]"):
                rows = []
            else:
                sys.stderr.write(f"JSON parse error; raw={raw[:200]!r}\n")
                return []

    if not isinstance(rows, list):
        return []
    return [_normalize(r) for r in rows]


def _normalize(row: dict) -> dict:
    test_type = STATCHECK_TO_OURS.get(row.get("test_type"), row.get("test_type"))
    df1 = row.get("df1")
    df2 = row.get("df2")
    if test_type in ("t", "z", "chi", "r", "Q"):
        # единственное df — сводим в df1
        single_df = df1 if df1 is not None else df2
        df1, df2 = single_df, None

    p_error = row.get("p_error")
    decision_error = row.get("decision_error")
    if p_error is True:
        p_consistency = "inconsistent"
    elif p_error is False:
        p_consistency = "consistent"
    else:
        p_consistency = "not_checkable"

    return {
        "test_type": test_type,
        "statistic_value": row.get("statistic_value"),
        "df1": df1,
        "df2": df2,
        "reported_p": row.get("reported_p"),
        "p_equality": row.get("p_equality"),
        "two_tailed": True,
        "raw_text": row.get("raw_text") or "",
        "textual_interpretation": "",
        "interpretation_direction": "unclear",
        "computed_p": row.get("computed_p"),
        "p_consistency": p_consistency,
        "decision_error": decision_error,
        "interpretation_consistency": "not_checkable",
        "_source": "statcheck",
    }


def _load_text_for_record(ex: dict, repo: Path) -> tuple[str, str]:
    source = ex.get("source")
    if source == "synthetic":
        return ex.get("fragment") or "", "synthetic-fragment"

    # real
    pdf_rel = ex.get("pdf_path")
    if pdf_rel:
        pdf_abs = repo / pdf_rel
        if pdf_abs.exists():
            try:
                sys.path.insert(0, str(repo))
                from pipeline.pdf_extractor import extract as extract_pdf
                doc = extract_pdf(str(pdf_abs))
                return doc.text, "real-pdf-extracted"
            except Exception as e:
                sys.stderr.write(
                    f"PDF extract failed for {pdf_rel}: {e}\n"
                    f"  fallback to fragment\n"
                )
    return ex.get("fragment") or "", "real-fragment-fallback"


def process_dataset(dataset_path: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    examples = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    n = len(examples)
    t0 = time.time()
    with output_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(examples, 1):
            text, src_mode = _load_text_for_record(ex, REPO_ROOT)
            preds = run_statcheck_on_text(text)
            out = {
                "example_id": ex.get("example_id"),
                "source": ex.get("source"),
                "source_id": ex.get("source_id"),
                "domain": ex.get("domain"),
                "environment": ex.get("environment"),
                "error_type": ex.get("error_type"),
                "text_source_mode": src_mode,
                "text_len": len(text),
                "n_gold_tests": len(ex.get("tests") or []),
                "n_predicted_tests": len(preds),
                "predicted_tests": preds,
                "model": "statcheck",
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            if i % 25 == 0 or i == n:
                dt = time.time() - t0
                print(
                    f"[{i}/{n}] {dt:.1f}s; predicted={len(preds)}; "
                    f"mode={src_mode}; id={ex.get('example_id')}"
                )
    print(f"{output_path}")


def process_pdfs(pdf_paths: list, output_path: Path):
    sys.path.insert(0, str(REPO_ROOT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    is_jsonl = output_path.suffix == ".jsonl"
    fh = output_path.open("w", encoding="utf-8") if is_jsonl else None

    try:
        for pdf in pdf_paths:
            print(f"Extracting {pdf.name}")
            try:
                doc = extract_pdf(str(pdf))
                text = doc.text if hasattr(doc, "text") else doc
            except Exception as e:
                sys.stderr.write(f"PDF extract failed for {pdf}: {e}\n")
                continue
            preds = run_statcheck_on_text(text, timeout=180)
            record = {
                "example_id": pdf.stem,
                "source": "real_paper",
                "environment": "real",
                "n_predicted_tests": len(preds),
                "predicted_tests": preds,
                "model": "statcheck",
                "pdf_path": str(pdf),
            }
            if is_jsonl:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                output_path.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            print(f"   {len(preds)} тестов")
    finally:
        if fh:
            fh.close()
    print(f"Готово {output_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", type=Path)
    src.add_argument("--pdf", type=Path)
    src.add_argument("--pdf-dir", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if args.dataset:
        process_dataset(args.dataset, args.output)
    elif args.pdf:
        process_pdfs([args.pdf], args.output)
    else:
        pdfs = sorted(args.pdf_dir.glob("*.pdf"))
        if not pdfs:
            ap.error(f"PDF-файлы не найдены в {args.pdf_dir}")
        process_pdfs(pdfs, args.output)


if __name__ == "__main__":
    main()
