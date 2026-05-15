# помощник для ручной разметки real-статей со сценарием C
# (p корректный, текст противоречит числам)
#
# принимает JSON-массив записей, валидирует, считает computed_p, кладёт в data/real_with_errors/
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline.test_verificator import compute_p, check_consistency, check_interpretation


REQUIRED_FIELDS = [
    "test_type", "statistic_value", "df1", "reported_p",
    "p_equality", "two_tailed", "raw_text",
    "textual_interpretation", "interpretation_direction",
    "consistent", "error_type", "notes",
]

VALID_TEST_TYPES = {"t", "F", "chi", "z", "r", "Q"}
VALID_DIRECTIONS = {"significant", "not_significant", "marginal", "unclear"}


def validate(record):
    # возвращаем список проблем
    problems = []
    for f in REQUIRED_FIELDS:
        if f not in record:
            problems.append("missing field: " + f)
    if record.get("test_type") not in VALID_TEST_TYPES:
        problems.append("test_type должен быть из " + str(VALID_TEST_TYPES))
    if record.get("interpretation_direction") not in VALID_DIRECTIONS:
        problems.append("interpretation_direction должен быть из " + str(VALID_DIRECTIONS))
    valid_errs = ("real_wrong_conclusion", "real_wrong_pvalue",
                  "real_rounding", "real_transcription", "real_other", None)
    if record.get("error_type") not in valid_errs:
        problems.append("error_type не из ожидаемого набора")
    if record.get("consistent") not in (True, False, None):
        problems.append("consistent должно быть True/False/null")
    return problems


def diagnose(record):
    # сравниваем reported и computed p, классифицируем сценарий
    cp = compute_p(
        test_type=record.get("test_type"),
        statistic_value=record.get("statistic_value"),
        df1=record.get("df1"),
        df2=record.get("df2"),
        two_tailed=record.get("two_tailed", True),
    )
    reported = record.get("reported_p")
    direction = record.get("interpretation_direction")

    diag = {
        "computed_p": cp,
        "reported_p": reported,
        "p_consistency": check_consistency(reported, cp, record.get("p_equality")),
        "interp_consistency": check_interpretation(direction, cp),
    }

    # классификация сценария
    if cp is None:
        diag["scenario"] = "not_checkable"
    else:
        if cp <= 0.05:
            true_sig = "significant"
        else:
            true_sig = "not_significant"

        if direction in ("significant", "not_significant"):
            author_says_truth = (direction == true_sig)
        else:
            author_says_truth = None

        p_match = (
            reported is not None
            and abs(reported - cp) / max(abs(cp), 1e-10) < 0.10
        )

        if author_says_truth is None:
            diag["scenario"] = "unclear_direction"
        elif p_match and author_says_truth:
            diag["scenario"] = "A — нет ошибки"
        elif p_match and not author_says_truth:
            diag["scenario"] = "C — ЧИСТЫЙ wrong_conclusion (главный кейс)"
        elif not p_match and author_says_truth:
            diag["scenario"] = "B — wrong reported_p (только число неверно)"
        else:
            diag["scenario"] = "D — числа и текст оба сломаны"
    return diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True,
                    help="идентификатор статьи (станет именем файла)")
    ap.add_argument("--input", required=True,
                    help="JSON-файл со списком записей")
    ap.add_argument("--output-dir", default="data/real_with_errors",
                    help="куда сохранить итог")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print("ERROR: " + str(in_path) + " не найден", file=sys.stderr)
        sys.exit(1)

    with in_path.open() as f:
        records = json.load(f)

    if not isinstance(records, list):
        print("ERROR: ожидается JSON-массив записей", file=sys.stderr)
        sys.exit(1)

    print("Validating " + str(len(records)) + " records...")
    n_clean_c = 0
    for i, rec in enumerate(records):
        problems = validate(rec)
        diag = diagnose(rec)
        if "C" in diag["scenario"]:
            n_clean_c += 1
        if problems:
            flag = "[X]"
        elif "C" in diag["scenario"]:
            flag = "[*]"
        else:
            flag = "[ok]"

        if diag.get("computed_p") is not None:
            print("  " + flag + " #" + str(i) + ": " + str(rec.get("test_type")) +
                  " stat=" + str(rec.get("statistic_value")) +
                  " reported_p=" + str(rec.get("reported_p")) +
                  " -> computed_p=" + format(diag["computed_p"], ".4f") +
                  " (" + diag["scenario"] + ")")
        else:
            print("  " + flag + " #" + str(i) + ": " + diag["scenario"])

        for p in problems:
            print("     ! " + p)

    out_dir = REPO / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (args.slug + ".json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print("\nWrote " + str(len(records)) + " records -> " + str(out_path))
    print("Чистых C-сценариев: " + str(n_clean_c))
    print()
    print("Дальше - добавить статью в scripts/build_datasets.py:")
    print('  ARTICLES["' + args.slug + '"] = ("real_with_errors/' + args.slug + '.json", "<имя_pdf>")')


if __name__ == "__main__":
    main()
