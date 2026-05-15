import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SEED = 42

ARTICLES = {
    "mbct": (
        "mbct_perceived_stress_psychosomatic_chinese_adolescent_girls.json",
        "40359_2025_Article_3472.pdf",
    ),
    "multicomponent": (
        "multicomponent_multiprofessional_interventions_cardiovascular_hypertensive_older_women.json",
        "jcm-15-00572.pdf",
    ),
    "sleep_deprivation": (
        "sleep_deprivation_cognitive_performance_ultramarathon.json",
        "journal.pone.0299475.pdf",
    ),
    "stress_burnout": (
        "stress_burnout_workplace_psychosocial_factors_mental_health_covid19_hungary.json",
        "fpsyt-15-1354612.pdf",
    ),
}


def load_synthetic():
    with (DATA / "synthetic_dataset.jsonl").open() as f:
        return [json.loads(line) for line in f]


def load_clean_c():
    # подгружаем clean C если есть, иначе пустой список
    path = DATA / "clean_c_dataset.jsonl"
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f]


def derive_interpretation_direction(test, alpha=0.05):
    # выводим gold direction из p_value и error_type
    error_type = test.get("error_type", "none")
    true_p = test.get("p_value")
    reported_p = test.get("reported_p")

    if true_p is None:
        return "unclear"

    if true_p <= alpha:
        true_sig = "significant"
    else:
        true_sig = "not_significant"

    if error_type == "none":
        return true_sig

    if error_type in ("wrong_conclusion", "wrong_conclusion_clean"):
        # автор пишет противоположный вывод
        if true_sig == "significant":
            return "not_significant"
        return "significant"

    if reported_p is None:
        return true_sig
    if reported_p <= alpha:
        return "significant"
    return "not_significant"


def derive_textual_interpretation(direction):
    # шаблонная интерпретация
    if direction == "significant":
        return "the result was statistically significant"
    if direction == "not_significant":
        return "the difference was not statistically significant"
    if direction == "marginal":
        return "the effect approached marginal significance"
    return ""


def synthetic_to_unified(row, new_id):
    tests = []
    for t in row.get("tests", []):
        direction = derive_interpretation_direction(t)
        tests.append({
            "test_type": t.get("test_type"),
            "statistic_value": t.get("statistic"),
            "df1": t.get("df1"),
            "df2": t.get("df2"),
            "reported_p": t.get("reported_p"),
            "p_equality": t.get("p_equality"),
            "two_tailed": t.get("two_tailed", True),
            "raw_text": "",
            "textual_interpretation": derive_textual_interpretation(direction),
            "interpretation_direction": direction,
            "consistent": t.get("consistent"),
            "error_type": t.get("error_type"),
            "notes": "synthetic n_sample=" + str(t.get("n_sample")) +
                     ", p_value_true=" + str(t.get("p_value")),
        })
    return {
        "example_id": new_id,
        "source": "synthetic",
        "source_id": str(row["example_id"]),
        "domain": row.get("domain"),
        "environment": row.get("environment"),
        "error_type": row.get("error_type"),
        "label_consistent": row.get("label_consistent"),
        "fragment": row.get("fragment", ""),
        "pdf_path": None,
        "tests": tests,
    }


def real_article_to_unified(slug, new_id):
    json_name, pdf_name = ARTICLES[slug]
    with (DATA / json_name).open() as f:
        records = json.load(f)
    first = records[0]
    tests = []
    for r in records:
        tests.append({
            "test_type": r.get("test_type"),
            "statistic_value": r.get("statistic_value"),
            "df1": r.get("df1"),
            "df2": r.get("df2"),
            "reported_p": r.get("reported_p"),
            "p_equality": r.get("p_equality"),
            "two_tailed": r.get("two_tailed", True),
            "raw_text": r.get("raw_text", ""),
            "textual_interpretation": r.get("textual_interpretation", ""),
            "interpretation_direction": r.get("interpretation_direction", "unclear"),
            "consistent": r.get("consistent"),
            "error_type": None,
            "notes": r.get("notes", ""),
        })
    fragment = "\n\n".join(r.get("raw_text", "") for r in records if r.get("raw_text"))
    return {
        "example_id": new_id,
        "source": "real",
        "source_id": first.get("article_id", slug),
        "domain": first.get("journal"),
        "environment": None,
        "error_type": None,
        "label_consistent": None,
        "fragment": fragment,
        "pdf_path": "data/" + pdf_name,
        "tests": tests,
    }


def stratified_split(rows, n_per_split, seed):
    # стратифицированный 50/50 сплит по (environment, error_type)
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r.get("environment"), r.get("error_type"))].append(r)

    dev = []
    test = []
    leftover = []
    for items in buckets.values():
        rng.shuffle(items)
        half = len(items) // 2
        dev.extend(items[:half])
        test.extend(items[half:half * 2])
        if len(items) % 2 == 1:
            leftover.append(items[-1])

    rng.shuffle(leftover)
    while len(dev) < n_per_split and leftover:
        dev.append(leftover.pop())
    while len(test) < n_per_split and leftover:
        test.append(leftover.pop())

    assert len(dev) == n_per_split, "dev has " + str(len(dev)) + ", expected " + str(n_per_split)
    assert len(test) == n_per_split, "test has " + str(len(test)) + ", expected " + str(n_per_split)
    rng.shuffle(dev)
    rng.shuffle(test)
    return dev, test


def main():
    synth = load_synthetic()
    dev_synth, test_synth = stratified_split(synth, n_per_split=250, seed=SEED)
    print("Synthetic: dev=" + str(len(dev_synth)) + ", test=" + str(len(test_synth)))
    clean_c = load_clean_c()
    if clean_c:
        rng = random.Random(SEED + 1)
        rng.shuffle(clean_c)
        half = len(clean_c) // 2
        dev_clean_c = clean_c[:half]
        test_clean_c = clean_c[half:half * 2]
        print("Clean C: dev=" + str(len(dev_clean_c)) +
              ", test=" + str(len(test_clean_c)) +
              " (всего " + str(len(clean_c)) + ")")
    else:
        dev_clean_c = []
        test_clean_c = []
        print("Clean C: файл data/clean_c_dataset.jsonl не найден - пропускаю")

    dev_real = [
        real_article_to_unified("mbct", "real-mbct"),
        real_article_to_unified("stress_burnout", "real-stress_burnout"),
    ]
    test_real = [
        real_article_to_unified("multicomponent", "real-multicomponent"),
        real_article_to_unified("sleep_deprivation", "real-sleep_deprivation"),
    ]
    n_dev_real_tests = sum(len(r["tests"]) for r in dev_real)
    n_test_real_tests = sum(len(r["tests"]) for r in test_real)
    print("Real: dev=" + str(n_dev_real_tests) + " тестов в " + str(len(dev_real)) + " статьях, "
          "test=" + str(n_test_real_tests) + " тестов в " + str(len(test_real)) + " статьях")

    def write_split(path, synth_rows, real_rows, clean_c_rows, prefix):
        with path.open("w", encoding="utf-8") as fh:
            for i, r in enumerate(synth_rows):
                unified = synthetic_to_unified(r, new_id=prefix + "-syn-" + format(i, "03d"))
                fh.write(json.dumps(unified, ensure_ascii=False) + "\n")
            for i, r in enumerate(clean_c_rows):
                unified = synthetic_to_unified(r, new_id=prefix + "-cleanC-" + format(i, "03d"))
                fh.write(json.dumps(unified, ensure_ascii=False) + "\n")
            for r in real_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    dev_path = DATA / "dev_dataset.jsonl"
    test_path = DATA / "test_dataset.jsonl"
    write_split(dev_path, dev_synth, dev_real, dev_clean_c, prefix="dev")
    write_split(test_path, test_synth, test_real, test_clean_c, prefix="test")

    print("\nЗаписано:")
    for p in (dev_path, test_path):
        lines = p.read_text().splitlines()
        n_tests = sum(len(json.loads(l)["tests"]) for l in lines)
        print("  " + str(p.relative_to(REPO)) + ": " + str(len(lines)) + " записей, " +
              str(n_tests) + " тестов в gold")


if __name__ == "__main__":
    main()
