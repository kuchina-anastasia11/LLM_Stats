import argparse
import json
import math
import random
from pathlib import Path

from scipy import stats as scipy_stats


P_SIG_RANGE = (0.001, 0.04)
P_NSIG_RANGE = (0.10, 0.80)


# генераторы для каждого типа теста
# возвращают (stat, df1, df2|None, n_sample, true_p, p_equality)

def gen_t(rng, target_sig):
    # t-test
    df = rng.randint(20, 250)
    if target_sig:
        target_p = rng.uniform(*P_SIG_RANGE)
    else:
        target_p = rng.uniform(*P_NSIG_RANGE)
    abs_t = scipy_stats.t.isf(target_p / 2, df)
    stat = round(abs_t * rng.choice([-1, 1]), 2)
    # пересчёт фактического p
    true_p = scipy_stats.t.sf(abs(stat), df) * 2
    return stat, df, None, df + 1, float(true_p), "="


def gen_F(rng, target_sig):
    # F-test (ANOVA)
    df1 = rng.randint(1, 5)
    df2 = rng.randint(20, 250)
    if target_sig:
        target_p = rng.uniform(*P_SIG_RANGE)
    else:
        target_p = rng.uniform(*P_NSIG_RANGE)
    f_val = scipy_stats.f.isf(target_p, df1, df2)
    stat = round(float(f_val), 2)
    true_p = scipy_stats.f.sf(stat, df1, df2)
    return stat, df1, df2, df1 + df2 + 1, float(true_p), "="


def gen_chi(rng, target_sig):
    df = rng.randint(1, 8)
    if target_sig:
        target_p = rng.uniform(*P_SIG_RANGE)
    else:
        target_p = rng.uniform(*P_NSIG_RANGE)
    chi_val = scipy_stats.chi2.isf(target_p, df)
    stat = round(float(chi_val), 2)
    true_p = scipy_stats.chi2.sf(stat, df)
    return stat, df, None, rng.randint(50, 500), float(true_p), "="


def gen_z(rng, target_sig):
    if target_sig:
        target_p = rng.uniform(*P_SIG_RANGE)
    else:
        target_p = rng.uniform(*P_NSIG_RANGE)
    abs_z = scipy_stats.norm.isf(target_p / 2)
    stat = round(float(abs_z) * rng.choice([-1, 1]), 2)
    true_p = scipy_stats.norm.sf(abs(stat)) * 2
    return stat, None, None, rng.randint(50, 500), float(true_p), "="


def gen_r(rng, target_sig):
    # pearson r
    df = rng.randint(20, 250)
    if target_sig:
        target_p = rng.uniform(*P_SIG_RANGE)
    else:
        target_p = rng.uniform(*P_NSIG_RANGE)
    abs_t = scipy_stats.t.isf(target_p / 2, df)
    abs_r = abs_t / math.sqrt(abs_t ** 2 + df)
    stat = round(float(abs_r) * rng.choice([-1, 1]), 2)
    if 1 - stat ** 2 <= 0:
        true_p = 0.0
    else:
        t_recomputed = stat * math.sqrt(df) / math.sqrt(1 - stat ** 2)
        true_p = scipy_stats.t.sf(abs(t_recomputed), df) * 2
    return stat, df, None, df + 2, float(true_p), "="


def gen_Q(rng, target_sig):
    df = rng.randint(2, 10)
    if target_sig:
        target_p = rng.uniform(*P_SIG_RANGE)
    else:
        target_p = rng.uniform(*P_NSIG_RANGE)
    q_val = scipy_stats.chi2.isf(target_p, df)
    stat = round(float(q_val), 2)
    true_p = scipy_stats.chi2.sf(stat, df)
    return stat, df, None, df + 1, float(true_p), "="


GENERATORS = [
    ("t", gen_t),
    ("F", gen_F),
    ("chi", gen_chi),
    ("z", gen_z),
    ("r", gen_r),
    ("Q", gen_Q),
]


# шаблоны фрагмента
INTRO_TEMPLATES = [
    "We examined whether {topic}.",
    "The current study tested the hypothesis that {topic}.",
    "An analysis was conducted to determine if {topic}.",
    "We investigated the association between {topic_pair}.",
]

TOPICS = [
    "the intervention reduces patient anxiety",
    "academic performance varies by socioeconomic status",
    "physical activity correlates with cognitive function",
    "training improves response accuracy",
    "treatment alters serum biomarker levels",
    "exposure influences attitudinal change",
    "method differs between subgroups",
    "stress predicts perceived job satisfaction",
]

TOPIC_PAIRS = [
    "sleep duration and reaction time",
    "income level and life satisfaction",
    "drug dose and symptom severity",
    "training hours and exam scores",
    "diet quality and inflammation markers",
]

PURPOSES = [
    "compare outcomes between groups",
    "evaluate the experimental effect",
    "assess differences across conditions",
    "test the predicted relationship",
]


def stat_inline(test_type, stat, df1, df2, reported_p, equality, env):
    # APA / non_apa строка типа "t(28) = 2.20, p = .03"
    if reported_p < 1:
        p_str = format(reported_p, ".3f").lstrip("0")
    else:
        p_str = format(reported_p, ".3f")
    if env == "non_apa":
        # без скобок и с лишним пробелом
        if test_type == "F":
            return test_type + " = " + str(stat) + " with df = " + str(int(df1)) + ", " + str(int(df2)) + ", p " + equality + " " + p_str
        if test_type in ("t", "r", "chi", "Q"):
            return test_type + " = " + str(stat) + " with df = " + str(int(df1)) + ", p " + equality + " " + p_str
        return test_type + " = " + str(stat) + ", p " + equality + " " + p_str
    # apa default
    if test_type == "F":
        return "F(" + str(int(df1)) + ", " + str(int(df2)) + ") = " + str(stat) + ", p " + equality + " " + p_str
    if test_type == "r":
        return "r(" + str(int(df1)) + ") = " + str(stat) + ", p " + equality + " " + p_str
    if test_type in ("t", "chi", "Q"):
        return test_type + "(" + str(int(df1)) + ") = " + str(stat) + ", p " + equality + " " + p_str
    return test_type + " = " + str(stat) + ", p " + equality + " " + p_str


WRONG_TEXTUAL_OPPOSITE = {
    True: [
        "The result was not statistically significant.",
        "These findings indicate no statistically significant effect.",
        "We observed no significant difference.",
        "The difference did not reach statistical significance.",
    ],
    False: [
        "The result was statistically significant.",
        "These findings indicate a statistically significant effect.",
        "We observed a significant difference.",
        "The difference reached statistical significance.",
    ],
}


def build_fragment(test_type, stat, df1, df2, reported_p, equality, target_sig, env, rng):
    intro = rng.choice(INTRO_TEMPLATES)
    if "{topic_pair}" in intro:
        intro = intro.format(topic_pair=rng.choice(TOPIC_PAIRS))
    else:
        intro = intro.format(topic=rng.choice(TOPICS))
    purpose = rng.choice(PURPOSES)
    inline = stat_inline(test_type, stat, df1, df2, reported_p, equality, env)
    wrong_text = rng.choice(WRONG_TEXTUAL_OPPOSITE[target_sig])
    return intro + " A statistical analysis was performed to " + purpose + ". The test yielded " + inline + ". " + wrong_text


def authors_textual_interpretation(target_sig):
    return WRONG_TEXTUAL_OPPOSITE[target_sig][0]


def authors_direction(target_sig):
    if target_sig:
        return "not_significant"
    return "significant"


def gen_one_example(rng, idx):
    test_type, gen = GENERATORS[idx % len(GENERATORS)]
    target_sig = (idx % 2 == 0)  # чередуем для баланса
    env = rng.choice(["text", "apa", "non_apa"])

    stat, df1, df2, n_sample, true_p, equality = gen(rng, target_sig)
    reported_p = round(true_p, 3)  # автор честно округляет
    fragment = build_fragment(test_type, stat, df1, df2, reported_p, equality,
                              target_sig, env, rng)

    return {
        "example_id": 100000 + idx,
        "environment": env,
        "domain": "synthetic_clean_C",
        "n_tests": 1,
        "tests": [{
            "test_type": test_type,
            "statistic": stat,
            "statistic_original": stat,
            "df1": df1,
            "df2": df2,
            "n_sample": n_sample,
            "p_value": round(true_p, 6),
            "reported_p": reported_p,
            "p_equality": equality,
            "two_tailed": True,
            "consistent": False,
            "error_type": "wrong_conclusion_clean",
        }],
        "error_type": "wrong_conclusion_clean",
        "label_consistent": False,
        "fragment": fragment,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-total", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="data/clean_c_dataset.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_by_type = {tt: 0 for tt, _ in GENERATORS}
    n_by_target = {True: 0, False: 0}
    with out_path.open("w", encoding="utf-8") as fh:
        for i in range(args.n_total):
            ex = gen_one_example(rng, i)
            t = ex["tests"][0]
            n_by_type[t["test_type"]] += 1
            target = t["p_value"] <= 0.05
            n_by_target[target] += 1
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
