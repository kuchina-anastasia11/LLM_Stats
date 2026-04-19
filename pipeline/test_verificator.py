import math
from scipy import stats


# нормализация написания типа теста (gold использует "Chi2", LLM может вернуть
# "chi2" / "chi-square" / "Q", приведём к одному ключу)
TEST_TYPE_ALIASES = {
    "t": "t",
    "f": "F",
    "chi": "chi", "chi2": "chi", "chisq": "chi", "chi-square": "chi", "chi_square": "chi",
    "z": "z",
    "r": "r",
    "q": "Q",
}


def normalize_test_type(test_type):
    if test_type is None:
        return None
    return TEST_TYPE_ALIASES.get(str(test_type).strip().lower())


# Пересчёт p-value из статистики теста (scipy.stats)
def compute_p(test_type, statistic_value, df1, df2, two_tailed=True):
    if statistic_value is None:
        return None
    tt = normalize_test_type(test_type)

    try:
        if tt == "t":
            if df1 is None:
                return None
            p = stats.t.sf(abs(statistic_value), df1) * (2 if two_tailed else 1)

        elif tt == "F":
            if df1 is None or df2 is None:
                return None
            p = stats.f.sf(statistic_value, df1, df2)

        elif tt == "chi":
            if df1 is None:
                return None
            p = stats.chi2.sf(statistic_value, df1)

        elif tt == "z":
            p = stats.norm.sf(abs(statistic_value)) * (2 if two_tailed else 1)

        elif tt == "r":
            if df1 is None:
                return None
            denom = 1 - statistic_value ** 2
            if denom <= 0:
                return None
            t_val = statistic_value * math.sqrt(df1) / math.sqrt(denom)
            p = stats.t.sf(abs(t_val), df1) * (2 if two_tailed else 1)

        elif tt == "Q":
            if df1 is None:
                return None
            p = stats.chi2.sf(statistic_value, df1)

        else:
            return None

        return round(p, 10)

    except (ValueError, ZeroDivisionError):
        return None


# Сравнение reported_p и computed_p с учётом p_equality ("<", ">", "=")
# tolerance задаёт относительную погрешность для меток consistent / marginal
def check_consistency(reported_p, computed_p, p_equality, tolerance=0.05):
    if reported_p is None or computed_p is None:
        return "not_checkable"

    if p_equality == "<":
        if computed_p < reported_p:
            return "consistent"
        elif abs(computed_p - reported_p) / max(reported_p, 1e-10) < tolerance:
            return "marginal"
        else:
            return "inconsistent"

    elif p_equality == ">":
        if computed_p > reported_p:
            return "consistent"
        elif abs(computed_p - reported_p) / max(reported_p, 1e-10) < tolerance:
            return "marginal"
        else:
            return "inconsistent"

    else:  # "=" или None
        ratio = abs(computed_p - reported_p) / max(reported_p, 1e-10)
        if ratio < tolerance:
            return "consistent"
        elif ratio < tolerance * 3:
            return "marginal"
        else:
            return "inconsistent"


# Проверка согласованности словесной интерпретации с реальным computed_p
def check_interpretation(interpretation_direction, computed_p, alpha=0.05):
    if computed_p is None:
        return "not_checkable"

    if interpretation_direction == "significant" and computed_p <= alpha:
        return "consistent"
    elif interpretation_direction == "not_significant" and computed_p > alpha:
        return "consistent"
    elif interpretation_direction == "marginal" and 0.01 < computed_p <= 0.10:
        return "consistent"
    elif interpretation_direction in ("significant", "not_significant", "marginal"):
        return "inconsistent"
    else:
        return "unclear"


# верификация одной записи: добавляем computed_p + p_consistency + interpretation_consistency
def verify_test(record):
    computed_p = compute_p(
        test_type=record.get("test_type"),
        statistic_value=record.get("statistic_value"),
        df1=record.get("df1"),
        df2=record.get("df2"),
        two_tailed=record.get("two_tailed", True),
    )

    p_check = check_consistency(
        reported_p=record.get("reported_p"),
        computed_p=computed_p,
        p_equality=record.get("p_equality"),
    )

    # primary_direction приходит из interpritation_extractor.aggregate;
    direction = record.get("primary_direction") or record.get("interpretation_direction") or "unclear"
    interp_check = check_interpretation(
        interpretation_direction=direction,
        computed_p=computed_p,
    )

    return {
        **record,
        "computed_p": computed_p,
        "p_consistency": p_check,
        "interpretation_consistency": interp_check,
    }

def verify_all(tests):
    return [verify_test(t) for t in tests]

def summary(verified_tests):
    total = len(verified_tests)
    checkable = [t for t in verified_tests if t["p_consistency"] != "not_checkable"]
    n_checkable = len(checkable)

    p_consistent = sum(1 for t in checkable if t["p_consistency"] == "consistent")
    p_marginal = sum(1 for t in checkable if t["p_consistency"] == "marginal")
    p_inconsistent = sum(1 for t in checkable if t["p_consistency"] == "inconsistent")

    interp_checkable = [t for t in verified_tests if t["interpretation_consistency"] != "not_checkable"]
    interp_consistent = sum(1 for t in interp_checkable if t["interpretation_consistency"] == "consistent")
    interp_inconsistent = sum(1 for t in interp_checkable if t["interpretation_consistency"] == "inconsistent")

    return {
        "total_tests": total,
        "p_checkable": n_checkable,
        "p_consistent": p_consistent,
        "p_marginal": p_marginal,
        "p_inconsistent": p_inconsistent,
        "p_consistency_rate": round(p_consistent / n_checkable, 3) if n_checkable else None,
        "interp_checkable": len(interp_checkable),
        "interp_consistent": interp_consistent,
        "interp_inconsistent": interp_inconsistent,
        "interp_consistency_rate": round(interp_consistent / len(interp_checkable), 3) if interp_checkable else None,
    }
