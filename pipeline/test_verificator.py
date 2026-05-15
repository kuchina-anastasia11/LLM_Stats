import math
from scipy import stats


# словарь с алиасами для типов тестов (приводим к одному виду)
TEST_TYPE_ALIASES = {
    "t": "t",
    "f": "F",
    "chi": "chi",
    "chi2": "chi",
    "chisq": "chi",
    "chi-square": "chi",
    "chi_square": "chi",
    "z": "z",
    "r": "r",
    "q": "Q",
}


def normalize_test_type(test_type):
    # приводим название теста к одному виду
    if test_type is None:
        return None
    s = str(test_type).strip().lower()
    return TEST_TYPE_ALIASES.get(s)


def compute_p(test_type, statistic_value, df1, df2, two_tailed=True):
    # пересчитываем p-value по статистике через scipy
    if statistic_value is None:
        return None
    tt = normalize_test_type(test_type)

    try:
        if tt == "t":
            if df1 is None:
                return None
            if two_tailed:
                p = stats.t.sf(abs(statistic_value), df1) * 2
            else:
                p = stats.t.sf(abs(statistic_value), df1)

        elif tt == "F":
            if df1 is None or df2 is None:
                return None
            p = stats.f.sf(statistic_value, df1, df2)

        elif tt == "chi":
            if df1 is None:
                return None
            p = stats.chi2.sf(statistic_value, df1)

        elif tt == "z":
            if two_tailed:
                p = stats.norm.sf(abs(statistic_value)) * 2
            else:
                p = stats.norm.sf(abs(statistic_value))

        elif tt == "r":
            # r переводим в t-статистику и считаем p как для t-теста
            if df1 is None:
                return None
            denom = 1 - statistic_value ** 2
            if denom <= 0:
                return None
            t_val = statistic_value * math.sqrt(df1) / math.sqrt(denom)
            if two_tailed:
                p = stats.t.sf(abs(t_val), df1) * 2
            else:
                p = stats.t.sf(abs(t_val), df1)

        elif tt == "Q":
            if df1 is None:
                return None
            p = stats.chi2.sf(statistic_value, df1)

        else:
            return None

        return round(p, 10)

    except (ValueError, ZeroDivisionError):
        return None


def check_consistency(reported_p, computed_p, p_equality, tolerance=0.05):
    # сравниваем reported и computed p
    if reported_p is None or computed_p is None:
        return "not_checkable"

    if p_equality == "<":
        if computed_p < reported_p:
            return "consistent"
        diff = abs(computed_p - reported_p) / max(reported_p, 1e-10)
        if diff < tolerance:
            return "marginal"
        return "inconsistent"

    if p_equality == ">":
        if computed_p > reported_p:
            return "consistent"
        diff = abs(computed_p - reported_p) / max(reported_p, 1e-10)
        if diff < tolerance:
            return "marginal"
        return "inconsistent"

    # случай "=" или None
    ratio = abs(computed_p - reported_p) / max(reported_p, 1e-10)
    if ratio < tolerance:
        return "consistent"
    if ratio < tolerance * 3:
        return "marginal"
    return "inconsistent"


def check_interpretation(interpretation_direction, computed_p, alpha=0.05):
    # проверяем интерпретацию автора по computed_p
    if computed_p is None:
        return "not_checkable"

    if interpretation_direction == "significant" and computed_p <= alpha:
        return "consistent"
    if interpretation_direction == "not_significant" and computed_p > alpha:
        return "consistent"
    if interpretation_direction == "marginal" and 0.01 < computed_p <= 0.10:
        return "consistent"
    if interpretation_direction in ("significant", "not_significant", "marginal"):
        return "inconsistent"
    return "unclear"


def verify_test(record):
    # верифицируем одну запись
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

    # primary_direction из агрегата интерпретаций, если есть
    direction = record.get("primary_direction") or record.get("interpretation_direction") or "unclear"
    interp_check = check_interpretation(
        interpretation_direction=direction,
        computed_p=computed_p,
    )

    out = dict(record)
    out["computed_p"] = computed_p
    out["p_consistency"] = p_check
    out["interpretation_consistency"] = interp_check
    return out


def verify_all(tests):
    res = []
    for t in tests:
        res.append(verify_test(t))
    return res


def summary(verified_tests):
    # считаем сводку по проверке
    total = len(verified_tests)
    checkable = [t for t in verified_tests if t["p_consistency"] != "not_checkable"]
    n_checkable = len(checkable)

    p_consistent = 0
    p_marginal = 0
    p_inconsistent = 0
    for t in checkable:
        if t["p_consistency"] == "consistent":
            p_consistent += 1
        elif t["p_consistency"] == "marginal":
            p_marginal += 1
        elif t["p_consistency"] == "inconsistent":
            p_inconsistent += 1

    interp_checkable = [t for t in verified_tests if t["interpretation_consistency"] != "not_checkable"]
    interp_consistent = sum(1 for t in interp_checkable if t["interpretation_consistency"] == "consistent")
    interp_inconsistent = sum(1 for t in interp_checkable if t["interpretation_consistency"] == "inconsistent")

    if n_checkable:
        p_rate = round(p_consistent / n_checkable, 3)
    else:
        p_rate = None

    if interp_checkable:
        interp_rate = round(interp_consistent / len(interp_checkable), 3)
    else:
        interp_rate = None

    return {
        "total_tests": total,
        "p_checkable": n_checkable,
        "p_consistent": p_consistent,
        "p_marginal": p_marginal,
        "p_inconsistent": p_inconsistent,
        "p_consistency_rate": p_rate,
        "interp_checkable": len(interp_checkable),
        "interp_consistent": interp_consistent,
        "interp_inconsistent": interp_inconsistent,
        "interp_consistency_rate": interp_rate,
    }
