import re

from pipeline.test_verificator import normalize_test_type


NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
STAT_TOL = 0.02
KEY_FIELDS = ("test_type", "statistic_value", "df1", "df2", "reported_p")


def normalise_whitespace(s):
    # приводим к нижнему регистру и схлопываем пробелы
    return " ".join(s.lower().split())


def normalise_aggressive(s):
    # удаляем спецсимволы и схлопываем пробелы
    s = s.lower()
    s = re.sub(r"[^\w\s.\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_numbers(s):
    return NUMBER_RE.findall(s)


def raw_text_coverage(gold_tests, markdown):
    # считаем сколько gold raw_text находим в маркдауне (3 варианта матчинга)
    considered = [t for t in gold_tests if t.get("raw_text")]
    if not considered:
        return {
            "coverage": None,
            "coverage_strict": None,
            "coverage_normalised": None,
            "coverage_numerical": None,
            "matched": 0, "total": 0, "missed": [],
        }

    md_strict = normalise_whitespace(markdown)
    md_norm = normalise_aggressive(markdown)
    md_numbers = set(extract_numbers(markdown))

    matched_strict = 0
    matched_norm = 0
    matched_num = 0
    missed = []

    for t in considered:
        rt = t["raw_text"]
        rt_strict = normalise_whitespace(rt)
        rt_norm = normalise_aggressive(rt)
        rt_numbers = extract_numbers(rt)

        hit_strict = bool(rt_strict) and rt_strict in md_strict
        hit_norm = bool(rt_norm) and rt_norm in md_norm
        hit_num = bool(rt_numbers) and all(n in md_numbers for n in rt_numbers)

        if hit_strict:
            matched_strict += 1
        if hit_norm:
            matched_norm += 1
        if hit_num:
            matched_num += 1
        if not hit_num:
            missed.append(rt)

    total = len(considered)
    return {
        "coverage": round(matched_num / total, 4),
        "coverage_strict": round(matched_strict / total, 4),
        "coverage_normalised": round(matched_norm / total, 4),
        "coverage_numerical": round(matched_num / total, 4),
        "matched": matched_num,
        "matched_strict": matched_strict,
        "matched_normalised": matched_norm,
        "matched_numerical": matched_num,
        "total": total,
        "missed": missed,
    }


def stat_close(a, b, rel_tol=STAT_TOL):
    # близость двух статистик с относительной погрешностью
    if a is None or b is None:
        return False
    denom = max(abs(b), 1e-9)
    return abs(a - b) / denom < rel_tol


def match_pair(pred, gold):
    # пара совпадает если тип одинаковый и статистика близка
    if normalize_test_type(pred.get("test_type")) != normalize_test_type(gold.get("test_type")):
        return False
    return stat_close(pred.get("statistic_value"), gold.get("statistic_value"))


def field_equal(pred_val, gold_val, is_stat_field):
    # сравнение полей с учётом типа
    if pred_val is None and gold_val is None:
        return True
    if pred_val is None or gold_val is None:
        return False
    if is_stat_field:
        return stat_close(pred_val, gold_val)
    return normalize_test_type(str(pred_val)) == normalize_test_type(str(gold_val))


def classify_fp(pred, gold_tests, matched_gold_idx, stat_tol=STAT_TOL):
    # классифицируем тип ошибки галлюцинации
    pred_type = normalize_test_type(pred.get("test_type"))
    pred_stat = pred.get("statistic_value")

    # 1. дубликат - совпадает с уже сматченным gold
    for gi in matched_gold_idx:
        g = gold_tests[gi]
        if match_pair(pred, g):
            return {"category": "duplicate", "pred": pred, "closest_gold": g}

    same_type_gold = []
    for gi, g in enumerate(gold_tests):
        if normalize_test_type(g.get("test_type")) == pred_type:
            same_type_gold.append((gi, g))

    same_stat_gold = []
    for gi, g in enumerate(gold_tests):
        if stat_close(pred_stat, g.get("statistic_value")):
            same_stat_gold.append((gi, g))

    # 2. wrong_test_type - значение близко но тип не тот
    if same_stat_gold:
        return {
            "category": "wrong_test_type",
            "pred": pred,
            "closest_gold": same_stat_gold[0][1],
        }

    # 3. off_stat_value - тип есть, значение не то
    if same_type_gold:
        if pred_stat is not None:
            def dist(g):
                gs = g[1].get("statistic_value")
                if gs is None:
                    return float("inf")
                return abs(pred_stat - gs) / max(abs(gs), 1e-9)
            closest = min(same_type_gold, key=dist)[1]
        else:
            closest = same_type_gold[0][1]
        return {
            "category": "off_stat_value",
            "pred": pred,
            "closest_gold": closest,
        }

    # 4. полностью выдуманный тест
    return {
        "category": "complete_fabrication",
        "pred": pred,
        "closest_gold": None,
    }


def eval_stage3(gold_tests, predicted_tests):
    # матчим predicted с gold и считаем метрики
    gold_taken = [False] * len(gold_tests)
    pairs = []  # (predicted_idx, gold_idx)

    for pi, p in enumerate(predicted_tests):
        for gi, g in enumerate(gold_tests):
            if gold_taken[gi]:
                continue
            if match_pair(p, g):
                gold_taken[gi] = True
                pairs.append((pi, gi))
                break

    tp = len(pairs)
    fp = len(predicted_tests) - tp
    fn = len(gold_tests) - tp

    # field accuracy и complete extraction rate
    n_tp = len(pairs)
    if n_tp:
        field_scores = []
        full_correct = 0
        for pi, gi in pairs:
            p = predicted_tests[pi]
            g = gold_tests[gi]
            hits = 0
            total = len(KEY_FIELDS)
            for f in KEY_FIELDS:
                is_stat = f != "test_type"
                if field_equal(p.get(f), g.get(f), is_stat_field=is_stat):
                    hits += 1
            field_scores.append(hits / total)
            if hits == total:
                full_correct += 1
        field_accuracy = sum(field_scores) / len(field_scores)
        complete_extraction_rate = full_correct / n_tp
    else:
        field_accuracy = None
        complete_extraction_rate = None

    if (tp + fp):
        precision = tp / (tp + fp)
    else:
        precision = None
    if (tp + fn):
        recall = tp / (tp + fn)
    else:
        recall = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    # hallucination rate = FP / (TP + FP)
    if (tp + fp):
        hallucination_rate = fp / (tp + fp)
    else:
        hallucination_rate = None

    # классифицируем все FP
    matched_pred = {pi for pi, _ in pairs}
    matched_gold = {gi for _, gi in pairs}
    fp_classifications = []
    for pi, p in enumerate(predicted_tests):
        if pi in matched_pred:
            continue
        cls = classify_fp(p, gold_tests, matched_gold)
        cls["pred_idx"] = pi
        fp_classifications.append(cls)

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "field_accuracy": field_accuracy,
        "complete_extraction_rate": complete_extraction_rate,
        "hallucination_rate": hallucination_rate,
        "pairs": pairs,
        "fp_classifications": fp_classifications,
    }


def eval_stage4(gold_tests, predicted_merged, stage3_pairs):
    # метрики для интерпретаций
    if not stage3_pairs:
        return {
            "primary_direction_accuracy": None,
            "found_interpretation_rate": None,
            "matched_pairs": 0,
        }

    allowed = {"significant", "not_significant", "marginal", "unclear"}
    direction_hits = 0
    direction_total = 0
    found = 0
    for pi, gi in stage3_pairs:
        pred = predicted_merged[pi]
        gold = gold_tests[gi]

        gold_dir = gold.get("interpretation_direction")
        pred_dir = pred.get("primary_direction")
        interps = pred.get("interpretations", [])

        if interps:
            found += 1

        # accuracy считаем только на парах где gold direction валидный (не unclear)
        if gold_dir in allowed and gold_dir != "unclear":
            direction_total += 1
            if pred_dir == gold_dir:
                direction_hits += 1

    if direction_total:
        primary_acc = direction_hits / direction_total
    else:
        primary_acc = None

    return {
        "primary_direction_accuracy": primary_acc,
        "found_interpretation_rate": found / len(stage3_pairs),
        "matched_pairs": len(stage3_pairs),
        "direction_total": direction_total,
        "direction_hits": direction_hits,
    }


def eval_stage5_example(gold_tests, verified_tests, stage3_pairs, example_has_errors):
    # метрики верификации (sanity / consistency)
    tpv = fpv = fnv = 0
    not_checkable = 0
    tpv_c = fpv_c = fnv_c = 0
    not_checkable_c = 0

    for pi, gi in stage3_pairs:
        gold = gold_tests[gi]
        verified = verified_tests[pi]
        p_consistency = verified.get("p_consistency")
        interp_consistency = verified.get("interpretation_consistency")
        conflict_flag = bool(verified.get("has_text_vs_numbers_conflict", False))

        gold_is_error = (gold.get("consistent") is False)

        # вариант только по p
        if p_consistency == "not_checkable":
            not_checkable += 1
        else:
            pred_is_error = (p_consistency == "inconsistent")
            if gold_is_error and pred_is_error:
                tpv += 1
            elif gold_is_error and not pred_is_error:
                fnv += 1
            elif (not gold_is_error) and pred_is_error:
                fpv += 1

        # combined вариант: учитываем все три сигнала
        any_signal_checkable = (
            p_consistency in ("consistent", "inconsistent")
            or interp_consistency in ("consistent", "inconsistent")
            or conflict_flag
        )
        if not any_signal_checkable:
            not_checkable_c += 1
        else:
            pred_is_error_c = (
                p_consistency == "inconsistent"
                or interp_consistency == "inconsistent"
                or conflict_flag
            )
            if gold_is_error and pred_is_error_c:
                tpv_c += 1
            elif gold_is_error and not pred_is_error_c:
                fnv_c += 1
            elif (not gold_is_error) and pred_is_error_c:
                fpv_c += 1

    return {
        "tpv": tpv, "fpv": fpv, "fnv": fnv,
        "not_checkable": not_checkable,
        "tpv_combined": tpv_c, "fpv_combined": fpv_c, "fnv_combined": fnv_c,
        "not_checkable_combined": not_checkable_c,
    }


def aggregate_stage5(per_example):
    # суммируем по всем примерам
    keys = ["tpv", "fpv", "fnv", "not_checkable",
            "tpv_combined", "fpv_combined", "fnv_combined", "not_checkable_combined"]
    total = {k: 0 for k in keys}
    for rec in per_example:
        for k in keys:
            total[k] += rec.get(k, 0)

    tpv, fpv, fnv = total["tpv"], total["fpv"], total["fnv"]
    if (tpv + fnv):
        idr = tpv / (tpv + fnv)
    else:
        idr = None
    if (fpv + tpv):
        far = fpv / (fpv + tpv)
    else:
        far = None

    tpv_c, fpv_c, fnv_c = total["tpv_combined"], total["fpv_combined"], total["fnv_combined"]
    if (tpv_c + fnv_c):
        combined_idr = tpv_c / (tpv_c + fnv_c)
    else:
        combined_idr = None
    if (fpv_c + tpv_c):
        combined_far = fpv_c / (fpv_c + tpv_c)
    else:
        combined_far = None

    out = dict(total)
    out["inconsistency_detection_rate"] = idr
    out["false_alarm_rate"] = far
    out["combined_idr"] = combined_idr
    out["combined_far"] = combined_far
    return out


def aggregate_stage3(per_example):
    # суммируем метрики stage3
    tp = fp = fn = 0
    fa_sum = 0.0
    fa_n = 0
    ce_sum = 0.0
    ce_n = 0
    for rec in per_example:
        tp += rec["tp"]
        fp += rec["fp"]
        fn += rec["fn"]
        if rec.get("field_accuracy") is not None:
            fa_sum += rec["field_accuracy"] * rec["tp"]
            fa_n += rec["tp"]
        if rec.get("complete_extraction_rate") is not None:
            ce_sum += rec["complete_extraction_rate"] * rec["tp"]
            ce_n += rec["tp"]

    if (tp + fp):
        precision = tp / (tp + fp)
    else:
        precision = None
    if (tp + fn):
        recall = tp / (tp + fn)
    else:
        recall = None
    if precision and recall and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    if (tp + fp):
        hallucination = fp / (tp + fp)
    else:
        hallucination = None

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "field_accuracy": (fa_sum / fa_n) if fa_n else None,
        "complete_extraction_rate": (ce_sum / ce_n) if ce_n else None,
        "hallucination_rate": hallucination,
    }


def aggregate_stage4(per_example):
    # суммируем метрики stage4
    dh = dt = fi = mp = 0
    for rec in per_example:
        dh += rec.get("direction_hits", 0)
        dt += rec.get("direction_total", 0)
        mp += rec.get("matched_pairs", 0)
        if rec.get("found_interpretation_rate") is not None:
            fi += rec["found_interpretation_rate"] * rec["matched_pairs"]
    return {
        "primary_direction_accuracy": (dh / dt) if dt else None,
        "found_interpretation_rate": (fi / mp) if mp else None,
        "direction_hits": dh, "direction_total": dt,
        "matched_pairs": mp,
    }
