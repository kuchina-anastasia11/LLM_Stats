#  chain-of-thought для stage 4
#
# идея: ловить случаи когда автор пишет правильное число но интерпретирует его наоборот.
# простая p-проверка тут не помогает - нужно сравнить что говорят числа vs текст.
#
# схема:
#   - computed_p считаем через scipy и кладём в промпт
#   - LLM проходит 4 шага: inferred (по числам), interpretations, authors_direction, conflict
#   - field direction = authors_direction (для совместимости с eval_stage4)
#   - inferred_direction и has_text_vs_numbers_conflict идут в combined_idr
import json
import os
import re

from pipeline._llm_client import post_openai_chat
from pipeline.interpritation_extractor import (
    MODELS, build_test_context, format_sectioned_text, SECTION_PRIORITY,
)
from pipeline.test_verificator import compute_p


ALPHA = 0.05
MARGINAL_HI = 0.10


SYSTEM_PROMPT = (
    "You are a precise information-extraction system for scientific papers. "
    "For each statistical test you perform a 4-step reasoning chain comparing "
    "what the NUMBERS say vs what the AUTHORS' TEXT says. "
    "Return ONLY valid JSON, no markdown fences, no commentary."
)


USER_PROMPT_TEMPLATE = '''\
You receive:
  1) A list of statistical tests already extracted from a paper. Each test
     comes with its test_type, statistic_value, degrees of freedom,
     reported_p (as written by the authors) and computed_p (recomputed
     deterministically from the statistic via scipy.stats; may be null
     if df is missing).
  2) Sectioned text blocks from the paper.

For EACH test perform this 4-step reasoning chain:

  STEP 1 — INFERRED DIRECTION (from numbers).
    Use computed_p if present, else reported_p, at alpha = {alpha}.
      - p <= {alpha}                       → "significant"
      - {alpha} < p <= {marg_hi}            → "marginal"
      - p >  {marg_hi}                      → "not_significant"
      - p is null and statistic is unusable → "unclear"

  STEP 2 — FIND TEXTUAL INTERPRETATIONS.
    Scan ALL section blocks for sentences where the authors interpret THIS
    test. A test may be discussed in multiple sections — return one object
    per section where it is meaningfully discussed (not just repeated as a
    number).

  STEP 3 — AUTHORS_DIRECTION (from text only, ignoring numbers).
    For each interpretation extract what the AUTHOR EXPLICITLY CLAIMS:
      "significant" / "not_significant" / "marginal" / "unclear".
    Look at literal phrases: "statistically significant", "no difference",
    "trend toward", "approaching significance", etc. This is independent of
    the numbers — quote what the author wrote.

  STEP 4 — CONFLICT FLAG.
    For each interpretation set
      conflict_with_authors_text = (inferred_direction != authors_direction
                                    and both are not "unclear").

Return a JSON array, one object per input test:
  test_id            — integer, echoed from input
  inferred_direction — STEP 1 result (string)
  interpretations    — list (possibly empty), each object:
      section            — section label from input
      sentence           — verbatim sentence(s) carrying the interpretation
      keywords           — list of exact words/phrases for significance/hedging
      authors_direction  — STEP 3, "significant"/"not_significant"/"marginal"/"unclear"
      effect_strength    — "strong"/"moderate"/"weak"/"none"/"unclear"
      hedging            — boolean (hedging language present)
      claim              — one-sentence paraphrase of authors' point
      conflict_with_authors_text — STEP 4 boolean

If a test has NO interpretation in the text, return
  {{"test_id": <id>, "inferred_direction": "<step1>", "interpretations": []}}.

TESTS:
{tests}

SECTIONED TEXT:
{sectioned_text}
'''


def infer_direction_from_p(p):
    # детерминированный фоллбэк для step 1
    if p is None:
        return "unclear"
    if p <= ALPHA:
        return "significant"
    if p <= MARGINAL_HI:
        return "marginal"
    return "not_significant"


def slim_test(t, test_id):
    # как slim_test в interpritation_extractor, но добавляем computed_p
    cp = compute_p(
        test_type=t.get("test_type"),
        statistic_value=t.get("statistic_value"),
        df1=t.get("df1"),
        df2=t.get("df2"),
        two_tailed=t.get("two_tailed", True),
    )
    return {
        "test_id": test_id,
        "raw_text": t.get("raw_text", ""),
        "test_type": t.get("test_type", ""),
        "statistic_value": t.get("statistic_value"),
        "df1": t.get("df1"),
        "df2": t.get("df2"),
        "reported_p": t.get("reported_p"),
        "computed_p": cp,
    }


def call_api(sectioned_text, tests, model_name, api_key=None, timeout=240):
    cfg = MODELS[model_name]
    key = api_key or os.environ.get(cfg["env_key"])
    if not key:
        raise RuntimeError(cfg["env_key"] + " не задан")

    slim = []
    for i, t in enumerate(tests):
        slim.append(slim_test(t, i))

    payload = {
        "model": cfg["model"],
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                alpha=ALPHA,
                marg_hi=MARGINAL_HI,
                tests=json.dumps(slim, ensure_ascii=False, indent=2),
                sectioned_text=sectioned_text,
            )},
        ],
    }
    return post_openai_chat(
        url=cfg["url"], api_key=key, payload=payload,
        timeout=timeout, model_hint=model_name + "-cot",
    )


DIR_ALLOWED = {"significant", "not_significant", "marginal", "unclear"}
STR_ALLOWED = {"strong", "moderate", "weak", "none", "unclear"}


def parse_response(raw, slim_tests):
    # парсим ответ CoT-промпта
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"(\[.*])", cleaned, re.DOTALL)
        if m:
            data = json.loads(m.group(1))
        else:
            data = []

    if isinstance(data, dict):
        found = []
        for k in ("results", "items"):
            if isinstance(data.get(k), list):
                found = data[k]
                break
        data = found

    if not isinstance(data, list):
        return []

    by_id = {s["test_id"]: s for s in slim_tests}

    out = []
    for r in data:
        tid = r.get("test_id")
        slim = by_id.get(tid, {})

        # step 1: проверяем inferred через scipy
        p_use = slim.get("computed_p")
        if p_use is None:
            p_use = slim.get("reported_p")
        deterministic_inferred = infer_direction_from_p(p_use)

        llm_inferred = str(r.get("inferred_direction", "")).strip()
        if llm_inferred not in DIR_ALLOWED:
            llm_inferred = "unclear"

        # если scipy дал ответ - используем его. иначе доверяем LLM
        if deterministic_inferred != "unclear":
            inferred_direction = deterministic_inferred
        else:
            inferred_direction = llm_inferred

        interps_raw = r.get("interpretations") or []
        interps = []
        for it in interps_raw:
            authors_dir = str(it.get("authors_direction", "unclear")).strip()
            if authors_dir not in DIR_ALLOWED:
                authors_dir = "unclear"
            strength = str(it.get("effect_strength", "unclear")).strip()
            if strength not in STR_ALLOWED:
                strength = "unclear"
            kws = it.get("keywords") or []
            if not isinstance(kws, list):
                kws = [str(kws)]
            cleaned_kws = []
            for k in kws:
                if str(k).strip():
                    cleaned_kws.append(str(k).strip())

            # перепроверяем conflict из обеих сторон
            llm_conflict = bool(it.get("conflict_with_authors_text", False))
            recomputed_conflict = (
                inferred_direction != authors_dir
                and inferred_direction != "unclear"
                and authors_dir != "unclear"
            )
            conflict = llm_conflict or recomputed_conflict

            interps.append({
                "section": str(it.get("section", "")).strip(),
                "sentence": str(it.get("sentence", "")).strip(),
                "keywords": cleaned_kws,
                # поле direction = authors_direction (для совместимости с eval_stage4)
                "direction": authors_dir,
                "authors_direction": authors_dir,
                "inferred_direction": inferred_direction,
                "effect_strength": strength,
                "hedging": bool(it.get("hedging", False)),
                "claim": str(it.get("claim", "")).strip(),
                "conflict_with_authors_text": conflict,
            })

        out.append({
            "test_id": tid,
            "inferred_direction": inferred_direction,
            "interpretations": interps,
        })
    return out


def extract_interpretations_cot(sections, tests, model_name, api_key=None):
    # главная функция CoT-извлечения
    if not tests:
        return []
    blocks_per_test = []
    for t in tests:
        blocks_per_test.append(build_test_context(sections, t.get("raw_text", "")))
    sectioned_text = format_sectioned_text(blocks_per_test)
    slim = []
    for i, t in enumerate(tests):
        slim.append(slim_test(t, i))
    raw = call_api(sectioned_text, tests, model_name, api_key)
    return parse_response(raw, slim)


def aggregate_cot(interps, fallback_inferred):
    # агрегат + CoT-поля
    if not interps:
        return {
            "primary_direction": "unclear",
            "primary_inferred_direction": fallback_inferred or "unclear",
            "primary_authors_direction": "unclear",
            "has_cross_section_conflict": False,
            "has_text_vs_numbers_conflict": False,
            "any_hedging": False,
            "sections": [],
        }

    by_sec = {}
    for i in interps:
        if i.get("section"):
            by_sec[i["section"]] = i

    primary_authors = None
    primary_inferred = None
    for s in SECTION_PRIORITY:
        if s in by_sec:
            primary_authors = by_sec[s].get("authors_direction", "unclear")
            primary_inferred = by_sec[s].get("inferred_direction", fallback_inferred or "unclear")
            break
    if primary_authors is None:
        primary_authors = interps[0].get("authors_direction", "unclear")
        primary_inferred = interps[0].get("inferred_direction", fallback_inferred or "unclear")

    # cross-section conflict считаем по authors_direction
    authors_dirs = set()
    for i in interps:
        d = i.get("authors_direction")
        if d and d != "unclear":
            authors_dirs.add(d)

    return {
        "primary_direction": primary_authors,
        "primary_inferred_direction": primary_inferred,
        "primary_authors_direction": primary_authors,
        "has_cross_section_conflict": len(authors_dirs) > 1,
        "has_text_vs_numbers_conflict": any(i.get("conflict_with_authors_text") for i in interps),
        "any_hedging": any(i.get("hedging") for i in interps),
        "sections": [i["section"] for i in interps],
    }


def merge_with_tests_cot(tests, interpretations):
    # склеиваем интерпретации с тестами
    by_id = {}
    for i in interpretations:
        if i.get("test_id") is not None:
            by_id[i["test_id"]] = i

    merged = []
    for idx, t in enumerate(tests):
        rec = by_id.get(idx, {"interpretations": [], "inferred_direction": None})
        agg = aggregate_cot(rec["interpretations"], rec.get("inferred_direction"))
        new_t = dict(t)
        new_t["interpretations"] = rec["interpretations"]
        new_t["inferred_direction"] = rec.get("inferred_direction")
        new_t.update(agg)
        merged.append(new_t)
    return merged
