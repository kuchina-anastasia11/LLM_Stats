# кросс-проверка через LLM-as-judge
# схема:
#   1. extractor (например DeepSeek) делает обычный stage 3
#   2. verifier (например Gemini) получает markdown статьи + список proposed_tests
#      и:
#        - сверяет каждый тест с текстом
#        - помечает галлюцинации (status=remove)
#        - правит ошибочные поля (status=needs_correction)
#        - добавляет пропущенные тесты
#   3. финал = verified + corrections + added
# bidirectional: запускаем DS->Gem и Gem->DS, сливаем с пометкой cross_confirmed
import json
import os
import re

from pipeline._llm_client import post_openai_chat
from pipeline.stats_extractor import MODELS, clean_predictions


SYSTEM_PROMPT = (
    "You are a careful verification system for statistical test extraction. "
    "You receive a scientific paper text and a list of tests that another "
    "extractor proposed. Cross-check each proposed test against the actual "
    "text, mark hallucinations, fix wrong fields, and add tests the extractor "
    "missed. Return ONLY valid JSON, no markdown fences, no commentary."
)


USER_PROMPT_TEMPLATE = '''\
You receive:
  1) PROPOSED_TESTS — a JSON list of statistical tests that another extraction
     system claims to have found in the paper text below. Each test has the
     fields: test_type, statistic_value, df1, df2, reported_p, p_equality,
     two_tailed, raw_text.
  2) PAPER_TEXT — the full markdown of the paper (or its sectioned summary).

Your job: cross-check every proposed test against PAPER_TEXT and report
errors/omissions. Be strict: if you cannot find the exact statistic value
in the text (or in a markdown table row), the test is hallucinated.

For each proposed test, determine its status:
  - "correct"          — test_type, statistic_value, df, reported_p all match
                         what is reported in the text
  - "needs_correction" — the test exists in text, but at least one field is wrong;
                         provide the corrected values in `corrections`
  - "remove"           — the test is NOT in the text (hallucination), or values
                         are so different that this is a different test;
                         provide a short `reason`

Additionally, SCAN the PAPER_TEXT for statistical tests that the proposed list
MISSED. Report each such test in `missed_tests` with FULL schema (same fields
as PROPOSED_TESTS).

Test types to consider: t, F, chi (chi-square), z, r (correlation), Q.
Ignore descriptive statistics (means, SDs, CIs) unless they are part of a
test result.

Return ONLY valid JSON with this exact schema:

{{
  "verified_tests": [
    {{
      "original_id": <integer index from PROPOSED_TESTS>,
      "status": "correct" | "needs_correction",
      "corrections": {{}}        // empty if status=correct; else field→new value, e.g. {{"df1": 28, "reported_p": 0.034}}
    }}
  ],
  "removed_tests": [
    {{
      "original_id": <integer>,
      "reason": "<short explanation of why this is hallucination>"
    }}
  ],
  "missed_tests": [
    {{
      "test_type": "...", "statistic_value": ..., "df1": ..., "df2": ...,
      "reported_p": ..., "p_equality": "...", "two_tailed": ...,
      "raw_text": "<exact substring or table row>",
      "textual_interpretation": "...",
      "interpretation_direction": "significant"|"not_significant"|"marginal"|"unclear",
      "consistent": true|false|null,
      "notes": "..."
    }}
  ]
}}

PROPOSED_TESTS:
{tests_json}

PAPER_TEXT:
"""
{markdown}
"""
'''


def build_payload(markdown, proposed_tests, model_cfg):
    # формируем slim-список для verifier'а
    slim = []
    for i, t in enumerate(proposed_tests):
        slim.append({
            "original_id": i,
            "test_type": t.get("test_type"),
            "statistic_value": t.get("statistic_value"),
            "df1": t.get("df1"),
            "df2": t.get("df2"),
            "reported_p": t.get("reported_p"),
            "p_equality": t.get("p_equality"),
            "two_tailed": t.get("two_tailed"),
            # обрезаем длинный raw_text
            "raw_text": t.get("raw_text", "")[:300],
        })
    return {
        "model": model_cfg["model"],
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                tests_json=json.dumps(slim, ensure_ascii=False, indent=2),
                markdown=markdown,
            )},
        ],
    }


def call_verifier(markdown, proposed_tests, verifier_model, api_key, timeout=300):
    cfg = MODELS[verifier_model]
    key = api_key or os.environ.get(cfg["env_key"])
    if not key:
        raise RuntimeError(cfg["env_key"] + " не задан")

    payload = build_payload(markdown, proposed_tests, cfg)
    return post_openai_chat(
        url=cfg["url"], api_key=key, payload=payload,
        timeout=timeout, model_hint=verifier_model + "-verifier",
    )


def parse_verifier_response(raw):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if m:
            data = json.loads(m.group(1))
        else:
            data = {}

    if not isinstance(data, dict):
        return {"verified_tests": [], "removed_tests": [], "missed_tests": []}

    return {
        "verified_tests": data.get("verified_tests") or [],
        "removed_tests": data.get("removed_tests") or [],
        "missed_tests": data.get("missed_tests") or [],
    }


def apply_corrections(test, corrections):
    # применяем corrections от verifier'а
    if not corrections or not isinstance(corrections, dict):
        return dict(test)
    fixed = dict(test)
    allowed = {"test_type", "statistic_value", "df1", "df2", "reported_p",
               "p_equality", "two_tailed", "raw_text", "textual_interpretation",
               "interpretation_direction", "consistent", "notes"}
    for k, v in corrections.items():
        if k in allowed:
            fixed[k] = v
    return fixed


def normalize_added_test(t):
    # нормализуем added-тест в формат stats_extractor
    def to_float(v):
        if v in (None, ""):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return {
        "test_type": str(t.get("test_type", "")).strip(),
        "statistic_value": to_float(t.get("statistic_value")),
        "df1": to_float(t.get("df1")),
        "df2": to_float(t.get("df2")),
        "reported_p": to_float(t.get("reported_p")),
        "p_equality": t.get("p_equality"),
        "two_tailed": t.get("two_tailed", True),
        "raw_text": str(t.get("raw_text", "")),
        "textual_interpretation": str(t.get("textual_interpretation", "")),
        "interpretation_direction": str(t.get("interpretation_direction", "unclear")),
        "consistent": t.get("consistent"),
        "notes": str(t.get("notes", "")),
    }


def cross_verify(markdown, proposed_tests, verifier_model, api_key=None, clean=True):
    # LLM-as-judge кросс-проверка
    if not proposed_tests and not markdown.strip():
        return {
            "final_tests": [],
            "verified_records": [],
            "removed_records": [],
            "added_records": [],
            "diagnostics": {
                "n_proposed": 0, "n_verified": 0, "n_corrected": 0,
                "n_removed": 0, "n_added": 0, "n_final": 0,
            },
        }

    raw = call_verifier(markdown, proposed_tests, verifier_model, api_key)
    parsed = parse_verifier_response(raw)

    # обрабатываем verified - применяем corrections
    verified_records = parsed["verified_tests"]
    verified_ids = set()
    final = []
    n_corrected = 0
    for vr in verified_records:
        oid = vr.get("original_id")
        if not isinstance(oid, int) or not (0 <= oid < len(proposed_tests)):
            continue
        verified_ids.add(oid)
        if vr.get("status") == "needs_correction" and vr.get("corrections"):
            final.append(apply_corrections(proposed_tests[oid], vr["corrections"]))
            n_corrected += 1
        else:
            final.append(dict(proposed_tests[oid]))

    # обрабатываем removed
    removed_records = []
    removed_ids = set()
    for rr in parsed["removed_tests"]:
        oid = rr.get("original_id")
        if not isinstance(oid, int) or not (0 <= oid < len(proposed_tests)):
            continue
        if oid in verified_ids:
            # если verifier одновременно verified и removed - доверяем removed
            continue
        removed_ids.add(oid)
        removed_records.append({
            "original_id": oid,
            "reason": rr.get("reason", ""),
            "original_test": proposed_tests[oid],
        })

    # тесты не упомянутые ни в verified ни в removed - считаем что verifier их пропустил, выкидываем
    unmentioned = []
    for i in range(len(proposed_tests)):
        if i not in verified_ids and i not in removed_ids:
            unmentioned.append(i)

    # added - нормализуем и добавляем в финал
    added_records = []
    for t in parsed["missed_tests"]:
        added_records.append(normalize_added_test(t))
    final.extend(added_records)

    if clean:
        final = clean_predictions(final)

    return {
        "final_tests": final,
        "verified_records": verified_records,
        "removed_records": removed_records,
        "added_records": added_records,
        "unmentioned_ids": unmentioned,
        "diagnostics": {
            "n_proposed":  len(proposed_tests),
            "n_verified":  len(verified_ids),
            "n_corrected": n_corrected,
            "n_removed":   len(removed_records),
            "n_unmentioned": len(unmentioned),
            "n_added":     len(added_records),
            "n_final":     len(final),
        },
    }


def cross_verify_bidirectional(markdown, proposed_a, proposed_b,
                               model_a, model_b, api_key_a, api_key_b, clean=True):
    # bidirectional: A->B и B->A, объединяем по ключу теста
    out_a = cross_verify(markdown, proposed_a, verifier_model=model_b,
                         api_key=api_key_b, clean=False)
    out_b = cross_verify(markdown, proposed_b, verifier_model=model_a,
                         api_key=api_key_a, clean=False)

    from pipeline.test_verificator import normalize_test_type

    def make_key(t):
        s = t.get("statistic_value")
        if s is not None:
            sk = round(s, 2)
        else:
            sk = None
        return (
            normalize_test_type(t.get("test_type")),
            sk,
            t.get("df1"),
            t.get("df2"),
        )

    a_by_key = {make_key(t): t for t in out_a["final_tests"]}
    b_by_key = {make_key(t): t for t in out_b["final_tests"]}

    final = []
    cross_confirmed_keys = set(a_by_key) & set(b_by_key)
    for k in cross_confirmed_keys:
        merged = dict(a_by_key[k])
        merged["cross_confirmed"] = True
        final.append(merged)
    for k, t in a_by_key.items():
        if k not in cross_confirmed_keys:
            new_t = dict(t)
            new_t["cross_confirmed"] = False
            new_t["source_chain"] = model_a + "→" + model_b
            final.append(new_t)
    for k, t in b_by_key.items():
        if k not in cross_confirmed_keys:
            new_t = dict(t)
            new_t["cross_confirmed"] = False
            new_t["source_chain"] = model_b + "→" + model_a
            final.append(new_t)

    if clean:
        final = clean_predictions(final)

    return {
        "final_tests": final,
        "out_a_to_b": out_a,
        "out_b_to_a": out_b,
        "diagnostics": {
            "n_a_final": len(out_a["final_tests"]),
            "n_b_final": len(out_b["final_tests"]),
            "n_cross_confirmed": len(cross_confirmed_keys),
            "n_total_final": len(final),
        },
    }
