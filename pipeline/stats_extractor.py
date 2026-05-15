import re
import json
import os
from pipeline.test_verificator import normalize_test_type


SYSTEM_PROMPT = (
    "You are a precise information-extraction system for scientific papers. "
    "Extract statistical hypothesis tests from text. "
    "Return ONLY valid JSON, no markdown fences."
)

USER_PROMPT_TEMPLATE = """\
Extract every statistical hypothesis test reported in the text below.
This includes tests reported inline in text AND in tables (markdown format).
Consider these test types: t, F, chi (chi-square), z, r (correlation), Q.
Ignore descriptive statistics (means, SDs, CIs) unless they are part of a test result.

For each test, return an object with exactly these fields:
  test_type                — one of: "t", "F", "chi", "z", "r", "Q"
  statistic_value          — float, the observed test statistic
  df1                      — float or null (first degrees of freedom)
  df2                      — float or null (second df, only for F-tests)
  reported_p               — float or null (the p-value as reported)
  p_equality               — one of "=", "<", ">" or null
  two_tailed               — boolean (true if two-tailed or not specified)
  raw_text                 — the exact substring or table row containing the test
  textual_interpretation   — the authors' conclusion about this test
  interpretation_direction — one of: "significant", "not_significant", "marginal", "unclear"
  consistent               — boolean: is the textual interpretation consistent with the p-value?
  notes                    — string: any relevant context (table number, sample size, etc.)

Return a JSON array. If no tests are found, return [].

TEXT:
\"\"\"
{text}
\"\"\"
"""

# конфиги моделей
MODELS = {
    "deepseek": {
        "url": "https://api.deepseek.com/v1/chat/completions",
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "model": "gemini-2.5-flash",
        "env_key": "GEMINI_API_KEY",
    },
    "openrouter-deepseek": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "deepseek/deepseek-chat",
        "env_key": "OPENROUTER_API_KEY",
    },
    "openrouter-gemini": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "google/gemini-2.5-flash",
        "env_key": "OPENROUTER_API_KEY",
    },
}


def call_api(text, model_name, api_key=None, timeout=120):
    # вызов LLM
    from pipeline._llm_client import post_openai_chat

    cfg = MODELS[model_name]
    key = api_key or os.environ.get(cfg["env_key"])
    if not key:
        raise RuntimeError(cfg["env_key"] + " не задан")

    payload = {
        "model": cfg["model"],
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
        ],
    }
    return post_openai_chat(
        url=cfg["url"], api_key=key, payload=payload,
        timeout=timeout, model_hint=model_name,
    )


def _to_float(v):
    # конвертим в float или None
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_response(raw):
    # парсим JSON-ответ от LLM
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
        # иногда LLM оборачивает массив в объект
        found = []
        for k in ("tests", "results"):
            if isinstance(data.get(k), list):
                found = data[k]
                break
        data = found

    if not isinstance(data, list):
        return []

    out = []
    for r in data:
        out.append({
            "test_type": str(r.get("test_type", "")).strip(),
            "statistic_value": _to_float(r.get("statistic_value")),
            "df1": _to_float(r.get("df1")),
            "df2": _to_float(r.get("df2")),
            "reported_p": _to_float(r.get("reported_p")),
            "p_equality": r.get("p_equality"),
            "two_tailed": r.get("two_tailed", True),
            "raw_text": str(r.get("raw_text", "")),
            "textual_interpretation": str(r.get("textual_interpretation", "")),
            "interpretation_direction": str(r.get("interpretation_direction", "unclear")),
            "consistent": r.get("consistent"),
            "notes": str(r.get("notes", "")),
        })
    return out


# допустимые диапазоны значений статистики
PLAUSIBILITY_RANGES = {
    "t":   (0.0, 50.0),
    "z":   (0.0, 20.0),
    "F":   (0.0, 500.0),
    "chi": (0.0, 500.0),
    "Q":   (0.0, 500.0),
    "r":   (-1.0, 1.0),
}


def is_plausible(test):
    # проверяем что значение в разумных пределах
    tt = normalize_test_type(test.get("test_type"))
    if tt is None or tt not in PLAUSIBILITY_RANGES:
        return False

    stat = test.get("statistic_value")
    if stat is None:
        return False
    lo, hi = PLAUSIBILITY_RANGES[tt]
    if tt == "r":
        val = stat
    else:
        val = abs(stat)
    if not (lo <= val <= hi):
        return False

    p = test.get("reported_p")
    if p is not None and not (0.0 <= p <= 1.0):
        return False

    for df_field in ("df1", "df2"):
        df = test.get(df_field)
        if df is not None and df <= 0:
            return False

    return True


def dedupe_predictions(tests):
    #   pass 1 (строгий ключ): test_type + round(stat, 2) + df1 + df2
    #   pass 2 (мягкий ключ):  test_type + round(stat, 1)
    #     срабатывает когда один из дублей имеет df1/df2 = None, а другой нет,
    #     или статистика округлена по-разному в разных местах текста
    seen_strict = set()
    seen_loose = set()
    out = []
    for t in tests:
        stat = t.get("statistic_value")
        stat_key2 = round(stat, 2) if stat is not None else None
        stat_key1 = round(stat, 1) if stat is not None else None
        tt = normalize_test_type(t.get("test_type"))

        strict_key = (tt, stat_key2, t.get("df1"), t.get("df2"))
        if strict_key in seen_strict:
            continue
        loose_key = (tt, stat_key1)
        if stat is not None and loose_key in seen_loose:
            continue

        seen_strict.add(strict_key)
        if stat is not None:
            seen_loose.add(loose_key)
        out.append(t)
    return out


def clean_predictions(tests):
    # фильтруем + дедуп
    plausible = [t for t in tests if is_plausible(t)]
    return dedupe_predictions(plausible)


def extract_tests(text, model_name, api_key=None, clean=True):
    # главная функция извлечения тестов
    if not text.strip():
        return []
    raw = call_api(text, model_name, api_key)
    parsed = parse_response(raw)
    if clean:
        return clean_predictions(parsed)
    return parsed
