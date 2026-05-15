import os
import re
from pipeline._llm_client import post_openai_chat
from pipeline.stats_extractor import (
    MODELS, parse_response, clean_predictions,
    call_api as call_api_baseline,
    extract_tests as extract_tests_baseline,
)


# регулярки для выделения markdown-таблиц
TABLE_BLOCK_RE = re.compile(
    r"""(?:^|\n)                                    # начало строки
    (?P<caption>(?:Table|Tab\.?|Таблица)\s+\d+[^\n]*\n)?  # опц. подпись
    (?P<header>\|[^\n]*\|\s*\n)                     # строка-заголовок
    (?P<sep>\|\s*[:\-]+[\|\s:\-]*\|\s*\n)           # разделитель |---|
    (?P<body>(?:\|[^\n]*\|\s*\n)+)                  # тело
    """,
    re.VERBOSE | re.IGNORECASE,
)

TABLE_REF_RE = re.compile(
    r"\b(?:Table|Tab\.?|Таблица)\s+(\d+)\b",
    re.IGNORECASE,
)


# Промпт для table sweep — короткий, заточен под перечисление строк.

SYSTEM_PROMPT_TABLE = (
    "You are a precise table-extraction system for scientific papers. "
    "You receive ONE table from a paper and must enumerate every "
    "statistical hypothesis test it contains. Return ONLY valid JSON, "
    "no markdown fences, no commentary."
)


USER_PROMPT_TABLE = """\
Below is ONE table from a scientific paper. It may contain anywhere from 1
to 30 statistical hypothesis tests.

YOUR JOB: enumerate every test in this table. Each row whose cells include
a F-, t-, chi-square-, z-, r- or Q-statistic AND a p-value (or a clear
significance marker) is ONE test.

CRITICAL RULES:
1. Each row with a numerical test statistic is a SEPARATE test object.
   If the table has 10 rows of F-statistics, return 10 objects.
2. Rows like "Within Groups", "Error", "Residual", "Total", "Total Variance"
   are NOT tests. They exist only to provide df2 (their "df" column) for
   the F-rows in the same block. Skip them.
3. Header rows, sub-headers, blank separator rows — skip.
4. df1 = the test row's own "df" column. df2 = nearest "Error"/"Within
   Groups"/"Residual" row in the same block.
5. Do NOT duplicate a row. Each row appears in the output AT MOST ONCE.
6. Allowed test_type values: "t", "F", "chi" (for chi-square), "z",
   "r" (for correlation), "Q".

For each test return an object with EXACTLY these fields:
  test_type, statistic_value, df1, df2, reported_p, p_equality,
  two_tailed, raw_text, textual_interpretation, interpretation_direction,
  consistent, notes

  raw_text                 = the full table row text, pipes-separated.
  textual_interpretation   = "" (leave empty — text interpretation is
                              attached at a later stage).
  interpretation_direction = "unclear" (same reason).
  consistent               = null.
  notes                    = optional context (block name, source table N).

EXAMPLE INPUT:
Table 3. ANOVA results for stress reduction.
| Source        | SS    | df | MS    | F     | p     |
| ---           | ---   | ---| ---   | ---   | ---   |
| Time          | 18.4  | 2  | 9.20  | 6.42  | 0.003 |
| Group         | 12.1  | 1  | 12.10 | 8.46  | 0.005 |
| Time x Group  |  9.3  | 2  | 4.65  | 3.25  | 0.045 |
| Within Groups | 86.1  | 60 | 1.44  |       |       |
| Total         |126.0  | 65 |       |       |       |

EXAMPLE OUTPUT:
[
  {{"test_type": "F", "statistic_value": 6.42, "df1": 2, "df2": 60,
    "reported_p": 0.003, "p_equality": "=", "two_tailed": false,
    "raw_text": "Time | 18.4 | 2 | 9.20 | 6.42 | 0.003",
    "textual_interpretation": "", "interpretation_direction": "unclear",
    "consistent": null, "notes": "Table 3, df2 from Within Groups"}},
  {{"test_type": "F", "statistic_value": 8.46, "df1": 1, "df2": 60,
    "reported_p": 0.005, "p_equality": "=", "two_tailed": false,
    "raw_text": "Group | 12.1 | 1 | 12.10 | 8.46 | 0.005",
    "textual_interpretation": "", "interpretation_direction": "unclear",
    "consistent": null, "notes": "Table 3, df2 from Within Groups"}},
  {{"test_type": "F", "statistic_value": 3.25, "df1": 2, "df2": 60,
    "reported_p": 0.045, "p_equality": "=", "two_tailed": false,
    "raw_text": "Time x Group | 9.3 | 2 | 4.65 | 3.25 | 0.045",
    "textual_interpretation": "", "interpretation_direction": "unclear",
    "consistent": null, "notes": "Table 3, df2 from Within Groups"}}
]

(Within Groups and Total are NOT in the output — they are df-providers.)

Return a JSON array. If the table contains no statistical tests, return [].

TABLE:
\"\"\"
{table_md}
\"\"\"
"""


def extract_tables(markdown):
    tables = []
    for m in TABLE_BLOCK_RE.finditer(markdown):
        start = m.start()
        end = m.end()
        caption = (m.group("caption") or "").strip()
        # если подписи не нашли перед таблицей — поищем после
        if not caption:
            after = markdown[end:end + 300]
            m_after = re.search(
                r"(Table|Tab\.?|Таблица)\s+\d+[^\n]*",
                after, re.IGNORECASE
            )
            if m_after:
                caption = m_after.group(0).strip()
        table_md = m.group("header") + m.group("sep") + m.group("body")
        if caption:
            table_md = caption + "\n" + table_md
        tables.append((start, end, table_md, caption))
    return tables


def strip_tables(markdown, tables):
    if not tables:
        return markdown
    out = []
    cursor = 0
    for (start, end, _table_md, caption) in tables:
        out.append(markdown[cursor:start])
        placeholder = "[TABLE: " + (caption or "without caption") + "]"
        out.append("\n" + placeholder + "\n")
        cursor = end
    out.append(markdown[cursor:])
    return "".join(out)



def call_api_table(table_md, model_name, api_key=None, timeout=120):
    cfg = MODELS[model_name]
    key = api_key or os.environ.get(cfg["env_key"])
    if not key:
        raise RuntimeError(cfg["env_key"] + " не задан")

    payload = {
        "model": cfg["model"],
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_TABLE},
            {"role": "user", "content": USER_PROMPT_TABLE.format(table_md=table_md)},
        ],
    }
    return post_openai_chat(
        url=cfg["url"], api_key=key, payload=payload,
        timeout=timeout, model_hint=model_name + "-table",
    )


def extract_table_tests(table_md, caption, model_name, api_key=None):

    if not table_md.strip():
        return []
    raw = call_api_table(table_md, model_name, api_key)
    tests = parse_response(raw)
    # промечаем источник, чтобы stage 4 умел искать "Table N" контекст
    for t in tests:
        t.setdefault("notes", "")
        if caption and caption not in t["notes"]:
            t["notes"] = (caption + "; " + t["notes"]).strip("; ")
        t["_source"] = "table_sweep"
        t["_table_caption"] = caption
    return tests


def extract_tests_two_pass(text, model_name, api_key=None, clean=True):
    if not text.strip():
        return []

    tables = extract_tables(text)
    if not tables:
        # быстрый путь: без таблиц = просто baseline
        return extract_tests_baseline(text, model_name, api_key, clean=clean)

    # pass A — inline extract на тексте с placeholder'ами вместо таблиц
    text_stripped = strip_tables(text, tables)
    inline_tests = extract_tests_baseline(
        text_stripped, model_name, api_key, clean=False
    )
    for t in inline_tests:
        t.setdefault("_source", "inline")

    # pass B — отдельный вызов на каждую таблицу
    table_tests = []
    for (_start, _end, table_md, caption) in tables:
        try:
            tt = extract_table_tests(table_md, caption, model_name, api_key)
            table_tests.extend(tt)
        except Exception as e:
            # таблица упала — пропускаем, остальное продолжаем
            print(
                "    table-sweep failed (caption='" + caption + "'): " + str(e)
            )

    merged = inline_tests + table_tests

    if clean:
        return clean_predictions(merged)
    return merged


def find_table_references(markdown, table_number, window=300):
    out = []
    for m in TABLE_REF_RE.finditer(markdown):
        if int(m.group(1)) != int(table_number):
            continue
        s = max(0, m.start() - window)
        e = min(len(markdown), m.end() + window)
        out.append((s, e, markdown[s:e]))
    return out
