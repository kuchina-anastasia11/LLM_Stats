#  few-shot для извлечения тестов из таблиц
# DeepSeek пропускает F-тесты в SPSS-таблицах, Gemini наоборот дублирует строки.
# добавляем 2-3 примера в промпт чтобы лечить обе боли.
# few-shot блок добавляется ТОЛЬКО если в тексте есть markdown-таблица
import os
import re

from pipeline._llm_client import post_openai_chat
from pipeline.stats_extractor import (
    SYSTEM_PROMPT, MODELS, parse_response, clean_predictions,
)


# регулярка маркера markdown-таблицы (строка-разделитель)
TABLE_MARKER_RE = re.compile(r"\|\s*-{2,}\s*\|")


def has_markdown_table(text):
    return bool(TABLE_MARKER_RE.search(text))


FEW_SHOT_BLOCK = """\
========================
EXAMPLES OF TABLE EXTRACTION
========================

EXAMPLE 1 — ANOVA / SPSS-style output table with multiple F-tests.

INPUT:
| Source         | Sum of Squares | df | Mean Square | F     | Sig.  |
| ---            | ---            | ---| ---         | ---   | ---   |
| Treatment      | 24.31          | 2  | 12.16       | 8.45  | 0.000 |
| Within Groups  | 86.12          | 60 | 1.44        |       |       |
| Total          | 110.43         | 62 |             |       |       |
| Time           |  9.18          | 1  |  9.18       | 6.37  | 0.014 |
| Time x Group   | 14.22          | 2  |  7.11       | 4.93  | 0.010 |
| Error          | 86.40          | 60 | 1.44        |       |       |

EXPECTED OUTPUT:
[
  {
    "test_type": "F", "statistic_value": 8.45, "df1": 2, "df2": 60,
    "reported_p": 0.000, "p_equality": "=", "two_tailed": false,
    "raw_text": "Treatment | 24.31 | 2 | 12.16 | 8.45 | 0.000",
    "textual_interpretation": "", "interpretation_direction": "unclear",
    "consistent": null,
    "notes": "ANOVA, df2 from Within Groups row"
  },
  {
    "test_type": "F", "statistic_value": 6.37, "df1": 1, "df2": 60,
    "reported_p": 0.014, "p_equality": "=", "two_tailed": false,
    "raw_text": "Time | 9.18 | 1 | 9.18 | 6.37 | 0.014",
    "textual_interpretation": "", "interpretation_direction": "unclear",
    "consistent": null,
    "notes": "ANOVA, df2 from Error row"
  },
  {
    "test_type": "F", "statistic_value": 4.93, "df1": 2, "df2": 60,
    "reported_p": 0.010, "p_equality": "=", "two_tailed": false,
    "raw_text": "Time x Group | 14.22 | 2 | 7.11 | 4.93 | 0.010",
    "textual_interpretation": "", "interpretation_direction": "unclear",
    "consistent": null,
    "notes": "ANOVA interaction, df2 from Error row"
  }
]

CRITICAL RULES for ANOVA / SPSS tables:
- Each row with a NUMERICAL F value is a SEPARATE test. Extract all of them.
- Rows like "Within Groups", "Error", "Residual", "Total" are NOT tests —
  they exist to provide df2 (their df column) for the F-tests in their block.
- df1 = the F-row's own df. df2 = the df from the nearest "Within Groups"
  / "Error" / "Residual" row in the same block.

========================

EXAMPLE 2 — pairwise t-test table.

INPUT:
| Comparison           | t     | df | p     |
| ---                  | ---   | ---| ---   |
| Pre vs Post          |  3.45 | 28 | 0.002 |
| Treatment vs Control | -2.18 | 56 | 0.034 |

EXPECTED OUTPUT:
[
  {
    "test_type": "t", "statistic_value": 3.45, "df1": 28, "df2": null,
    "reported_p": 0.002, "p_equality": "=", "two_tailed": true,
    "raw_text": "Pre vs Post | 3.45 | 28 | 0.002",
    "textual_interpretation": "", "interpretation_direction": "unclear",
    "consistent": null, "notes": ""
  },
  {
    "test_type": "t", "statistic_value": -2.18, "df1": 56, "df2": null,
    "reported_p": 0.034, "p_equality": "=", "two_tailed": true,
    "raw_text": "Treatment vs Control | -2.18 | 56 | 0.034",
    "textual_interpretation": "", "interpretation_direction": "unclear",
    "consistent": null, "notes": ""
  }
]

GENERAL RULES for any table:
- DO NOT duplicate: each numerical row corresponds to AT MOST ONE test.
- If the same statistic appears in inline text AND in a table, return it ONCE.
- Preserve sign of statistic_value (negative t / r values are valid).
========================
"""


USER_PROMPT_FEW_SHOT = """\
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

{few_shot_block}

Return a JSON array. If no tests are found, return [].

TEXT:
\"\"\"
{text}
\"\"\"
"""


def call_api_few_shot(text, model_name, api_key=None, timeout=180):
    cfg = MODELS[model_name]
    key = api_key or os.environ.get(cfg["env_key"])
    if not key:
        raise RuntimeError(cfg["env_key"] + " не задан")

    # подмешиваем few-shot блок только если есть таблица
    if has_markdown_table(text):
        fs_block = FEW_SHOT_BLOCK
    else:
        fs_block = ""

    payload = {
        "model": cfg["model"],
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_FEW_SHOT.format(
                few_shot_block=fs_block,
                text=text,
            )},
        ],
    }
    return post_openai_chat(
        url=cfg["url"], api_key=key, payload=payload,
        timeout=timeout, model_hint=model_name + "-fewshot",
    )


def extract_tests_few_shot(text, model_name, api_key=None, clean=True):
    # drop-in замена stats_extractor.extract_tests с few-shot
    if not text.strip():
        return []
    raw = call_api_few_shot(text, model_name, api_key)
    parsed = parse_response(raw)
    if clean:
        return clean_predictions(parsed)
    return parsed
