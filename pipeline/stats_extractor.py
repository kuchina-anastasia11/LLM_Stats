import re
import fitz  # PyMuPDF
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import pdfplumber
import sys
import json, os, urllib.request, urllib.error



# Укажим промпты для обработки текста
# на данном жтапе важно чтобы текст доставался структурировано, чет в описанном формате, без лишних знаков/слов/и тд
os.environ["GEMINI_API_KEY"] = ""
os.environ["DEEPSEEK_API_KEY"] = ""

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

# Настроим модели, которые планируем использовать и их апи
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
}

# Функция возвращает сырой ответ модели, далее будем его обрабатывать
def call_api(text, model_name, api_key=None, timeout=120):
    cfg = MODELS[model_name]
    key = api_key or os.environ.get(cfg["env_key"])
    if not key:
        raise RuntimeError(f"{cfg['env_key']} не задан")

    payload = json.dumps({
        "model": cfg["model"],
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
        ],
    }).encode()

    req = urllib.request.Request(
        cfg["url"], data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def parse_response(raw):
      def to_float(v):
          try:
              return float(v) if v not in (None, "") else None
          except (TypeError, ValueError):
              return None

      cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
      try:
          data = json.loads(cleaned)
      except json.JSONDecodeError:
          m = re.search(r"(\[.*])", cleaned, re.DOTALL)
          data = json.loads(m.group(1)) if m else []

      if isinstance(data, dict):
          data = next((data[k] for k in ("tests", "results") if isinstance(data.get(k), list)), [])
      if not isinstance(data, list):
          return []

      return [
          {
              "test_type": str(r.get("test_type", "")).strip(),
              "statistic_value": to_float(r.get("statistic_value")),
              "df1": to_float(r.get("df1")),
              "df2": to_float(r.get("df2")),
              "reported_p": to_float(r.get("reported_p")),
              "p_equality": r.get("p_equality"),
              "two_tailed": r.get("two_tailed", True),
              "raw_text": str(r.get("raw_text", "")),
              "textual_interpretation": str(r.get("textual_interpretation", "")),
              "interpretation_direction": str(r.get("interpretation_direction", "unclear")),
              "consistent": r.get("consistent"),
              "notes": str(r.get("notes", "")),
          }
          for r in data
      ]


def extract_tests(text, model_name, api_key=None):
      if not text.strip():
          return []
      raw = call_api(text, model_name, api_key)
      return parse_response(raw)
