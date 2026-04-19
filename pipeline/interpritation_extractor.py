import os
import re
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Tuple, Optional


BODY_HEADINGS = [
    "Abstract",
    "Introduction", "Background",
    "Materials and methods", "Methods", "Method", "Methodology", "Study design",
    "Results and discussion",
    "Results", "Findings",
    "Discussion",
    "Conclusions", "Conclusion", "Concluding remarks", "Summary",
    "Limitations", "Limitation",
]
BODY_HEADINGS_SORTED = sorted(BODY_HEADINGS, key=len, reverse=True)

NAME_MAP = {
    "abstract": "abstract",
    "background": "introduction", "introduction": "introduction",
    "materials and methods": "methods", "methods": "methods", "method": "methods",
    "methodology": "methods", "study design": "methods",
    "results and discussion": "results_and_discussion",
    "results": "results", "findings": "results",
    "discussion": "discussion",
    "conclusions": "conclusion", "conclusion": "conclusion",
    "concluding remarks": "conclusion", "summary": "conclusion",
    "limitations": "limitations", "limitation": "limitations",
}

HEADING_RE = re.compile(
    r"(?<=\s)"
    r"(?:\d+\.?\d*\.?\s+)?"
    r"(" + "|".join(BODY_HEADINGS_SORTED) + r")"
    r"(?!:)"
    r"\s+"
    r"(?=[A-Z][a-z])"
)

# References / Bibliography / Acknowledgments — отдельный regex,
# в PDF они часто лежат битым табличным куском и не матчатся основным.
REFS_RE = re.compile(
    r"(?<=\s)(References|Bibliography|Acknowledgments?|Funding|Author contributions)\b(?!:)",
    re.IGNORECASE,
)
# приоритет секций при выборе primary direction
_SECTION_PRIORITY = ["results", "results_and_discussion", "discussion", "conclusion", "abstract"]


SYSTEM_PROMPT = (
    "You are a precise information-extraction system for scientific papers. "
    "For each statistical test, locate ALL places in the provided sectioned text "
    "where the authors interpret this test. A single test may be discussed in "
    "multiple sections (Results and Discussion), possibly with different framing. "
    "Return ONLY valid JSON, no markdown fences."
)

USER_PROMPT_TEMPLATE = '''\
You receive:
  1) A list of statistical tests already extracted from a paper.
  2) Sectioned text blocks from the paper. Each block is labeled with its section name.

For EACH test, scan ALL blocks and return every interpretation you find for it.
A test may be interpreted in multiple sections — return one object per section where
the test is meaningfully discussed (not just repeated as a number).

Return a JSON array, one object per input test:
  test_id            — integer, echoed from the input
  interpretations    — list of objects, each with:
      section            — string: section label from the input ("results", "discussion", ...)
      sentence           — verbatim sentence(s) carrying the interpretation
      keywords           — list of exact words/phrases signalling significance or hedging
                           ("significantly higher", "trend toward", "no difference",
                           "marginal", "approaching significance", ...)
      direction          — "significant" / "not_significant" / "marginal" / "unclear"
      effect_strength    — "strong" / "moderate" / "weak" / "none" / "unclear"
                           (what authors *claim* about magnitude)
      hedging            — boolean: hedging language present
                           ("may", "might", "suggests", "approaching", "trend toward", "appears")
      claim              — one-sentence paraphrase of the authors' point

If a test has NO interpretation anywhere in the provided blocks, return
{{"test_id": <id>, "interpretations": []}}.

TESTS:
{tests}

SECTIONED TEXT:
{sectioned_text}
'''

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



# Обработка текста статьи: разбиваем на секции, отрубаем References и тд, чтобы потом искать там интерпретации тестов.
def is_in_caption(text, start, lookback = 30):
    pre = text[max(0, start - lookback):start].upper()
    return "TABLE" in pre or "FIGURE" in pre or "FIG." in pre

def find_refs_cutoff(text, min_pos = 0):
    for m in REFS_RE.finditer(text):
        if m.start() < min_pos:
            continue
        if is_in_caption(text, m.start()):
            continue
        after = text[m.end():m.end() + 60]
        if re.match(r"[\s|]*(?:\d+\.|[A-Z])", after):
            return m.start()
    return -1


def split_sections(text):
    matches = []
    for m in HEADING_RE.finditer(text):
        if is_in_caption(text, m.start()):
            continue
        name = m.group(1).lower()
        canonical = NAME_MAP.get(name, name.replace(" ", "_"))
        matches.append((m.start(), m.end(), canonical))

    seen = set()
    unique = []
    for start, end, canonical in matches:
        if canonical in seen:
            continue
        seen.add(canonical)
        unique.append((start, end, canonical))
    unique.sort(key=lambda x: x[0])

    last_boundary = len(text)
    if unique:
        refs_pos = find_refs_cutoff(text, min_pos=unique[-1][1])
        if refs_pos != -1:
            last_boundary = refs_pos

    sections: Dict[str, str] = {}
    if unique and unique[0][0] > 0:
        sections["preamble"] = text[:unique[0][0]].strip()

    for i, (_, end, canonical) in enumerate(unique):
        next_start = unique[i + 1][0] if i + 1 < len(unique) else last_boundary
        body = text[end:next_start].strip()
        if body:
            sections[canonical] = body

    return sections


def find_local_block(sections, raw_text, window_chars = 600):
    if not raw_text:
        return None, None
    for sec_name, sec_text in sections.items():
        idx = sec_text.find(raw_text)
        if idx != -1:
            start = max(0, idx - window_chars)
            end = min(len(sec_text), idx + len(raw_text) + window_chars)
            return sec_name, sec_text[start:end]
    # ищем по первому числу из raw_text (если raw_text пришёл из таблицы
    # с другими пробелами и буквальный поиск не сработал)
    m = re.search(r"-?\d+\.\d+", raw_text)
    if m:
        num = m.group()
        for sec_name, sec_text in sections.items():
            idx = sec_text.find(num)
            if idx != -1:
                start = max(0, idx - window_chars)
                end = min(len(sec_text), idx + window_chars)
                return sec_name, sec_text[start:end]
    return None, None


def build_test_context(sections, raw_text, window_chars = 600, extra_sections = ("discussion", "conclusion", "results_and_discussion")):
    blocks = []
    local_sec, local_snip = find_local_block(sections, raw_text, window_chars)
    if local_snip:
        blocks.append((local_sec, local_snip))
    for name in extra_sections:
        if name in sections and name != local_sec:
            blocks.append((name, sections[name]))
    return blocks


def slim_test(t, test_id):
    return {
        "test_id": test_id,
        "raw_text": t.get("raw_text", ""),
        "test_type": t.get("test_type", ""),
        "statistic_value": t.get("statistic_value"),
        "reported_p": t.get("reported_p"),
    }


def format_sectioned_text(blocks_per_test):
    seen = set()
    parts = []
    for blocks in blocks_per_test:
        for sec, snip in blocks:
            key = (sec, snip[:50])
            if key in seen:
                continue
            seen.add(key)
            parts.append(f"[SECTION: {sec}]\n{snip}")
    return "\n\n===\n\n".join(parts)


# Вызов LLM API для извлечения интерпретаций тестов из текстовых блоков
def call_api(sectioned_text, tests, model_name, api_key = None, timeout = 180):
    cfg = MODELS[model_name]
    key = api_key or os.environ.get(cfg["env_key"])
    if not key:
        raise RuntimeError(f"{cfg['env_key']} не задан")

    slim = [slim_test(t, i) for i, t in enumerate(tests)]
    payload = json.dumps({
        "model": cfg["model"],
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                tests=json.dumps(slim, ensure_ascii=False, indent=2),
                sectioned_text=sectioned_text,
            )},
        ],
    }).encode()

    req = urllib.request.Request(
        cfg["url"], data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def parse_response(raw):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"(\[.*])", cleaned, re.DOTALL)
        data = json.loads(m.group(1)) if m else []

    if isinstance(data, dict):
        data = next((data[k] for k in ("results", "items") if isinstance(data.get(k), list)), [])
    if not isinstance(data, list):
        return []

    allowed_dir = {"significant", "not_significant", "marginal", "unclear"}
    allowed_str = {"strong", "moderate", "weak", "none", "unclear"}

    out = []
    for r in data:
        interps_raw = r.get("interpretations") or []
        interps = []
        for it in interps_raw:
            direction = str(it.get("direction", "unclear")).strip()
            strength = str(it.get("effect_strength", "unclear")).strip()
            kws = it.get("keywords") or []
            if not isinstance(kws, list):
                kws = [str(kws)]
            interps.append({
                "section": str(it.get("section", "")).strip(),
                "sentence": str(it.get("sentence", "")).strip(),
                "keywords": [str(k).strip() for k in kws if str(k).strip()],
                "direction": direction if direction in allowed_dir else "unclear",
                "effect_strength": strength if strength in allowed_str else "unclear",
                "hedging": bool(it.get("hedging", False)),
                "claim": str(it.get("claim", "")).strip(),
            })
        out.append({"test_id": r.get("test_id"), "interpretations": interps})
    return out


def extract_interpretations(sections, tests, model_name, api_key = None):
    """sections (split_sections) + tests (stats_extractor) -> интерпретации по каждому тесту."""
    if not tests:
        return []
    blocks_per_test = [build_test_context(sections, t.get("raw_text", "")) for t in tests]
    sectioned_text = format_sectioned_text(blocks_per_test)
    raw = call_api(sectioned_text, tests, model_name, api_key)
    return parse_response(raw)




def aggregate(interps):
    if not interps:
        return {
            "primary_direction": "unclear",
            "has_cross_section_conflict": False,
            "any_hedging": False,
            "sections": [],
        }
    by_sec = {i["section"]: i for i in interps if i.get("section")}
    primary = None
    for s in _SECTION_PRIORITY:
        if s in by_sec:
            primary = by_sec[s]["direction"]
            break
    if primary is None:
        primary = interps[0]["direction"]
    directions = {i["direction"] for i in interps if i["direction"] != "unclear"}
    return {
        "primary_direction": primary,
        "has_cross_section_conflict": len(directions) > 1,
        "any_hedging": any(i.get("hedging") for i in interps),
        "sections": [i["section"] for i in interps],
    }


def merge_with_tests(tests, interpretations):
    by_id = {i["test_id"]: i for i in interpretations if i.get("test_id") is not None}
    merged = []
    for idx, t in enumerate(tests):
        rec = by_id.get(idx, {"interpretations": []})
        agg = aggregate(rec["interpretations"])
        merged.append({**t, "interpretations": rec["interpretations"], **agg})
    return merged
