import os
import re
import json


# заголовки которые ищем в тексте статьи
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
    "background": "introduction",
    "introduction": "introduction",
    "materials and methods": "methods",
    "methods": "methods",
    "method": "methods",
    "methodology": "methods",
    "study design": "methods",
    "results and discussion": "results_and_discussion",
    "results": "results",
    "findings": "results",
    "discussion": "discussion",
    "conclusions": "conclusion",
    "conclusion": "conclusion",
    "concluding remarks": "conclusion",
    "summary": "conclusion",
    "limitations": "limitations",
    "limitation": "limitations",
}

HEADING_RE = re.compile(
    r"(?<=\s)"
    r"(?:\d+\.?\d*\.?\s+)?"
    r"(" + "|".join(BODY_HEADINGS_SORTED) + r")"
    r"(?!:)"
    r"\s+"
    r"(?=[A-Z][a-z])"
)

REFS_RE = re.compile(
    r"(?<=\s)(References|Bibliography|Acknowledgments?|Funding|Author contributions)\b(?!:)",
    re.IGNORECASE,
)

# приоритет секций для выбора primary_direction
SECTION_PRIORITY = ["results", "results_and_discussion", "discussion", "conclusion", "abstract"]


SYSTEM_PROMPT = (
    "You are a precise information-extraction system for scientific papers. "
    "For each statistical test, locate ALL places in the provided sectioned text "
    "where the authors interpret this test. A single test may be discussed in "
    "multiple sections (Results and Discussion), possibly with different framing. "
    "Return ONLY valid JSON, no markdown fences."
)

USER_PROMPT_TEMPLATE = '''\
You receive:
  1) A list of statistical tests already extracted from a paper. Each test has
     a `raw_text` (the substring it came from), `test_type`, `statistic_value`,
     `df1`, `df2`, `reported_p`.
  2) Sectioned text blocks from the paper. Each block is labeled with its
     section name. Blocks labelled like "discussion::table_ref" are windows
     around a "Table N" mention — they often contain the textual interpretation
     of a test that was originally extracted from that table.

For EACH test, scan ALL blocks and return every interpretation you find for it.
A test may be interpreted in multiple sections — return one object per section
where the test is meaningfully discussed (not just repeated as a number).

MATCHING TESTS TO INTERPRETATIONS:
  - If `raw_text` is a normal sentence fragment (e.g. "t(28) = 2.20, p = .03"),
    match by literal substring AND by statistic_value.
  - If `raw_text` looks like a markdown TABLE ROW (contains "|" separators,
    e.g. "Treatment | 8.45 | 2 | 60 | 0.001"), do NOT try to find this row
    verbatim in the prose sections. Instead, MATCH BY VALUE: find sentences
    that mention the same `statistic_value` (e.g. 8.45) together with the
    same df (2, 60), or that reference "Table N" where this test lives.
    The interpretation is in the prose, not in the table cell.
  - When in doubt about which sentence belongs to which test, prefer using
    `statistic_value` and df as anchors — multiple tests of the same type in
    one paper are distinguished by their numbers.

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


# режем текст на секции и удаляем References, чтобы потом искать там интерпретации

def is_in_caption(text, start, lookback=30):
    # проверяем находится ли позиция в подписи к таблице/рисунку
    pre = text[max(0, start - lookback):start].upper()
    return "TABLE" in pre or "FIGURE" in pre or "FIG." in pre


def find_refs_cutoff(text, min_pos=0):
    # ищем границу References
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
    # бьём текст на секции по заголовкам
    matches = []
    for m in HEADING_RE.finditer(text):
        if is_in_caption(text, m.start()):
            continue
        name = m.group(1).lower()
        canonical = NAME_MAP.get(name, name.replace(" ", "_"))
        matches.append((m.start(), m.end(), canonical))

    # оставляем по одному вхождению каждой секции
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

    sections = {}
    if unique and unique[0][0] > 0:
        sections["preamble"] = text[:unique[0][0]].strip()

    for i, (_, end, canonical) in enumerate(unique):
        if i + 1 < len(unique):
            next_start = unique[i + 1][0]
        else:
            next_start = last_boundary
        body = text[end:next_start].strip()
        if body:
            sections[canonical] = body

    return sections


def find_local_block(sections, raw_text, window_chars=600):
    # ищем raw_text в секциях и возвращаем окно вокруг него
    if not raw_text:
        return None, None
    for sec_name, sec_text in sections.items():
        idx = sec_text.find(raw_text)
        if idx != -1:
            start = max(0, idx - window_chars)
            end = min(len(sec_text), idx + len(raw_text) + window_chars)
            return sec_name, sec_text[start:end]

    # если не нашли - ищем по первому числу из raw_text
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


# регулярка для извлечения номера таблицы из подписи (Table 3, Tab. 3, Таблица 3)
TABLE_NUM_RE = re.compile(r"(?:Table|Tab\.?|Таблица)\s+(\d+)", re.IGNORECASE)


def find_table_ref_blocks(sections, table_num, window_chars=300):
    """Для табличных тестов: ищем упоминания 'Table N' во всех секциях
    и возвращаем окна вокруг каждого. Это нужно потому что raw_text
    табличной строки буквально не встречается в Discussion/Conclusion,
    но автор там цитирует таблицу как 'Table 3 shows...' — это и есть
    интерпретация, которую stage 4 должен подхватить."""
    try:
        n = int(table_num)
    except (TypeError, ValueError):
        return []
    pattern = re.compile(
        r"(?:Table|Tab\.?|Таблица)\s+" + str(n) + r"\b",
        re.IGNORECASE,
    )
    blocks = []
    for sec_name, sec_text in sections.items():
        for m in pattern.finditer(sec_text):
            s = max(0, m.start() - window_chars)
            e = min(len(sec_text), m.end() + window_chars)
            blocks.append((sec_name + "::table_ref", sec_text[s:e]))
    return blocks


def build_test_context(sections, raw_text, window_chars=600,
                       extra_sections=("discussion", "conclusion", "results_and_discussion"),
                       test=None):
    # собираем релевантные блоки текста для каждого теста
    blocks = []
    local_sec, local_snip = find_local_block(sections, raw_text, window_chars)
    if local_snip:
        blocks.append((local_sec, local_snip))
    for name in extra_sections:
        if name in sections and name != local_sec:
            blocks.append((name, sections[name]))
    # для табличных тестов (из table_sweep) подсасываем окна вокруг "Table N"
    if test and test.get("_source") == "table_sweep":
        caption = test.get("_table_caption") or ""
        m = TABLE_NUM_RE.search(caption)
        if m:
            blocks.extend(find_table_ref_blocks(sections, m.group(1)))
    return blocks


def slim_test(t, test_id):
    # выкусываем только нужные поля для промпта; df1/df2 нужны как anchor
    # для табличных тестов, где raw_text = строка с пайпами
    return {
        "test_id": test_id,
        "raw_text": t.get("raw_text", ""),
        "test_type": t.get("test_type", ""),
        "statistic_value": t.get("statistic_value"),
        "df1": t.get("df1"),
        "df2": t.get("df2"),
        "reported_p": t.get("reported_p"),
    }


def format_sectioned_text(blocks_per_test):
    # склеиваем блоки в один текст, без дублей
    seen = set()
    parts = []
    for blocks in blocks_per_test:
        for sec, snip in blocks:
            key = (sec, snip[:50])
            if key in seen:
                continue
            seen.add(key)
            parts.append("[SECTION: " + str(sec) + "]\n" + snip)
    return "\n\n===\n\n".join(parts)


def call_api(sectioned_text, tests, model_name, api_key=None, timeout=180):
    # вызов LLM для извлечения интерпретаций
    from pipeline._llm_client import post_openai_chat

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
                tests=json.dumps(slim, ensure_ascii=False, indent=2),
                sectioned_text=sectioned_text,
            )},
        ],
    }
    return post_openai_chat(
        url=cfg["url"], api_key=key, payload=payload,
        timeout=timeout, model_hint=model_name,
    )


def parse_response(raw):
    # парсим ответ модели
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
        # иногда модель оборачивает массив в объект
        found = []
        for k in ("results", "items"):
            if isinstance(data.get(k), list):
                found = data[k]
                break
        data = found

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
            cleaned_kws = []
            for k in kws:
                if str(k).strip():
                    cleaned_kws.append(str(k).strip())

            if direction not in allowed_dir:
                direction = "unclear"
            if strength not in allowed_str:
                strength = "unclear"

            interps.append({
                "section": str(it.get("section", "")).strip(),
                "sentence": str(it.get("sentence", "")).strip(),
                "keywords": cleaned_kws,
                "direction": direction,
                "effect_strength": strength,
                "hedging": bool(it.get("hedging", False)),
                "claim": str(it.get("claim", "")).strip(),
            })
        out.append({"test_id": r.get("test_id"), "interpretations": interps})
    return out


def extract_interpretations(sections, tests, model_name, api_key=None):
    # главная функция: sections + tests -> интерпретации
    if not tests:
        return []
    blocks_per_test = []
    for t in tests:
        blocks_per_test.append(
            build_test_context(sections, t.get("raw_text", ""), test=t)
        )
    sectioned_text = format_sectioned_text(blocks_per_test)
    raw = call_api(sectioned_text, tests, model_name, api_key)
    return parse_response(raw)


def aggregate(interps):
    # агрегируем интерпретации одного теста
    if not interps:
        return {
            "primary_direction": "unclear",
            "has_cross_section_conflict": False,
            "any_hedging": False,
            "sections": [],
        }

    by_sec = {}
    for i in interps:
        if i.get("section"):
            by_sec[i["section"]] = i

    # primary_direction берём по приоритету секций
    primary = None
    for s in SECTION_PRIORITY:
        if s in by_sec:
            primary = by_sec[s]["direction"]
            break
    if primary is None:
        primary = interps[0]["direction"]

    directions = set()
    for i in interps:
        if i["direction"] != "unclear":
            directions.add(i["direction"])

    return {
        "primary_direction": primary,
        "has_cross_section_conflict": len(directions) > 1,
        "any_hedging": any(i.get("hedging") for i in interps),
        "sections": [i["section"] for i in interps],
    }


def merge_with_tests(tests, interpretations):
    # склеиваем тесты с интерпретациями по test_id
    by_id = {}
    for i in interpretations:
        if i.get("test_id") is not None:
            by_id[i["test_id"]] = i

    merged = []
    for idx, t in enumerate(tests):
        rec = by_id.get(idx, {"interpretations": []})
        agg = aggregate(rec["interpretations"])
        new_t = dict(t)
        new_t["interpretations"] = rec["interpretations"]
        new_t.update(agg)
        merged.append(new_t)
    return merged
