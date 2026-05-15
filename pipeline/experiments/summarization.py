# в baseline этап суммаризации отключён - извлечение работает на полном markdown.
# тут добавляем сжатие текста, сохраняя все числа, чтобы потом
# stats_extractor работал по более короткой версии
import os
from pipeline._llm_client import post_openai_chat
from pipeline.stats_extractor import MODELS


SYSTEM_PROMPT = (
    "You are a precise text condenser for scientific papers. "
    "Your job is to shorten section text while preserving EVERY numerical "
    "value that could belong to a statistical test result. "
    "Return ONLY the condensed text — no commentary, no markdown fences, "
    "no preface like 'Here is the summary'."
)

USER_PROMPT_TEMPLATE = """\
Condense the following section. STRICTLY preserve:
- All numerical values: test statistics (t, F, chi-square, z, r, Q), degrees of freedom, p-values, alpha, effect sizes (d, eta-squared, r)
- Test type indicators: "t-test", "ANOVA", "chi-square", "correlation", "Q-test", "MANOVA", etc.
- Significance language verbatim: "significant", "not significant", "marginally significant", "approaching significance", "no difference"
- Hedging language: "trend toward", "may", "suggests", "appears"
- ALL markdown tables (any line containing `|`): copy them character-by-character, do NOT modify, do NOT summarise rows
- Sentences that contain a test result OR an interpretation of a test result — copy verbatim

You MAY remove:
- Methodology details that are not part of a test (e.g. recruitment procedures)
- Background literature and citations not tied to a numerical result
- Figure captions without numbers
- Repetitive descriptive prose

Target length: 40-60% of original. If the section is short (<500 chars), return it unchanged.

Do not add any text that wasn't in the original. Do not paraphrase numerical results.

SECTION_NAME: {section_name}
SECTION_TEXT:
\"\"\"
{section_text}
\"\"\"
"""


# короткие секции не суммаризируем
MIN_SECTION_LEN = 500
# короткие документы тоже не трогаем (чтобы не тратить токены на вызов API, если и так коротко)
MIN_DOC_LEN = 5000


def call_api(section_text, section_name, model_name, api_key, timeout=120):
    # вызов LLM на суммаризацию одной секции
    cfg = MODELS[model_name]
    key = api_key or os.environ.get(cfg["env_key"])
    if not key:
        raise RuntimeError(cfg["env_key"] + " не задан")
    payload = {
        "model": cfg["model"],
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                section_name=section_name,
                section_text=section_text,
            )},
        ],
    }
    return post_openai_chat(
        url=cfg["url"], api_key=key, payload=payload,
        timeout=timeout, model_hint=model_name + "-summarise",
    )


def summarize_section(section_name, section_text, model_name, api_key):
    # суммаризируем одну секцию
    if len(section_text) < MIN_SECTION_LEN:
        return section_text, False
    out = call_api(section_text, section_name, model_name, api_key)
    out = out.strip()
    # на всякий случай чистим обёртки ```
    if out.startswith("```"):
        out = out.strip("`").lstrip("markdown").strip()
    # если пусто или сильно короткое - возвращаем оригинал
    if not out or len(out) < 0.1 * len(section_text):
        return section_text, False
    return out, True


def summarize_markdown(markdown, model_name, api_key, min_doc_len=MIN_DOC_LEN):
    # суммаризируем markdown посекционно
    from pipeline.interpritation_extractor import split_sections

    if len(markdown) < min_doc_len:
        return markdown, {
            "applied": False,
            "skip_reason": "below_threshold",
            "n_calls": 0,
            "n_sections": 0,
            "len_before": len(markdown),
            "len_after": len(markdown),
            "ratio": 1.0,
        }

    sections = split_sections(markdown)
    if not sections:
        return markdown, {
            "applied": False,
            "skip_reason": "no_sections",
            "n_calls": 0,
            "n_sections": 0,
            "len_before": len(markdown),
            "len_after": len(markdown),
            "ratio": 1.0,
        }

    out_parts = []
    n_calls = 0
    for sec_name, sec_text in sections.items():
        condensed, was_called = summarize_section(sec_name, sec_text, model_name, api_key)
        if was_called:
            n_calls += 1
        # preamble - всё что до первого заголовка, без заголовка
        if sec_name == "preamble":
            out_parts.append(condensed)
        else:
            heading = sec_name.replace("_", " ").title()
            out_parts.append(heading + "\n\n" + condensed)

    # начинаем с newline чтобы регекс заголовков HEADING_RE мог увидеть пробел перед заголовком
    summarised = "\n" + "\n\n".join(out_parts)
    return summarised, {
        "applied": True,
        "skip_reason": None,
        "n_calls": n_calls,
        "n_sections": len(sections),
        "len_before": len(markdown),
        "len_after": len(summarised),
        "ratio": round(len(summarised) / max(len(markdown), 1), 3),
    }
