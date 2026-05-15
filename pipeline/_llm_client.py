import json
import sys
import time
import urllib.error
import urllib.request

# настройки ретраев
MAX_RETRIES = 4
BACKOFF_BASE_S = 5
RETRY_CODES = {429, 500, 502, 503, 504}


def post_openai_chat(url, api_key, payload, timeout=120, model_hint=""):
    # отправляем запрос в openai-совместимый чат
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
    )

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"]

        except urllib.error.HTTPError as e:
            # читаем тело ошибки
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            last_error = (e.code, err_body)

            # если код ретраебельный - спим и повторяем
            if e.code in RETRY_CODES and attempt < MAX_RETRIES - 1:
                wait = BACKOFF_BASE_S * (3 ** attempt)
                print("  [retry] HTTP " + str(e.code) +
                      " (attempt " + str(attempt + 1) + "/" + str(MAX_RETRIES) + "), "
                      "waiting " + str(wait) + "s...",
                      file=sys.stderr, flush=True)
                time.sleep(wait)
                continue

            # иначе кидаем ошибку дальше
            short_body = err_body[:500].replace("\n", " ")
            print("  [ERROR] HTTP " + str(e.code) + " " + e.reason + ": " + short_body,
                  file=sys.stderr, flush=True)
            raise urllib.error.HTTPError(
                e.url, e.code,
                e.reason + " — body: " + short_body,
                e.headers, None,
            ) from None

        except urllib.error.URLError as e:
            # сеть упала - тоже ретраим
            last_error = (None, str(e))
            if attempt < MAX_RETRIES - 1:
                wait = BACKOFF_BASE_S * (3 ** attempt)
                print("  [retry] " + type(e).__name__ + ": " + str(e) +
                      " (attempt " + str(attempt + 1) + "/" + str(MAX_RETRIES) + "), "
                      "waiting " + str(wait) + "s...",
                      file=sys.stderr, flush=True)
                time.sleep(wait)
                continue
            raise

    raise RuntimeError("All " + str(MAX_RETRIES) + " attempts failed. Last error: " + str(last_error))
