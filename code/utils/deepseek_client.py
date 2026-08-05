import re, time
from openai import OpenAI

_LEAK_RE = re.compile(r"<think>.*?</think>|<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE)


class DeepSeekHybridClient:
    """LLM -> DeepSeek (OpenAI-compatible); embed/rerank -> XInference bge."""

    def __init__(self, deepseek_base_url, deepseek_api_key, xinference_base_url):
        self.llm = OpenAI(base_url=deepseek_base_url, api_key=deepseek_api_key)
        from utils.xinference_client import XInferenceClient
        self._bge = XInferenceClient(xinference_base_url)
        print(f"[ok] DeepSeekHybridClient: LLM={deepseek_base_url} | bge={xinference_base_url}")

    def chat(self, messages, model=None, temperature=0.0, max_tokens=200, timeout=None):
        from config import config
        model = model or config.LLM_MODEL
        timeout = timeout or getattr(config, "LLM_TIMEOUT", 90)
        extra_body = getattr(config, "LLM_CHAT_EXTRA_BODY", None) or {}
        sysp = getattr(config, "LLM_SYSTEM_PROMPT", None) or None
        msgs = list(messages)
        if sysp and msgs and msgs[0].get("role") != "system":
            msgs = [{"role": "system", "content": sysp}, *msgs]
        last = None
        for attempt in range(4):
            try:
                resp = self.llm.chat.completions.create(
                    model=model, messages=msgs, temperature=temperature,
                    max_tokens=max_tokens, timeout=timeout,
                    extra_body=extra_body if extra_body else None)
                msg = resp.choices[0].message
                content = (msg.content or "").strip()
                reasoning = getattr(msg, "reasoning_content", None)
                if not content:
                    raise RuntimeError(
                        f"DeepSeek empty content (reasoning_content len="
                        f"{len(reasoning or '')}): thinking not disabled or max_tokens too small.")
                return _LEAK_RE.sub("", content).strip()
            except Exception as e:
                last = e
                if attempt == 3:
                    break
                time.sleep(2 ** attempt)
        raise Exception(f"LLM call failed: {last}")

    def embed(self, texts, model=None):
        return self._bge.embed(texts, model=model)

    def rerank(self, query, documents, model=None, top_k=5):
        return self._bge.rerank(query, documents, model=model, top_k=top_k)

    def health_check(self):
        return self._bge.health_check()
