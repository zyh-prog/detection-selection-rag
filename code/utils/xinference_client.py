"""
XInference client: a thin wrapper over the OpenAI-compatible XInference
endpoints for LLM chat, embedding, and rerank, with retries and graceful
degradation.
"""

from openai import OpenAI
import requests
import re
import time
from typing import List, Union, Dict, Any


class XInferenceClient:
    """Unified access to the XInference models (LLM / embedding / rerank)."""

    def __init__(self, base_url: str = "http://localhost:9997"):
        self.base_url = base_url
        self.xinference_url = base_url

        self.llm = OpenAI(
            api_key="not-needed",
            base_url=f"{base_url}/v1"
        )

        # failure counters
        self._embed_failures = 0
        self._rerank_failures = 0
        self._max_failures = 50

        print(f"[ok] XInference client initialized: {base_url}")

    # ==================== LLM ====================

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.0,      # default 0.0 for reproducibility
        max_tokens: int = 200,
        timeout: int = None
    ) -> str:
        """
        Call the LLM (OpenAI-compatible; works for Qwen2.5-Instruct / Qwen3 etc.).

        Args:
            messages:    OpenAI-format message list
            model:       model name, defaults to config.LLM_MODEL
            temperature: sampling temperature, 0.0 for reproducibility
            max_tokens:  max generated tokens
            timeout:     request timeout in seconds, defaults to config.LLM_TIMEOUT

        Returns:
            str: model output (chain-of-thought tags stripped if present)
        """
        extra_body = None
        system_prompt = None
        try:
            from config import config
            model = model or config.LLM_MODEL
            timeout = timeout or config.LLM_TIMEOUT
            extra_body = getattr(config, "LLM_CHAT_EXTRA_BODY", None)
            system_prompt = getattr(config, "LLM_SYSTEM_PROMPT", None) or None
        except ImportError:
            model = model or "Qwen2.5-7B-Instruct"
            timeout = timeout or 60

        msgs = list(messages)
        if system_prompt and msgs and msgs[0].get("role") != "system":
            msgs = [{"role": "system", "content": system_prompt}, *msgs]

        create_kwargs: Dict[str, Any] = {}
        if extra_body:
            create_kwargs["extra_body"] = extra_body

        try:
            response = self.llm.chat.completions.create(
                model=model,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **create_kwargs,
            )

            content = response.choices[0].message.content or ""

            # strip chain-of-thought blocks some models emit (Qwen3 etc.)
            for pattern in (
                r"<think>.*?</think>",
                r"<reasoning>.*?</reasoning>",
            ):
                if re.search(pattern, content, flags=re.DOTALL | re.IGNORECASE):
                    content = re.sub(pattern, "", content, flags=re.DOTALL | re.IGNORECASE).strip()

            return content

        except Exception as e:
            raise Exception(f"LLM call failed: {str(e)}")

    # ==================== embedding ====================

    def embed(
        self,
        texts: Union[str, List[str]],
        model: str = None
    ) -> List[List[float]]:
        """
        Embed text.

        Args:
            texts: a single string or a list of strings
            model: embedding model name, defaults to config.EMBEDDING_MODEL

        Returns:
            List[List[float]]: one vector per input text
        """
        if isinstance(texts, str):
            texts = [texts]

        try:
            from config import config
            model = model or config.EMBEDDING_MODEL
        except ImportError:
            model = model or "bge-large-en-v1.5"

        return self._embed_batch(texts, model)

    def _embed_batch(
        self,
        texts: List[str],
        model: str,
        max_retries: int = 10
    ) -> List[List[float]]:
        """Embedding request with exponential backoff."""

        for attempt in range(max_retries):
            try:
                url = f"{self.xinference_url}/v1/embeddings"
                response = requests.post(
                    url,
                    json={"model": model, "input": texts},
                    timeout=120
                )
                response.raise_for_status()
                return [item['embedding'] for item in response.json()['data']]

            except Exception as e:
                self._embed_failures += 1

                if self._embed_failures == self._max_failures:
                    print(f"\n  [warn] embedding failed {self._embed_failures} times; "
                          f"check the XInference server.")

                wait_time = min(2 ** attempt, 30)   # exponential backoff, capped at 30s
                print(f"  [warn] embedding failed (attempt {attempt+1}/{max_retries}): "
                      f"{e}, waiting {wait_time}s...")
                time.sleep(wait_time)

        # all retries failed -> raise so the outer resume logic takes over
        raise Exception(
            f"Embedding service unresponsive after {max_retries} retries; "
            f"check the XInference server."
        )

    # ==================== rerank ====================

    def rerank(
            self,
            query: str,
            documents: List[str],
            model: str = None,
            top_k: int = 5
    ) -> List[Dict[str, Any]]:
        try:
            from config import config
            model = model or config.RERANK_MODEL
            timeout = int(getattr(config, "RERANK_TIMEOUT", 30))
        except ImportError:
            model = model or "bge-reranker-v2-m3"
            timeout = 30

        top_k = min(top_k, len(documents))

        try:
            url = f"{self.xinference_url}/v1/rerank"
            response = requests.post(
                url,
                json={
                    "model": model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_k
                },
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()['results']
        except Exception as e:
            self._rerank_failures += 1
            print(f"  [warn] rerank failed, falling back to original order (equal scores): {e}")
            return [{'index': i, 'relevance_score': 0.5} for i in range(top_k)]

    def health_check(self) -> bool:
        """Check whether the XInference service is reachable."""
        try:
            resp = requests.get(
                f"{self.xinference_url}/v1/models",
                timeout=5
            )
            return resp.status_code == 200
        except Exception:
            return False

    def get_available_models(self) -> List[str]:
        """List deployed models."""
        try:
            resp = requests.get(
                f"{self.xinference_url}/v1/models",
                timeout=5
            )
            resp.raise_for_status()
            return [m.get('id', '') for m in resp.json().get('data', [])]
        except Exception as e:
            print(f"  [warn] could not list models: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Return client statistics."""
        return {
            'embed_failures': self._embed_failures,
            'rerank_failures': self._rerank_failures,
            'embed_failure_warning': self._embed_failures >= self._max_failures
        }


# ==================== convenience test ====================

def test_xinference_client(base_url: str = None) -> bool:
    """
    Test the XInference connection. Returns True if all components respond.
    """
    try:
        from config import config
        base_url = base_url or config.XINFERENCE_BASE_URL
    except ImportError:
        base_url = base_url or "http://localhost:9997"

    client = XInferenceClient(base_url)
    all_ok = True

    # 1. health check
    if client.health_check():
        print(f"  [ok] service reachable: {base_url}")
        models = client.get_available_models()
        if models:
            print(f"  [ok] deployed models: {', '.join(models)}")
    else:
        print(f"  [x] service unreachable: {base_url}")
        return False

    # 2. embedding test
    try:
        embs = client.embed(["hello world"])
        assert len(embs) == 1 and len(embs[0]) > 0
        print(f"  [ok] embedding works (dim: {len(embs[0])})")
    except Exception as e:
        print(f"  [x] embedding failed: {e}")
        all_ok = False

    # 3. LLM test
    try:
        answer = client.chat(
            messages=[{"role": "user", "content": "Reply with just the word: OK"}],
            max_tokens=10
        )
        print(f"  [ok] LLM works (answer: {answer.strip()[:30]})")
    except Exception as e:
        print(f"  [x] LLM failed: {e}")
        all_ok = False

    # 4. rerank test (failure is non-fatal; degraded path is used)
    try:
        results = client.rerank("test query", ["doc1", "doc2", "doc3"], top_k=2)
        assert len(results) == 2
        print(f"  [ok] rerank works (returned {len(results)})")
    except Exception as e:
        print(f"  [warn] rerank error (degraded strategy will be used): {e}")

    return all_ok
