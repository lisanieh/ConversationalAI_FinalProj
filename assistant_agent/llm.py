"""LLM client for the agent (Llama-3.1-8B-Instruct by default).

Any OpenAI-compatible chat endpoint works, the same as the customer brain:
  vLLM on PACE:  vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype float16 \\
                     --max-model-len 8192 --gpu-memory-utilization 0.4 --port 8001
  Ollama:        ollama pull llama3.1:8b   ->  http://localhost:11434/v1

The customer runs on Qwen and the agent on Llama, so the agent never
"understands itself" (one model talking to itself makes extraction look easier
than it is with real people). Temperature 0 and a fixed seed make reruns exact.
"""

from __future__ import annotations

import json
import re
import urllib.request


class ChatLLM:
    def __init__(self, base_url: str = "http://localhost:8001/v1",
                 model: str = "meta-llama/Llama-3.1-8B-Instruct",
                 temperature: float = 0.0, seed: int = 0,
                 api_key: str = "EMPTY", timeout: float = 120):
        self.base_url = base_url.rstrip("/")
        self.model, self.temperature, self.seed = model, temperature, seed
        self.api_key, self.timeout = api_key, timeout
        self.calls = 0

    def chat(self, messages: list[dict], max_tokens: int = 200) -> str:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature,
                   "seed": self.seed, "max_tokens": max_tokens}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read())
        self.calls += 1
        return body["choices"][0]["message"]["content"]


def parse_json(text: str, default: dict | None = None) -> dict:
    """Small LLMs wrap JSON in prose or code fences: take the first {...} block."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            out = json.loads(m.group(0))
            if isinstance(out, dict):
                return out
        except json.JSONDecodeError:
            pass
    return {} if default is None else default
