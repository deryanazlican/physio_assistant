from __future__ import annotations

import time
import requests
from google import genai
from together import Together


class BaseBackend:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str) -> tuple[str, float]:
        raise NotImplementedError


class GeminiBackend(BaseBackend):
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        super().__init__(model_name=model_name)
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str) -> tuple[str, float]:
        start = time.perf_counter()
        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        latency = time.perf_counter() - start
        text = (getattr(resp, "text", "") or "").strip()
        return text, latency


class TogetherBackend(BaseBackend):
    def __init__(self, api_key: str, model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"):
        super().__init__(model_name=model_name)
        self.client = Together(api_key=api_key)

    def generate(self, prompt: str) -> tuple[str, float]:
        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=220,
        )
        latency = time.perf_counter() - start
        text = (resp.choices[0].message.content or "").strip()
        return text, latency


class OllamaBackend(BaseBackend):
    def __init__(self, model_name: str = "llama3.1", base_url: str = "http://localhost:11434"):
        super().__init__(model_name=model_name)
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str) -> tuple[str, float]:
        start = time.perf_counter()

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )
        resp.raise_for_status()

        latency = time.perf_counter() - start
        data = resp.json()
        text = (data.get("response") or "").strip()
        return text, latency