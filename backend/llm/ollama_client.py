"""
ollama_client.py
Local LLM — Ollama

Wraps the Ollama HTTP API.
Ollama must be running locally on http://localhost:11434.

Supports:
  - generate()   : single-turn generation
  - chat()       : multi-turn messages
  - list_models(): available local models
"""

from __future__ import annotations

import json
import logging
from typing import Iterator, List, Optional

import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE = "http://localhost:11434"
# 3B fits fully in a 4GB GPU 100% GPU, ~5-7x faster than
# llama3:8b which spilled ~56% to CPU. See project_gpu_setup memory.
DEFAULT_MODEL = "llama3.2:3b"


class OllamaClient:

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE,
        temperature: float = 0.2,
        # 2048 keeps llama3.2:3b 100% on the 4GB GPU (4096 forced ~20% onto CPU).
        # Real prompts are ~1275 tok + ~300 output; bump to 3072/4096 only if you
        # hit context truncation (cost: the LLM spills partly back to CPU).
        num_ctx: int = 2048,
        keep_alive: str = "10m",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        # Keep the model resident in VRAM between queries so the ~1.3s reload is
        # paid once, not on every question (big win for the interactive loop).
        self.keep_alive = keep_alive
        self._check_connection()

    # ---------------------------------------------------------------- #

    def _check_connection(self) -> None:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            logger.info("Ollama connected. Available models: %s", models)
            if self.model not in models:
                logger.warning(
                    "Model '%s' not found locally. "
                    "Run: ollama pull %s",
                    self.model, self.model,
                )
        except Exception as exc:
            logger.warning(
                "Ollama not reachable at %s — %s. "
                "Ensure Ollama is running.",
                self.base_url, exc,
            )

    # ---------------------------------------------------------------- #

    def generate(
        self,
        prompt: str,
        stream: bool = False,
    ) -> str:
        payload = {
            "model": self.model,
            "keep_alive": self.keep_alive,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=300,
            stream=stream,
        )
        resp.raise_for_status()

        if stream:
            return self._collect_stream(resp)

        data = resp.json()
        return data.get("response", "").strip()

    def generate_stream(self, prompt: str) -> Iterator[str]:
        payload = {
            "model": self.model,
            "keep_alive": self.keep_alive,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }
        with requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=300,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break

    # ---------------------------------------------------------------- #

    def chat(
        self,
        messages: List[dict],
        stream: bool = False,
    ) -> str:
        """
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
        """
        payload = {
            "model": self.model,
            "keep_alive": self.keep_alive,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=300,
            stream=stream,
        )
        resp.raise_for_status()

        if stream:
            return self._collect_chat_stream(resp)

        data = resp.json()
        return data.get("message", {}).get("content", "").strip()

    def chat_stream(self, messages: List[dict]) -> Iterator[str]:
        """
        Streaming variant of chat() — yields content tokens as they arrive.

        Uses the same /api/chat endpoint and message format as chat(), so a
        streamed answer is generated from an identical prompt to the
        non-streamed one (unlike generate_stream, which hits /api/generate).
        """
        payload = {
            "model": self.model,
            "keep_alive": self.keep_alive,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }
        with requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=300,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break

    def vlm_stream(
        self,
        prompt: str,
        image_paths: List[str],
        num_predict: int = 350,
    ) -> Iterator[str]:
        """Stream a vision-model response for prompt + image(s).

        Encodes the image files as base64 and streams from /api/generate with
        the `images` field (Ollama VLM API). Used with a VLM model such as
        qwen2.5vl:3b to interpret a retrieved figure. `num_predict` hard-caps the
        output length — VLMs tend to ignore "be concise" and ramble otherwise.
        """
        import base64

        images: List[str] = []
        for p in image_paths:
            try:
                with open(p, "rb") as f:
                    images.append(base64.b64encode(f.read()).decode())
            except OSError as exc:
                logger.warning("VLM: could not read image %s — %s", p, exc)

        if not images:
            yield "[no readable image to interpret]"
            return

        payload = {
            "model": self.model,
            "keep_alive": self.keep_alive,
            "prompt": prompt,
            "images": images,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": num_predict,
            },
        }
        with requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=600,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break

    # ---------------------------------------------------------------- #

    @staticmethod
    def _collect_stream(resp: requests.Response) -> str:
        tokens: List[str] = []
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                tokens.append(data.get("response", ""))
                if data.get("done"):
                    break
        return "".join(tokens).strip()

    @staticmethod
    def _collect_chat_stream(resp: requests.Response) -> str:
        tokens: List[str] = []
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                tokens.append(data.get("message", {}).get("content", ""))
                if data.get("done"):
                    break
        return "".join(tokens).strip()

    # ---------------------------------------------------------------- #

    def list_models(self) -> List[str]:
        resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
