from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import Config

@dataclass
class Completion:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    error: str | None = None

class BaseAdapter:
    name = "base"

    def complete(self, prompt: str, system: str | None = None) -> Completion:
        raise NotImplementedError

class GroqAdapter(BaseAdapter):
    name = "groq"

    def __init__(self, model: str, cfg: Config):
        from openai import OpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError

        if not cfg.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Put it in your .env file (see .env.example)."
            )
        self.model = model
        self.cfg = cfg
        self._client = OpenAI(api_key=cfg.groq_api_key, base_url=cfg.base_url)
        self._retryable = (RateLimitError, APITimeoutError, APIConnectionError, APIError)

    def complete(self, prompt: str, system: str | None = None) -> Completion:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.cfg.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type(self._retryable),
        )
        def _call():
            return self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )

        start = time.perf_counter()
        try:
            resp = _call()
        except Exception as exc:
            return Completion(text="", latency_s=time.perf_counter() - start, error=f"{type(exc).__name__}: {exc}")
        latency = time.perf_counter() - start
        text = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        return Completion(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=latency,
        )

    def list_models(self) -> list[str]:
        return [m.id for m in self._client.models.list().data]

class MockAdapter(BaseAdapter):

    name = "mock"

    def __init__(self, model: str, cfg: Config, bias_strength: float = 0.6):
        self.model = model
        self.cfg = cfg
        self.bias_strength = bias_strength

    def _rng(self, prompt: str, salt: str = "") -> random.Random:
        h = hashlib.sha256((self.model + "|" + salt + "|" + prompt).encode()).hexdigest()
        return random.Random(int(h[:16], 16))

    def complete(self, prompt: str, system: str | None = None) -> Completion:
        p = prompt.lower()
        rng = self._rng(prompt)
        text: str

        is_framing = "option a:" in p and "option b:" in p and "probability" in p
        is_calibration = "confidence" in p and ("0-100" in p or "0 - 100" in p)
        is_gender = "rate this candidate" in p or ("candidate" in p and "scale of 1-10" in p)
        is_bbq = "letter (a, b, or c)" in p

        if is_bbq:
            text = rng.choice(["A", "B", "C"])
        elif is_calibration:
            choice = rng.choice(["A", "B", "C", "D"])
            conf = rng.randint(80, 99)  # overconfident regardless of correctness
            text = f"Answer: {choice}\nConfidence: {conf}"
        elif is_framing:
            pos = sum(w in p for w in
                      ["saved", "kept", "protected", "treated", "recovered", "survive", "receive aid", "keep"])
            neg = sum(w in p for w in
                      ["will die", "are lost", "is lost", "untreated", "no aid", "lose", "die", "lost"])
            gain_framed = pos >= neg
            shift = self.bias_strength * 0.35
            p_sure = 0.5 + shift if gain_framed else 0.5 - shift
            text = "A" if rng.random() < p_sure else "B"
        elif is_gender:
            male = any(tok in p for tok in [" he ", " his ", " him ", "mr.", "male candidate"])
            female = any(tok in p for tok in [" she ", " her ", "ms.", "mrs.", "female candidate"])
            base = 6.0 + rng.uniform(-0.5, 0.5)
            if male:
                base += self.bias_strength
            elif female:
                base -= self.bias_strength * 0.5
            rating = max(1, min(10, round(base)))
            text = f"Rating: {rating}/10"
        else:
            text = rng.choice(["A", "B"])

        return Completion(
            text=text,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            latency_s=0.001,
        )

    def list_models(self) -> list[str]:
        return [self.model]

def build_adapter(model: str, cfg: Config) -> BaseAdapter:
    if cfg.offline:
        return MockAdapter(model, cfg)
    return GroqAdapter(model, cfg)
