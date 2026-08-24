import os
import time
from collections import deque
from threading import Lock
from langchain_openai import ChatOpenAI
from openai import (
    RateLimitError,
    InternalServerError,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,   # catches everything else with an HTTP status: 402, 403, 400, etc.
)
from dotenv import load_dotenv
load_dotenv()

# ── Config ─────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Gemini via Google's OpenAI-compatible endpoint — get a key at aistudio.google.com/apikey
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

GROQ_MODEL   = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-2.5-flash"  # verify current free-tier model/limits at ai.google.dev/pricing
NVIDIA_MODEL = "nvidia/nvidia-nemotron-nano-9b-v2"

# Free-tier RPM ceilings — tune these to whatever your actual account tier gives you.
GROQ_RPM   = 20
GEMINI_RPM = 10   # conservative default for Flash free tier — check your actual quota in AI Studio
NVIDIA_RPM = 40

TRANSIENT_ERRORS = (
    RateLimitError,
    InternalServerError,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,  # broad catch: 402 payment_required, 403 permission_denied, 400 bad_request, etc.
                      # A dead/broke/misconfigured provider should never kill the whole chain —
                      # it should just get skipped in favor of the next one.
)


# ── Rate limiter ───────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, rpm: int, name: str = ""):
        self.rpm = rpm
        self.name = name
        self.calls = deque()
        self.lock = Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            while self.calls and now - self.calls[0] > 60:
                self.calls.popleft()

            if len(self.calls) >= self.rpm:
                sleep_for = 60 - (now - self.calls[0]) + 0.05
                if sleep_for > 0:
                    print(f"[RateLimiter:{self.name}] at {self.rpm}/min cap — waiting {sleep_for:.1f}s")
                    time.sleep(sleep_for)
                now = time.monotonic()
                while self.calls and now - self.calls[0] > 60:
                    self.calls.popleft()

            self.calls.append(time.monotonic())


groq_limiter   = RateLimiter(GROQ_RPM, "Groq")
gemini_limiter = RateLimiter(GEMINI_RPM, "Gemini")
nvidia_limiter = RateLimiter(NVIDIA_RPM, "NVIDIA")


# ── LLM wrapper ────────────────────────────────────────────────────
def _build_llm(model: str, api_key: str, base_url: str):
    return ChatOpenAI(
        model=model,
        temperature=0.9,
        api_key=api_key,
        base_url=base_url,
        max_retries=1,
        request_timeout=60
    )


class SmartLLM:
    """
    Tries providers in order, falling back on transient errors:
    Groq (primary) -> Gemini (secondary) -> NVIDIA (tertiary / last resort)
    """
    def __init__(self, groq, gemini, nvidia):
        # Each entry: (llm, limiter, name)
        self.chain = [
            (groq,   groq_limiter,   "Groq"),
            (gemini, gemini_limiter, "Gemini"),
            (nvidia, nvidia_limiter, "NVIDIA"),
        ]

    def invoke(self, prompt):
        last_exc = None
        for i, (llm, limiter, name) in enumerate(self.chain):
            is_last = (i == len(self.chain) - 1)
            limiter.wait()
            try:
                return llm.invoke(prompt)
            except TRANSIENT_ERRORS as e:
                last_exc = e
                detail = getattr(e, "message", None) or str(e)
                status = getattr(e, "status_code", "?")
                print(f"[{name}] {type(e).__name__} (status={status}) — {detail[:120]}")
                if is_last:
                    print("[SmartLLM] all providers exhausted for this call")
                    raise
                print(f"[SmartLLM] falling back: {name} -> {self.chain[i + 1][2]}")
        # Should never reach here, but just in case
        if last_exc:
            raise last_exc


def get_llm(output_schema=None):
    groq_llm   = _build_llm(GROQ_MODEL, GROQ_API_KEY, GROQ_BASE_URL)
    gemini_llm = _build_llm(GEMINI_MODEL, GEMINI_API_KEY, GEMINI_BASE_URL)
    nvidia_llm = _build_llm(NVIDIA_MODEL, NVIDIA_API_KEY, NVIDIA_BASE_URL)

    if output_schema:
        groq_llm   = groq_llm.with_structured_output(output_schema)
        gemini_llm = gemini_llm.with_structured_output(output_schema)
        nvidia_llm = nvidia_llm.with_structured_output(output_schema)

    return SmartLLM(groq=groq_llm, gemini=gemini_llm, nvidia=nvidia_llm)


# ── Example usage ──────────────────────────────────────────────────
if __name__ == "__main__":
    llm = get_llm()
    result = llm.invoke("Generate one reasoning-style dataset sample in the given schema.")
    print(result)