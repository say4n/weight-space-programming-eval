"""Model-agnostic completion backends.

Two backends cover the realistic cases:
  PiRuntime     any provider/model registered in pi. Local models work too, via a
                pi custom provider that points at an OpenAI-compatible server.
  OpenAIRuntime any /chat/completions endpoint: vLLM, Ollama, LM Studio, OpenRouter, TGI.

Both expose the same stateful interface, so an experiment never knows which
model it is talking to: `send(text) -> Reply`.
"""

import http.client
import json
import subprocess
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

# The original harness used exactly one neutral system line for every run.
SYSTEM = "Answer the user's request."


@dataclass
class Reply:
    text: str
    usage: dict = field(default_factory=dict)
    tool_calls: int = 0
    wall_s: float = 0.0

    @property
    def tool_free(self):
        return self.tool_calls == 0


def extract_reply(stdout):
    """Pull the final assistant text and usage out of pi's --mode json stream."""
    text, usage, tools = "", {}, 0
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") != "message_end":
            continue
        msg = ev.get("message", {})
        if msg.get("role") != "assistant":
            continue
        blocks = msg.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
        usage = msg.get("usage") or {}
        tools += sum(
            1
            for b in blocks
            if isinstance(b, dict) and b.get("type") in ("toolCall", "tool_call", "tool_use")
        )
    return text, {k: usage.get(k) for k in ("input", "output", "reasoning", "totalTokens")}, tools


def _post_json(url, body, api_key, timeout):
    """POST to an http(s) URL. The scheme is checked here, not at the caller."""
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError(f"refusing non-http(s) URL: {url!r}")
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    conn = cls(u.hostname, u.port, timeout=timeout)
    path = (u.path or "/") + (f"?{u.query}" if u.query else "")
    try:
        conn.request(
            "POST",
            path,
            body=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


class Runtime:
    """One conversation. send() appends a user turn and returns the reply."""

    name = "runtime"

    def send(self, text):
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class PiRuntime(Runtime):
    """A tool-free pi conversation. Continuation reuses the session, so the model
    keeps its context without us resending the transcript each turn."""

    def __init__(
        self,
        provider,
        model,
        thinking="high",
        cwd=None,
        timeout=7200,
        session_dir=None,
        system=SYSTEM,
    ):
        self.provider, self.model = provider, model
        self.thinking, self.cwd, self.timeout = thinking, cwd, timeout
        self.system = system
        self.sid = None
        self.session_dir = Path(session_dir) if session_dir else None
        self.name = f"{provider}/{model}"
        self.turns = 0

    def send(self, text):
        cmd = [
            "pi",
            "-p",
            "--no-tools",
            "-ne",
            "-ns",
            "-np",
            "-nc",
            "--provider",
            self.provider,
            "--model",
            self.model,
            "--thinking",
            self.thinking,
            "--system-prompt",
            self.system,
            "--mode",
            "json",
        ]
        if self.sid:
            # continue the same conversation
            cmd += ["--session-id", self.sid]
        else:
            cmd += ["--no-session"]
        if self.session_dir:
            cmd += ["--session-dir", str(self.session_dir)]

        t0 = time.time()
        p = subprocess.run(
            cmd, input=text, capture_output=True, text=True, timeout=self.timeout, cwd=self.cwd
        )
        wall = time.time() - t0
        if self.sid is None:
            self.sid = _session_id(p.stdout)
        self.turns += 1
        reply_text, usage, tools = extract_reply(p.stdout)
        if not reply_text:
            # An empty reply is a harness/API failure, never a model answer. Say so
            # loudly: scoring silence as a wrong answer corrupts results silently.
            detail = (p.stderr or p.stdout or "")[-600:].strip()
            reply_text = (
                f"[harness error: empty reply, rc={p.returncode}, {wall:.1f}s]\n{detail}"
            )
        return Reply(reply_text, usage, tools, round(wall, 1))

    def close(self):
        pass


def _session_id(stdout):
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "session":
            return ev.get("id")
    return None


class OpenAIRuntime(Runtime):
    """Any OpenAI-compatible /chat/completions server. Native multi-turn."""

    def __init__(
        self,
        base_url,
        model,
        api_key=None,
        thinking=None,
        timeout=7200,
        temperature=1.0,
        max_tokens=None,
        system=SYSTEM,
    ):
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(f"base_url must be http(s), got {base_url!r}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or "not-needed"
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system = system
        self.messages = []
        self.name = model

    def send(self, text):
        msgs = [{"role": "system", "content": self.system}] if self.system else []
        msgs = msgs + self.messages + [{"role": "user", "content": text}]
        body = {"model": self.model, "messages": msgs, "temperature": self.temperature}
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        req_body = json.dumps(body).encode()
        t0 = time.time()
        try:
            status, raw = _post_json(
                f"{self.base_url}/chat/completions", req_body, self.api_key, self.timeout
            )
            if status != 200:
                return Reply(
                    f"[harness error: HTTP {status}] {raw[:500]!r}",
                    {},
                    0,
                    round(time.time() - t0, 1),
                )
            out = json.loads(raw)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            return Reply(f"[harness error: {e}]", {}, 0, round(time.time() - t0, 1))
        wall = time.time() - t0
        text_out = out["choices"][0]["message"].get("content") or ""
        u = out.get("usage") or {}
        usage = {
            "input": u.get("prompt_tokens"),
            "output": u.get("completion_tokens"),
            "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "totalTokens": u.get("total_tokens"),
        }
        self.messages += [
            {"role": "user", "content": text},
            {"role": "assistant", "content": text_out},
        ]
        return Reply(text_out, usage, 0, round(wall, 1))


def make_runtime(model, provider=None, thinking="high", base_url=None, cwd=None, **kw):
    """Convenience: provider/model -> PiRuntime, base_url/model -> OpenAIRuntime."""
    if base_url:
        return OpenAIRuntime(base_url, model, **kw)
    return PiRuntime(provider or "opencode-go", model, thinking=thinking, cwd=cwd)
