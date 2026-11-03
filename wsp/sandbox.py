"""Run model-written Python with no network and no access to the answer keys.

The original harness used bubblewrap on Linux. That does not exist on macOS, so
this uses `sandbox-exec` profiles instead, and falls back to an unsandboxed
subprocess (recorded as `network_isolated=False`) where neither is available.

Isolation is not decoration here. The repository holds the graded answer keys, the
code that derives them, and earlier runs; an experiment is meaningless if the model
can read any of it. So model code may read only its own workdir and the virtualenv.
"""

import json
import os
import resource
import shutil
import subprocess
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(REPO, "upstream")
_TMP = tempfile.gettempdir()

MAC_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny file-read* (subpath "{repo}"))
(deny file-write* (subpath "{repo}"))
(allow file-read* (subpath "{venv}"))
(allow file-read* (subpath "{cases}"))
(allow file-read* (subpath "{workdir}"))
(allow file-write* (subpath "{workdir}"))
"""


def _limits(mem_bytes):
    def apply():
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            resource.setrlimit(resource.RLIMIT_CPU, (1800, 1800))
        except (ValueError, OSError):
            pass  # macOS refuses some limits; the wall timeout still applies

    return apply


def _makedirs(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"cannot create {path}: {e}") from e


def _remove(path):
    try:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.unlink(path)
    except OSError:
        pass  # a file we cannot remove is not worth failing the run over


def _open(path, mode, data=None):
    try:
        if data is None:
            with open(path, mode) as f:
                return f.read()
        with open(path, mode) as f:
            f.write(data)
        return None
    except OSError as e:
        raise RuntimeError(f"cannot {'read' if data is None else 'write'} {path}: {e}") from e


class Sandbox:
    def __init__(
        self, workdir, python=None, timeout=900, mem_gb=8, max_out=20000, fresh=False, keep=()
    ):
        self.workdir = os.path.abspath(workdir)
        _makedirs(self.workdir)
        self.python = python or os.path.join(REPO, ".venv", "bin", "python")
        if not os.path.exists(self.python):
            self.python = "python3"
        self.timeout, self.mem_gb, self.max_out = timeout, mem_gb, max_out
        # fresh=True enforces the original's "nothing is kept between runs"
        self.fresh, self.keep = fresh, set(keep)
        self.sandbox_exec = shutil.which("sandbox-exec")
        self.bwrap = shutil.which("bwrap")
        self.network_isolated = bool(self.sandbox_exec or self.bwrap)
        self.log = []

    def reset(self):
        """Clear everything except the preloaded inputs."""
        if not self.fresh:
            return
        for root, dirs, files in os.walk(self.workdir, topdown=False):
            for name in files:
                rel = os.path.relpath(os.path.join(root, name), self.workdir)
                if rel not in self.keep:
                    _remove(os.path.join(root, name))
            for name in dirs:
                _remove(os.path.join(root, name))

    def _cmd(self):
        if self.sandbox_exec:
            # Deny the whole repository, not just upstream/: it holds the graded answers,
            # the code that derives them, and prior runs. Only the virtualenv and this
            # run's own workdir stay readable.
            profile = MAC_PROFILE.format(
                repo=REPO,
                venv=os.path.join(REPO, ".venv"),
                cases=os.path.join(REPO, "results", "exp7", "cases"),
                workdir=self.workdir,
            )
            return ["sandbox-exec", "-p", profile, self.python]
        if self.bwrap:
            return [
                "bwrap",
                "--unshare-all",
                "--die-with-parent",
                "--clearenv",
                "--ro-bind",
                "/usr",
                "/usr",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                _TMP,
                "--bind",
                self.workdir,
                "/work",
                "--chdir",
                "/work",
                "--setenv",
                "HOME",
                _TMP,
                self.python,
            ]
        return [self.python]

    def env(self):
        """Nothing from the parent that carries credentials or harness state."""
        return {
            "PATH": "/usr/bin:/bin",
            "HOME": self.workdir,
            "LANG": "C.UTF-8",
            "OMP_NUM_THREADS": "8",
            "PYTHONHASHSEED": "0",
            "MPLBACKEND": "Agg",
            "TOKENIZERS_PARALLELISM": "false",
        }

    def run(self, code, prelude=""):
        src = (prelude + "\n" if prelude else "") + code
        t0 = time.time()
        try:
            p = subprocess.run(
                self._cmd() + ["-I", "-c", src],
                capture_output=True,
                text=True,
                cwd=self.workdir,
                env=self.env(),
                timeout=self.timeout,
                preexec_fn=_limits(self.mem_gb << 30),
            )
            out = p.stdout + (("\n[stderr]\n" + p.stderr[-6000:]) if p.stderr.strip() else "")
        except subprocess.TimeoutExpired:
            out = f"[timed out after {self.timeout} s]"
        dt = round(time.time() - t0, 2)
        if len(out) > self.max_out:
            out = out[: self.max_out] + f"\n[output cut at {self.max_out:,} characters]"
        self.log.append({"t": time.time(), "s": dt, "code": code, "out": out[:6000]})
        return out or "(no output)"

    def _path(self, name):
        """Resolve inside the workdir; model-chosen names must not escape it."""
        path = os.path.abspath(os.path.join(self.workdir, name))
        if path != self.workdir and not path.startswith(self.workdir + os.sep):
            raise ValueError(f"path escapes the sandbox workdir: {name!r}")
        return path

    def write(self, name, data):
        path = self._path(name)
        _makedirs(os.path.dirname(path) or self.workdir)
        _open(path, "wb" if isinstance(data, bytes) else "w", data)
        return path

    def read(self, name, mode="r"):
        return _open(self._path(name), mode)

    def save_log(self, path):
        _open(path, "w", "".join(json.dumps(row) + "\n" for row in self.log))
