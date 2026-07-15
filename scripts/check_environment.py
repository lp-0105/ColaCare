"""Print a non-secret environment report for ColaCare reproduction."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
LLM_ENV_NAMES = (
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL_NAME",
    "LLM_MAX_TOKENS",
    "LLM_CONTEXT_LENGTH",
    "LLM_TEMPERATURE",
    "LLM_REASONING_EFFORT",
    "LLM_TIMEOUT_SECONDS",
)


def redacted_llm_environment(environment: Mapping[str, str]) -> dict[str, str]:
    result = {}
    for name in LLM_ENV_NAMES:
        if name == "LLM_API_KEY":
            result[name] = "<redacted:set>" if environment.get(name) else "<redacted:unset>"
        else:
            result[name] = environment.get(name, "<unset>")
    return result


def _run(arguments: list[str], cwd: Path = ROOT) -> dict[str, object]:
    try:
        completed = subprocess.run(
            arguments,
            cwd=cwd,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": type(exc).__name__}
    stdout = completed.stdout.replace("\x00", "").strip()
    stderr = completed.stderr.replace("\x00", "").strip()
    result: dict[str, object] = {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
    }
    if stdout:
        result["stdout"] = stdout
    if stderr:
        result["stderr"] = stderr
    return result


def _torch_report() -> dict[str, object]:
    if importlib.util.find_spec("torch") is None:
        return {"installed": False}
    import torch

    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }


def build_environment_report() -> dict[str, object]:
    return {
        "repository": {
            "branch": _run(["git", "branch", "--show-current"]),
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "remotes": _run(["git", "remote", "-v"]),
        },
        "host": {
            "os": platform.platform(),
            "python": sys.version.replace("\n", " "),
            "executable": sys.executable,
            "torch": _torch_report(),
            "gpu": _run([
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]),
            "ollama_version": _run(["ollama", "--version"]),
            "ollama_models": _run(["ollama", "list"]),
        },
        "llm_environment": redacted_llm_environment(os.environ),
    }


def main() -> int:
    print(json.dumps(build_environment_report(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
