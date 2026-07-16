"""Download and integrity-check one pinned Hugging Face model snapshot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable


DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
GENERATED_FILES = {"MODEL_MANIFEST.json", "SHA256SUMS"}
TOKENIZER_PAYLOADS = {
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
}


class SnapshotValidationError(RuntimeError):
    """Raised when a downloaded snapshot is incomplete or internally inconsistent."""


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotValidationError(f"invalid {description}: {path.name}") from exc
    if not isinstance(value, dict):
        raise SnapshotValidationError(f"{description} must contain a JSON object: {path.name}")
    return value


def _has_chat_template(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(
            isinstance(item, dict)
            and isinstance(item.get("template"), str)
            and item["template"].strip()
            for item in value
        )
    return False


def validate_snapshot(root: Path) -> dict[str, Any]:
    """Validate critical Transformers files without importing Transformers."""

    root = root.resolve()
    if not root.is_dir():
        raise SnapshotValidationError(f"snapshot directory does not exist: {root}")

    config_path = root / "config.json"
    tokenizer_config_path = root / "tokenizer_config.json"
    for path in (config_path, tokenizer_config_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise SnapshotValidationError(f"required snapshot file is missing or empty: {path.name}")

    config = _read_json(config_path, "model config")
    tokenizer_config = _read_json(tokenizer_config_path, "tokenizer config")
    if not any((root / name).is_file() and (root / name).stat().st_size > 0 for name in TOKENIZER_PAYLOADS):
        raise SnapshotValidationError(
            "tokenizer payload is missing; expected tokenizer.json, tokenizer.model, or vocabulary files"
        )

    weights = sorted(path for path in root.glob("*.safetensors") if path.is_file())
    if not weights:
        raise SnapshotValidationError("no safetensors weight files found")
    empty_weights = [path.name for path in weights if path.stat().st_size == 0]
    if empty_weights:
        raise SnapshotValidationError(f"empty safetensors weight files: {', '.join(empty_weights)}")

    index_path = root / "model.safetensors.index.json"
    indexed_weights: list[str] = []
    if index_path.exists():
        index = _read_json(index_path, "safetensors index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise SnapshotValidationError("safetensors index has no non-empty weight_map")
        indexed_weights = sorted({name for name in weight_map.values() if isinstance(name, str)})
        missing = [name for name in indexed_weights if not (root / name).is_file()]
        if missing:
            raise SnapshotValidationError(
                f"safetensors index references missing files: {', '.join(missing)}"
            )
    elif len(weights) > 1:
        raise SnapshotValidationError("multiple safetensors shards require model.safetensors.index.json")

    template_file = root / "chat_template.jinja"
    if template_file.is_file() and template_file.stat().st_size > 0:
        template_source = template_file.name
    elif _has_chat_template(tokenizer_config.get("chat_template")):
        template_source = tokenizer_config_path.name
    else:
        raise SnapshotValidationError(
            "chat template is missing from tokenizer_config.json and chat_template.jinja"
        )

    return {
        "model_type": config.get("model_type"),
        "weight_files": [path.name for path in weights],
        "weight_index": index_path.name if index_path.is_file() else None,
        "indexed_weight_files": indexed_weights,
        "chat_template_source": template_source,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.name in GENERATED_FILES or ".cache" in relative.parts:
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_sha256sums(root: Path, files: list[Path]) -> None:
    lines = [f"{_sha256(path)}  {path.relative_to(root).as_posix()}" for path in files]
    temporary = root / "SHA256SUMS.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(root / "SHA256SUMS")


def _default_snapshot_downloader(**kwargs: Any) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required on the online preparation machine"
        ) from exc
    return snapshot_download(**kwargs)


def prepare_snapshot(
    *,
    model_id: str,
    revision: str,
    output_dir: Path,
    snapshot_downloader: Callable[..., str] | None = None,
) -> dict[str, Any]:
    if not model_id.strip():
        raise SnapshotValidationError("model ID must not be empty")
    if not revision.strip():
        raise SnapshotValidationError("revision must not be empty")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in GENERATED_FILES:
        generated = output_dir / name
        if generated.exists():
            generated.unlink()

    downloader = snapshot_downloader or _default_snapshot_downloader
    downloader(repo_id=model_id, revision=revision, local_dir=str(output_dir))
    validation = validate_snapshot(output_dir)

    payload_files = _payload_files(output_dir)
    manifest_files = [
        {
            "path": path.relative_to(output_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in payload_files
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "model_id": model_id,
        "revision": revision,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_size_bytes": sum(item["size_bytes"] for item in manifest_files),
        "validation": validation,
        "files": manifest_files,
    }
    manifest_path = output_dir / "MODEL_MANIFEST.json"
    _write_json_atomic(manifest_path, manifest)
    _write_sha256sums(output_dir, payload_files + [manifest_path])
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = prepare_snapshot(
            model_id=args.model_id,
            revision=args.revision,
            output_dir=args.output_dir,
        )
    except (SnapshotValidationError, RuntimeError, OSError) as exc:
        print(f"Snapshot preparation failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "success",
                "model_id": manifest["model_id"],
                "revision": manifest["revision"],
                "total_size_bytes": manifest["total_size_bytes"],
                "manifest": str(args.output_dir / "MODEL_MANIFEST.json"),
                "sha256sums": str(args.output_dir / "SHA256SUMS"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

