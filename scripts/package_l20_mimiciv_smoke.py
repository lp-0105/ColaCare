from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_smoke(processed_dir: Path, output_dir: Path) -> dict[str, Any]:
    required_roots = [
        processed_dir / "mimic4_formatted_ehr.parquet",
        processed_dir / "formatted_ehr_summary.json",
    ]
    fold_files = sorted(path for path in (processed_dir / "fold_1").iterdir() if path.is_file())
    inputs = [*required_roots, *fold_files]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing smoke package inputs: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "L20_SMOKE_PACKAGE_MANIFEST.json"
    manifest = {
        "purpose": "controlled L20 environment smoke only; MIMIC-derived data; never commit publicly",
        "cohort": "600 pseudonymized ICU episodes",
        "files": [
            {
                "path": str(path.relative_to(processed_dir)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in inputs
        ],
    }
    manifest["total_input_bytes"] = sum(item["bytes"] for item in manifest["files"])
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    archive_path = output_dir / "l20-mimiciv-smoke-600.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(manifest_path, arcname="mimiciv-smoke-600/L20_SMOKE_PACKAGE_MANIFEST.json")
        for path in inputs:
            relative = path.relative_to(processed_dir)
            archive.add(path, arcname=(Path("mimiciv-smoke-600") / relative).as_posix())
    archive_hash = sha256_file(archive_path)
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    checksum_path.write_text(f"{archive_hash}  {archive_path.name}\n", encoding="ascii")
    return {
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_hash,
        "checksum": str(checksum_path),
        "manifest": str(manifest_path),
        "file_count": len(inputs),
        "input_bytes": manifest["total_input_bytes"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the 600-episode L20 MIMIC-IV smoke data.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("ehr_datasets/mimic-iv/processed"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ehr_datasets/mimic-iv/processed/l20_smoke_package"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = package_smoke(args.processed_dir, args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
