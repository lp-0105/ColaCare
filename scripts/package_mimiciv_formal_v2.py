from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


ARCHIVE_ROOT = "mimiciv_formal_v2"
EXCLUDED_NAMES = {
    "smoke_results",
    "results",
    "__pycache__",
}
EXCLUDED_SUFFIXES = {
    ".ckpt",
    ".pt",
    ".pth",
    ".sqlite",
    ".sqlite3",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def package_files(data_root: Path) -> list[Path]:
    required = [
        data_root / "mimic4_formatted_ehr.parquet",
        data_root / "cohort_summary.json",
        data_root / "feature_mapping.json",
        data_root / "formal_raw_to_parquet.log",
        data_root / "fold_1" / "validation_report.json",
        data_root / "fold_1" / "preprocess_summary.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"formal-v2 required files missing: {missing}")
    files = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(data_root)
        if any(part in EXCLUDED_NAMES for part in relative.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        files.append(path)
    return files


def validate_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member {name!r}")
    if not path.parts or path.parts[0] != ARCHIVE_ROOT:
        raise ValueError(f"archive member outside {ARCHIVE_ROOT}: {name!r}")


def verify_restored(
    archive_path: Path,
    manifest: dict[str, Any],
    restore_root: Path,
) -> dict[str, Any]:
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            validate_member_name(member.name)
        archive.extractall(restore_root, filter="data")
    restored_root = restore_root / ARCHIVE_ROOT
    failures = []
    restored_bytes = 0
    for item in manifest["files"]:
        path = restored_root / item["path"]
        if not path.is_file():
            failures.append({"path": item["path"], "status": "MISSING"})
            continue
        restored_bytes += path.stat().st_size
        actual = sha256_file(path)
        if actual != item["sha256"]:
            failures.append(
                {
                    "path": item["path"],
                    "status": "FAIL",
                    "expected": item["sha256"],
                    "actual": actual,
                }
            )
    return {
        "status": "PASS" if not failures else "FAIL",
        "checked_files": len(manifest["files"]),
        "restored_bytes": restored_bytes,
        "failures": failures,
    }


def package(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "l20-mimiciv-formal-v2.tar.gz"
    checksum_path = output_dir / "l20-mimiciv-formal-v2.tar.gz.sha256"
    manifest_path = output_dir / "formal_v2_package_manifest.json"
    for path in (archive_path, checksum_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    files = package_files(data_root)
    validation = json.loads(
        (data_root / "fold_1" / "validation_report.json").read_text(
            encoding="utf-8"
        )
    )
    if validation.get("status") != "PASS":
        raise ValueError("formal-v2 validation report is not PASS")
    tensor_contract = validation.get("categorical_tensor_contract", {})
    if (
        tensor_contract.get("status") != "PASS"
        or tensor_contract.get("missing_one_hot_violations") != 0
        or tensor_contract.get("actual_all_zero_fixed_level_count") != 18
    ):
        raise ValueError("formal-v2 categorical tensor gate failed")

    entries = []
    for path in files:
        relative = path.relative_to(data_root).as_posix()
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    data_decision = {
        "data_version": "formal_v2",
        "source_data_version": "formal_v1 formatted parquet",
        "source_parquet_sha256": sha256_file(
            data_root / "mimic4_formatted_ehr.parquet"
        ),
        "code_commit": args.code_commit,
        "decision": "regenerated pickles without stringifying missing GCS values",
        "raw_event_rescan": False,
        "cohort_changed": False,
        "labels_changed": False,
        "split_changed": False,
        "feature_dimension": 59,
        "all_zero_compatibility_columns": 18,
        "missing_one_hot_violations": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest = {
        "format_version": 1,
        "archive_root": ARCHIVE_ROOT,
        "data_decision": data_decision,
        "file_count": len(entries),
        "total_uncompressed_bytes": sum(item["bytes"] for item in entries),
        "files": entries,
        "contains_raw_mimic": False,
        "contains_patient_text": False,
    }

    with tempfile.TemporaryDirectory(prefix="colacare-formal-v2-package-") as tmp:
        temporary = Path(tmp)
        decision_path = temporary / "data_decision.json"
        internal_manifest_path = temporary / "formal_v2_package_manifest.json"
        write_json(decision_path, data_decision)
        entries_with_metadata = [
            *entries,
            {
                "path": "data_decision.json",
                "bytes": decision_path.stat().st_size,
                "sha256": sha256_file(decision_path),
            },
        ]
        internal_manifest = {
            **manifest,
            "file_count": len(entries_with_metadata),
            "total_uncompressed_bytes": sum(
                item["bytes"] for item in entries_with_metadata
            ),
            "files": entries_with_metadata,
        }
        write_json(internal_manifest_path, internal_manifest)
        with tarfile.open(archive_path, "w:gz") as archive:
            for path in files:
                relative = path.relative_to(data_root).as_posix()
                archive.add(
                    path,
                    arcname=f"{ARCHIVE_ROOT}/{relative}",
                    recursive=False,
                )
            archive.add(
                decision_path,
                arcname=f"{ARCHIVE_ROOT}/data_decision.json",
                recursive=False,
            )
            archive.add(
                internal_manifest_path,
                arcname=f"{ARCHIVE_ROOT}/formal_v2_package_manifest.json",
                recursive=False,
            )
        with tempfile.TemporaryDirectory(
            prefix="colacare-formal-v2-restore-"
        ) as restore_tmp:
            verification = verify_restored(
                archive_path, internal_manifest, Path(restore_tmp)
            )
        if verification["status"] != "PASS":
            raise RuntimeError(f"local restore verification failed: {verification}")
        outer_sha256 = sha256_file(archive_path)
        checksum_path.write_text(
            f"{outer_sha256}  {archive_path.name}\n", encoding="utf-8"
        )
        external_manifest = {
            **internal_manifest,
            "archive": {
                "name": archive_path.name,
                "bytes": archive_path.stat().st_size,
                "sha256": outer_sha256,
            },
            "local_restore_verification": verification,
        }
        write_json(manifest_path, external_manifest)
    return {
        "status": "PASS",
        "archive": str(archive_path),
        "checksum": str(checksum_path),
        "manifest": str(manifest_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "files": len(external_manifest["files"]),
        "local_restore_verification": verification,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package validated formal-v2 MIMIC-IV processed data for L20."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def main() -> int:
    report = package(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
