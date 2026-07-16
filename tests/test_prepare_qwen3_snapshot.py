import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_qwen3_snapshot import (
    DEFAULT_MODEL_ID,
    SnapshotValidationError,
    prepare_snapshot,
    validate_snapshot,
)


PINNED_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def create_fake_snapshot(root: Path, *, include_chat_template: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "torch_dtype": "bfloat16"}),
        encoding="utf-8",
    )
    tokenizer_config = {}
    if include_chat_template:
        tokenizer_config["chat_template"] = "{% for message in messages %}{{ message.content }}{% endfor %}"
    (root / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config), encoding="utf-8"
    )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    shard = root / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"fake-safetensors-for-unit-test")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": shard.name}}),
        encoding="utf-8",
    )


class SnapshotValidationTests(unittest.TestCase):
    def test_complete_snapshot_with_embedded_chat_template_is_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_fake_snapshot(root)
            report = validate_snapshot(root)
            self.assertEqual(report["weight_files"], ["model-00001-of-00001.safetensors"])
            self.assertEqual(report["chat_template_source"], "tokenizer_config.json")

    def test_external_chat_template_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_fake_snapshot(root, include_chat_template=False)
            (root / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
            report = validate_snapshot(root)
            self.assertEqual(report["chat_template_source"], "chat_template.jinja")

    def test_missing_chat_template_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_fake_snapshot(root, include_chat_template=False)
            with self.assertRaisesRegex(SnapshotValidationError, "chat template"):
                validate_snapshot(root)

    def test_index_referencing_missing_weight_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_fake_snapshot(root)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"layer": "missing.safetensors"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SnapshotValidationError, "missing.safetensors"):
                validate_snapshot(root)


class SnapshotPreparationTests(unittest.TestCase):
    def test_prepare_downloads_exact_revision_and_writes_integrity_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "snapshot"
            captured = {}

            def fake_download(**kwargs):
                captured.update(kwargs)
                create_fake_snapshot(Path(kwargs["local_dir"]))
                return str(kwargs["local_dir"])

            manifest = prepare_snapshot(
                model_id=DEFAULT_MODEL_ID,
                revision=PINNED_REVISION,
                output_dir=output,
                snapshot_downloader=fake_download,
            )

            self.assertEqual(captured["repo_id"], DEFAULT_MODEL_ID)
            self.assertEqual(captured["revision"], PINNED_REVISION)
            self.assertEqual(Path(captured["local_dir"]), output.resolve())
            self.assertEqual(manifest["model_id"], DEFAULT_MODEL_ID)
            self.assertEqual(manifest["revision"], PINNED_REVISION)
            self.assertIn("downloaded_at_utc", manifest)
            self.assertGreater(manifest["total_size_bytes"], 0)

            manifest_path = output / "MODEL_MANIFEST.json"
            sums_path = output / "SHA256SUMS"
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(sums_path.is_file())
            stored = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(stored, manifest)

            sum_lines = sums_path.read_text(encoding="utf-8").splitlines()
            self.assertTrue(any(line.endswith("  MODEL_MANIFEST.json") for line in sum_lines))
            weight_line = next(
                line for line in sum_lines if line.endswith("  model-00001-of-00001.safetensors")
            )
            expected = hashlib.sha256(
                (output / "model-00001-of-00001.safetensors").read_bytes()
            ).hexdigest()
            self.assertEqual(weight_line.split()[0], expected)

    def test_prepare_does_not_write_manifest_when_snapshot_is_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "snapshot"

            def incomplete_download(**kwargs):
                Path(kwargs["local_dir"]).mkdir(parents=True, exist_ok=True)
                (Path(kwargs["local_dir"]) / "config.json").write_text("{}")
                return str(kwargs["local_dir"])

            with self.assertRaises(SnapshotValidationError):
                prepare_snapshot(
                    model_id=DEFAULT_MODEL_ID,
                    revision=PINNED_REVISION,
                    output_dir=output,
                    snapshot_downloader=incomplete_download,
                )
            self.assertFalse((output / "MODEL_MANIFEST.json").exists())
            self.assertFalse((output / "SHA256SUMS").exists())


if __name__ == "__main__":
    unittest.main()
