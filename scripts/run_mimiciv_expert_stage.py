from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


EXPECTED_SPLITS = {
    "train": (21287, 2714),
    "val": (1183, 151),
    "test": (1183, 151),
}
MODEL_NAMES = ("RETAIN", "ConCare", "AdaCare")
INPUT_SHAPE = (48, 61)
LABEL_SHAPE = (48, 3)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class PeakRSSMonitor:
    def __init__(self, interval_seconds: float = 0.1) -> None:
        self.interval_seconds = interval_seconds
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "PeakRSSMonitor":
        process = psutil.Process()

        def poll() -> None:
            while not self._stop.is_set():
                self.peak_rss = max(self.peak_rss, int(process.memory_info().rss))
                self._stop.wait(self.interval_seconds)

        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.peak_rss = max(self.peak_rss, int(psutil.Process().memory_info().rss))


class FixedSequenceDataset(Dataset):
    def __init__(self, data_dir: Path, split: str) -> None:
        started = time.perf_counter()
        with (data_dir / f"{split}_x.pkl").open("rb") as handle:
            x_values = pickle.load(handle)
        with (data_dir / f"{split}_y.pkl").open("rb") as handle:
            y_values = pickle.load(handle)
        self.x = np.asarray(x_values, dtype=np.float32)
        y_array = np.asarray(y_values, dtype=np.float32)
        del x_values, y_values
        if self.x.ndim != 3 or tuple(self.x.shape[1:]) != INPUT_SHAPE:
            raise ValueError(f"{split} unexpected x shape {self.x.shape}")
        if y_array.ndim != 3 or tuple(y_array.shape[1:]) != LABEL_SHAPE:
            raise ValueError(f"{split} unexpected y shape {y_array.shape}")
        if len(self.x) != len(y_array):
            raise ValueError(f"{split} x/y length mismatch")
        self.y = y_array[:, -1, 0].astype(np.float32, copy=True)
        if not np.isfinite(self.x).all() or not np.isfinite(self.y).all():
            raise ValueError(f"{split} contains NaN/inf")
        if not set(np.unique(self.y).tolist()) <= {0.0, 1.0}:
            raise ValueError(f"{split} labels are not binary")
        expected_count, expected_positive = EXPECTED_SPLITS[split]
        if len(self.y) != expected_count or int(self.y.sum()) != expected_positive:
            raise ValueError(
                f"{split} count/label gate failed: "
                f"{len(self.y)}/{int(self.y.sum())}"
            )
        self.load_seconds = time.perf_counter() - started

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[index]), torch.tensor(self.y[index])


def load_model_classes(repo_root: Path) -> dict[str, type[nn.Module]]:
    pyehr_root = repo_root / "pyehr"
    if str(pyehr_root) not in sys.path:
        sys.path.insert(0, str(pyehr_root))
    from models.adacare import AdaCare
    from models.concare import ConCare
    from models.retain import RETAIN

    return {"RETAIN": RETAIN, "ConCare": ConCare, "AdaCare": AdaCare}


class OutcomeModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        model_class: type[nn.Module],
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        if model_name == "ConCare":
            self.encoder = model_class(lab_dim=59, demo_dim=2, hidden_dim=hidden_dim)
        else:
            self.encoder = model_class(input_dim=61, hidden_dim=hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if self.model_name == "ConCare":
            embedding, aux_loss, feature_weight = self.encoder(
                x[:, :, 2:], x[:, 0, :2], mask
            )
        else:
            embedding, feature_weight = self.encoder(x, mask)
            aux_loss = torch.zeros((), device=x.device)
        logits = self.head(embedding).reshape(-1)
        probability = torch.sigmoid(logits)
        return logits, probability, embedding, feature_weight, aux_loss


def fresh_model(
    model_name: str,
    repo_root: Path,
    hidden_dim: int,
    device: torch.device,
) -> OutcomeModel:
    classes = load_model_classes(repo_root)
    return OutcomeModel(model_name, classes[model_name], hidden_dim).to(device)


def load_checkpoint_model(
    model_name: str,
    checkpoint_path: Path,
    repo_root: Path,
    hidden_dim: int,
    device: torch.device,
) -> tuple[OutcomeModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    for key, expected in (
        ("model", model_name),
        ("task", "outcome"),
        ("input_dim", 61),
        ("lab_dim", 59),
        ("demo_dim", 2),
        ("data_version", "formal_v2"),
    ):
        if checkpoint.get(key) != expected:
            raise ValueError(
                f"checkpoint {key} mismatch: {checkpoint.get(key)!r} != {expected!r}"
            )
    model = fresh_model(model_name, repo_root, hidden_dim, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def make_loader(
    dataset: FixedSequenceDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        generator=generator if shuffle else None,
    )


def best_validation_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if not len(thresholds):
        return 0.5
    scores = np.minimum(precision[:-1], recall[:-1])
    return float(thresholds[int(np.nanargmax(scores))])


def binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    if not np.isfinite(probabilities).all():
        raise ValueError("non-finite probabilities")
    if float(probabilities.min()) < 0 or float(probabilities.max()) > 1:
        raise ValueError("probability outside [0,1]")
    predictions = (probabilities >= threshold).astype(np.int64)
    precision = float(precision_score(labels, predictions, zero_division=0))
    sensitivity = float(recall_score(labels, predictions, zero_division=0))
    return {
        "auprc": float(average_precision_score(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "precision": precision,
        "sensitivity": sensitivity,
        "min_precision_sensitivity": min(precision, sensitivity),
        "accuracy": float(accuracy_score(labels, predictions)),
        "threshold": float(threshold),
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "samples": int(len(labels)),
        "positive": int(labels.sum()),
        "negative": int((labels == 0).sum()),
    }


def evaluate(
    model: OutcomeModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    criterion = nn.BCELoss()
    losses: list[float] = []
    logits_values: list[np.ndarray] = []
    probability_values: list[np.ndarray] = []
    label_values: list[np.ndarray] = []
    with torch.inference_mode():
        for x, labels in loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, probabilities, _, _, _ = model(x)
            losses.append(float(criterion(probabilities, labels).item()))
            logits_values.append(logits.cpu().numpy())
            probability_values.append(probabilities.cpu().numpy())
            label_values.append(labels.cpu().numpy())
    return {
        "loss": float(np.mean(losses)),
        "logits": np.concatenate(logits_values),
        "probabilities": np.concatenate(probability_values),
        "labels": np.concatenate(label_values).astype(np.int64),
    }


def write_history(rows: list[dict[str, Any]], output_dir: Path) -> None:
    write_json(output_dir / "epoch_history.json", rows)
    with (output_dir / "epoch_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "auprc": [],
        "auroc": [],
        "min_precision_sensitivity": [],
    }
    n = len(labels)
    for _ in range(iterations):
        indices = rng.integers(0, n, size=n)
        sample_labels = labels[indices]
        if len(np.unique(sample_labels)) < 2:
            raise RuntimeError("bootstrap produced a single-class resample")
        metrics = binary_metrics(
            sample_labels, probabilities[indices], threshold
        )
        for key in values:
            values[key].append(float(metrics[key]))
    return {
        "iterations": iterations,
        "seed": seed,
        **{
            key: {
                "mean": float(np.mean(items)),
                "std": float(np.std(items)),
                "mean_percent": float(np.mean(items) * 100),
                "std_percent": float(np.std(items) * 100),
            }
            for key, items in values.items()
        },
    }


def train_model(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with PeakRSSMonitor() as rss_monitor:
        data_started = time.perf_counter()
        train_dataset = FixedSequenceDataset(args.data_dir, "train")
        val_dataset = FixedSequenceDataset(args.data_dir, "val")
        test_dataset = FixedSequenceDataset(args.data_dir, "test")
        data_loading_seconds = time.perf_counter() - data_started
        train_loader = make_loader(
            train_dataset, args.batch_size, True, args.seed
        )
        val_loader = make_loader(val_dataset, args.batch_size, False, args.seed)
        test_loader = make_loader(test_dataset, args.batch_size, False, args.seed)
        model = fresh_model(args.model, args.repo_root, args.hidden_dim, device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        history: list[dict[str, Any]] = []
        best_auprc = -math.inf
        best_epoch = 0
        stale_epochs = 0
        best_checkpoint = args.output_dir / "best_checkpoint.pt"
        for epoch in range(1, args.max_epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            losses: list[float] = []
            for x, labels in train_loader:
                x = x.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                _, probabilities, _, _, aux_loss = model(x)
                loss = criterion(probabilities, labels)
                if args.model == "ConCare":
                    loss = loss + 10.0 * aux_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
            validation = evaluate(model, val_loader, device)
            threshold = best_validation_threshold(
                validation["labels"], validation["probabilities"]
            )
            val_metrics = binary_metrics(
                validation["labels"], validation["probabilities"], threshold
            )
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_loss": validation["loss"],
                "val_auprc": val_metrics["auprc"],
                "val_auroc": val_metrics["auroc"],
                "val_precision": val_metrics["precision"],
                "val_sensitivity": val_metrics["sensitivity"],
                "val_min_precision_sensitivity": val_metrics[
                    "min_precision_sensitivity"
                ],
                "val_threshold": threshold,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "gpu_peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "gpu_peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
                "rss_bytes": int(psutil.Process().memory_info().rss),
            }
            history.append(row)
            print("EPOCH", json.dumps(row, sort_keys=True), flush=True)
            if float(val_metrics["auprc"]) > best_auprc + 1e-12:
                best_auprc = float(val_metrics["auprc"])
                best_epoch = epoch
                stale_epochs = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model": args.model,
                        "task": "outcome",
                        "seed": args.seed,
                        "hidden_dim": args.hidden_dim,
                        "input_dim": 61,
                        "lab_dim": 59,
                        "demo_dim": 2,
                        "data_version": "formal_v2",
                        "epoch": epoch,
                        "validation_auprc": best_auprc,
                    },
                    best_checkpoint,
                )
            else:
                stale_epochs += 1
            write_history(history, args.output_dir)
            if stale_epochs >= args.patience:
                print(
                    f"EARLY_STOP epoch={epoch} best_epoch={best_epoch}", flush=True
                )
                break

        best_model, checkpoint = load_checkpoint_model(
            args.model,
            best_checkpoint,
            args.repo_root,
            args.hidden_dim,
            device,
        )
        fixed_x, _ = next(iter(val_loader))
        fixed_x = fixed_x.to(device)
        with torch.inference_mode():
            before = best_model(fixed_x)[1].cpu()
        reload_model, _ = load_checkpoint_model(
            args.model,
            best_checkpoint,
            args.repo_root,
            args.hidden_dim,
            device,
        )
        with torch.inference_mode():
            after = reload_model(fixed_x)[1].cpu()
        reload_max_abs_error = float((before - after).abs().max().item())
        if reload_max_abs_error > 1e-7:
            raise RuntimeError(
                f"checkpoint reload mismatch {reload_max_abs_error}"
            )
        best_validation = evaluate(best_model, val_loader, device)
        validation_threshold = best_validation_threshold(
            best_validation["labels"], best_validation["probabilities"]
        )
        validation_metrics = binary_metrics(
            best_validation["labels"],
            best_validation["probabilities"],
            validation_threshold,
        )
        write_json(
            args.output_dir / "validation_threshold.json",
            {
                "selected_on": "validation",
                "criterion": "maximize min(precision, sensitivity)",
                "threshold": validation_threshold,
                "metrics": validation_metrics,
            },
        )
        test_started = time.perf_counter()
        test_output = evaluate(best_model, test_loader, device)
        test_metrics = binary_metrics(
            test_output["labels"],
            test_output["probabilities"],
            validation_threshold,
        )
        test_metrics.update(
            {
                "loss": float(test_output["loss"]),
                "evaluation_seconds": time.perf_counter() - test_started,
                "evaluations_after_model_selection": 1,
                "auprc_percent": float(test_metrics["auprc"]) * 100,
                "auroc_percent": float(test_metrics["auroc"]) * 100,
                "min_precision_sensitivity_percent": float(
                    test_metrics["min_precision_sensitivity"]
                )
                * 100,
            }
        )
        write_json(args.output_dir / "test_metrics.json", test_metrics)
        np.savez_compressed(
            args.output_dir / "test_predictions.npz",
            logits=test_output["logits"],
            probabilities=test_output["probabilities"],
            labels=test_output["labels"],
        )
        bootstrap = bootstrap_metrics(
            test_output["labels"],
            test_output["probabilities"],
            validation_threshold,
            args.bootstrap_iterations,
            args.seed,
        )
        write_json(args.output_dir / "bootstrap_metrics.json", bootstrap)
        report = {
            "status": "PASS",
            "model": args.model,
            "data_version": "formal_v2",
            "best_epoch": best_epoch,
            "best_validation_auprc": best_auprc,
            "checkpoint": str(best_checkpoint),
            "checkpoint_sha256": sha256_file(best_checkpoint),
            "checkpoint_reload_max_abs_error": reload_max_abs_error,
            "data_loading_seconds": data_loading_seconds,
            "total_seconds": time.perf_counter() - started,
            "gpu_peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "gpu_peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)
            ),
            "peak_rss_bytes": rss_monitor.peak_rss,
            "test_metrics": test_metrics,
            "bootstrap_metrics": bootstrap,
            "config": {
                "seed": args.seed,
                "batch_size": args.batch_size,
                "hidden_dim": args.hidden_dim,
                "learning_rate": args.learning_rate,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "optimizer": "AdamW",
                "loss": "BCE plus repository ConCare auxiliary loss",
            },
            "checkpoint_metadata": {
                key: value
                for key, value in checkpoint.items()
                if key != "model_state_dict"
            },
        }
        write_json(args.output_dir / "run_manifest.json", report)
        return report


def predict_in_batches(
    model: OutcomeModel,
    values: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    logits: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    feature_weights: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            x = torch.from_numpy(values[start : start + batch_size]).to(
                device=device, dtype=torch.float32
            )
            logit, probability, embedding, feature_weight, _ = model(x)
            logits.append(logit.cpu().numpy())
            probabilities.append(probability.cpu().numpy())
            embeddings.append(embedding.cpu().numpy())
            feature_weights.append(feature_weight.cpu().numpy())
    return {
        "logits": np.concatenate(logits),
        "probabilities": np.concatenate(probabilities),
        "embeddings": np.concatenate(embeddings),
        "feature_importance": np.concatenate(feature_weights),
    }


def run_invariance_tests(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = torch.device(args.device)
    dataset = FixedSequenceDataset(args.data_dir, "test")
    values = dataset.x[:128]
    model, _ = load_checkpoint_model(
        args.model,
        args.checkpoint,
        args.repo_root,
        args.hidden_dim,
        device,
    )
    reference = predict_in_batches(model, values, 128, device)["probabilities"]
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(values))
    inverse = np.argsort(permutation)
    permuted = predict_in_batches(
        model, values[permutation], 128, device
    )["probabilities"][inverse]
    permutation_error = float(np.max(np.abs(reference - permuted)))
    batch_size_errors: dict[str, float] = {}
    for batch_size in (1, 2, 8, 32, 128):
        prediction = predict_in_batches(
            model, values, batch_size, device
        )["probabilities"]
        batch_size_errors[str(batch_size)] = float(
            np.max(np.abs(reference - prediction))
        )
    target = values[31:32]
    companion_a = np.concatenate([values[:31], target], axis=0)
    companion_b = np.concatenate([values[32:63], target], axis=0)
    target_alone = predict_in_batches(
        model, target, 1, device
    )["probabilities"][0]
    target_with_a = predict_in_batches(
        model, companion_a, 32, device
    )["probabilities"][-1]
    target_with_b = predict_in_batches(
        model, companion_b, 32, device
    )["probabilities"][-1]
    companion_error = float(
        max(abs(target_alone - target_with_a), abs(target_alone - target_with_b))
    )
    reload_model, _ = load_checkpoint_model(
        args.model,
        args.checkpoint,
        args.repo_root,
        args.hidden_dim,
        device,
    )
    reload_prediction = predict_in_batches(
        reload_model, values, 128, device
    )["probabilities"]
    reload_error = float(np.max(np.abs(reference - reload_prediction)))
    tolerance = 1e-5
    report = {
        "status": "PASS"
        if (
            permutation_error <= tolerance
            and companion_error <= tolerance
            and max(batch_size_errors.values()) <= tolerance
            and reload_error <= 1e-7
        )
        else "FAIL",
        "model": args.model,
        "samples": len(values),
        "permutation_max_abs_error": permutation_error,
        "single_vs_companion_max_abs_error": companion_error,
        "batch_size_max_abs_errors": batch_size_errors,
        "checkpoint_reload_max_abs_error": reload_error,
        "tolerance": tolerance,
        "probability_finite": bool(np.isfinite(reference).all()),
        "probability_range": [float(reference.min()), float(reference.max())],
    }
    write_json(args.output, report)
    if report["status"] != "PASS":
        raise RuntimeError(f"invariance gate failed: {report}")
    return report


def select_anonymous_test_indices(
    labels: np.ndarray, sample_count: int = 32, seed: int = 42
) -> np.ndarray:
    if sample_count > len(labels):
        raise ValueError("sample_count exceeds available labels")
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if not len(positive) or not len(negative):
        raise ValueError("both label classes are required")
    rng = np.random.default_rng(seed)
    positive_count = min(len(positive), max(1, sample_count // 2))
    negative_count = sample_count - positive_count
    if negative_count > len(negative):
        negative_count = len(negative)
        positive_count = sample_count - negative_count
    chosen = np.concatenate(
        [
            rng.choice(negative, size=negative_count, replace=False),
            rng.choice(positive, size=positive_count, replace=False),
        ]
    )
    rng.shuffle(chosen)
    return chosen.astype(np.int64)


def make_sample_manifest(args: argparse.Namespace) -> dict[str, Any]:
    dataset = FixedSequenceDataset(args.data_dir, "test")
    indices = select_anonymous_test_indices(
        dataset.y.astype(np.int64), args.sample_count, args.seed
    )
    labels = dataset.y[indices].astype(np.int64)
    manifest = {
        "status": "PASS",
        "split": "test",
        "seed": args.seed,
        "samples": len(indices),
        "positive": int(labels.sum()),
        "negative": int((labels == 0).sum()),
        "anonymous_names": [f"sample_{index:03d}" for index in range(len(indices))],
        "split_positions": indices.tolist(),
        "contains_database_ids": False,
    }
    write_json(args.output, manifest)
    return manifest


def load_sample_indices(path: Path) -> np.ndarray:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("contains_database_ids") is not False:
        raise ValueError("sample manifest privacy gate failed")
    return np.asarray(manifest["split_positions"], dtype=np.int64)


def export_expert_smoke(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    dataset = FixedSequenceDataset(args.data_dir, "test")
    indices = load_sample_indices(args.sample_manifest)
    model, _ = load_checkpoint_model(
        args.model,
        args.checkpoint,
        args.repo_root,
        args.hidden_dim,
        device,
    )
    output = predict_in_batches(
        model, dataset.x[indices], args.batch_size, device
    )
    official = np.load(args.test_predictions)
    probability_error = float(
        np.max(
            np.abs(
                output["probabilities"]
                - official["probabilities"][indices]
            )
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output_dir / "expert_output_smoke.npz",
        anonymous_sample_indices=np.arange(len(indices), dtype=np.int64),
        labels=dataset.y[indices].astype(np.int64),
        **output,
    )
    finite = all(np.isfinite(value).all() for value in output.values())
    report = {
        "status": "PASS" if finite and probability_error <= 1e-5 else "FAIL",
        "model": args.model,
        "samples": len(indices),
        "logits_shape": list(output["logits"].shape),
        "probabilities_shape": list(output["probabilities"].shape),
        "embeddings_shape": list(output["embeddings"].shape),
        "native_importance_shape": list(output["feature_importance"].shape),
        "checkpoint_prediction_max_abs_error": probability_error,
        "finite": finite,
        "contains_database_ids": False,
    }
    write_json(args.output_dir / "validation_report.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(f"expert output gate failed: {report}")
    return report


def aggregate_shap_values(values: np.ndarray) -> dict[str, Any]:
    if values.ndim != 2 or values.shape[1] != 61:
        raise ValueError(f"repository SHAP values must have shape [N,61], got {values.shape}")
    absolute = np.abs(values)
    return {
        "sample_feature_abs": absolute,
        "sample_abs_sum": absolute.sum(axis=1),
        "feature_abs_mean": absolute.mean(axis=0),
        "sample_time_abs": None,
        "time_semantics": "last_observation_only",
    }


def normalize_shap_output(value: Any, samples: int) -> np.ndarray:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"unexpected SHAP output list length {len(value)}")
        value = value[0]
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.shape == (61, samples):
        array = array.T
    if array.shape != (samples, 61):
        raise ValueError(f"unexpected SHAP output shape {array.shape}")
    return array.astype(np.float64, copy=False)


def run_shap_smoke(args: argparse.Namespace) -> dict[str, Any]:
    import shap

    seed_everything(args.seed)
    device = torch.device(args.device)
    train_dataset = FixedSequenceDataset(args.data_dir, "train")
    test_dataset = FixedSequenceDataset(args.data_dir, "test")
    indices = load_sample_indices(args.sample_manifest)
    rng = np.random.default_rng(args.seed)
    background_indices = rng.choice(
        len(train_dataset), size=args.background_size, replace=False
    )
    background = train_dataset.x[background_indices, -1, :]
    explained = test_dataset.x[indices, -1, :]
    model, _ = load_checkpoint_model(
        args.model,
        args.checkpoint,
        args.repo_root,
        args.hidden_dim,
        device,
    )

    def predict_last_observation(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        original_count = len(values)
        if args.model == "ConCare" and original_count == 1:
            values = np.repeat(values, 2, axis=0)
        sequence = values[:, np.newaxis, :]
        probabilities = predict_in_batches(
            model, sequence, max(1, min(args.predict_batch_size, len(sequence))), device
        )["probabilities"]
        return probabilities[:original_count]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with PeakRSSMonitor() as rss_monitor:
        reduced_background = shap.kmeans(background, args.kmeans_clusters)
        explainer = shap.KernelExplainer(
            predict_last_observation, reduced_background
        )
        if args.shap_nsamples:
            raw_values = explainer.shap_values(
                explained, nsamples=args.shap_nsamples
            )
        else:
            raw_values = explainer.shap_values(explained)
        shap_values = normalize_shap_output(raw_values, len(explained))
    seconds = time.perf_counter() - started
    model_outputs = predict_last_observation(explained)
    independent_outputs = np.asarray(
        [
            predict_last_observation(explained[index : index + 1])[0]
            for index in range(len(explained))
        ]
    )
    one_step_batch_invariance_error = float(
        np.max(np.abs(model_outputs - independent_outputs))
    )
    base_values = np.asarray(explainer.expected_value, dtype=np.float64).reshape(-1)
    base_value = float(base_values[0])
    reconstructed = base_value + shap_values.sum(axis=1)
    additivity_errors = np.abs(reconstructed - model_outputs)
    aggregation = aggregate_shap_values(shap_values)
    finite = bool(
        np.isfinite(shap_values).all()
        and np.isfinite(model_outputs).all()
        and np.isfinite(base_value)
    )
    all_zero_samples = int(
        np.count_nonzero(aggregation["sample_abs_sum"] == 0)
    )
    feature_names = list(
        pickle.load((args.data_dir / "labtest_features.pkl").open("rb"))
    )
    feature_names = ["Sex", "Age", *feature_names]
    if len(feature_names) != 61:
        raise ValueError("feature name contract is not 61 dimensions")
    all_zero_contract_indices = [
        index
        for index, name in enumerate(feature_names)
        if np.all(train_dataset.x[:, :, index] == 0)
        and np.all(test_dataset.x[:, :, index] == 0)
    ]
    all_zero_contract_max_abs = (
        float(np.abs(shap_values[:, all_zero_contract_indices]).max())
        if all_zero_contract_indices
        else 0.0
    )
    shap_output_path = args.output_dir / "shap_values.npz"
    np.savez_compressed(
        shap_output_path,
        shap_values=shap_values,
        base_values=np.full(len(explained), base_value),
        model_outputs=model_outputs,
        anonymous_sample_indices=np.arange(len(explained), dtype=np.int64),
        sample_feature_abs=aggregation["sample_feature_abs"],
        sample_abs_sum=aggregation["sample_abs_sum"],
        feature_abs_mean=aggregation["feature_abs_mean"],
    )
    write_json(args.output_dir / "feature_names.json", feature_names)
    report = {
        "status": "PASS"
        if (
            finite
            and all_zero_samples == 0
            and float(additivity_errors.max()) <= args.additivity_tolerance
            and all_zero_contract_max_abs <= args.zero_feature_tolerance
            and len(all_zero_contract_indices) == 18
            and one_step_batch_invariance_error <= 1e-5
        )
        else "FAIL",
        "model": args.model,
        "explainer": "shap.KernelExplainer",
        "explained_output": "sigmoid probability from one-timestep model",
        "repository_contract": {
            "source": "pyehr/importance.py",
            "input_semantics": "last observation only",
            "input_shape": [len(explained), 61],
            "shap_shape": list(shap_values.shape),
            "demo_dim": 2,
            "dynamic_dim": 59,
            "time_bins_explained": 1,
            "full_sequence_time_bins": 48,
            "sample_time_aggregation": "not applicable: repository implementation drops earlier 47 bins",
            "mask_as_feature": False,
        },
        "samples": len(explained),
        "background_raw_size": args.background_size,
        "background_kmeans_clusters": args.kmeans_clusters,
        "seconds": seconds,
        "seconds_per_sample": seconds / len(explained),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "peak_rss_bytes": rss_monitor.peak_rss,
        "finite_ratio": float(np.isfinite(shap_values).mean()),
        "nan_count": int(np.isnan(shap_values).sum()),
        "inf_count": int(np.isinf(shap_values).sum()),
        "all_zero_explanation_samples": all_zero_samples,
        "output_file_bytes": shap_output_path.stat().st_size,
        "one_step_batch_invariance_max_abs_error": one_step_batch_invariance_error,
        "sample_abs_shap_sum": aggregation["sample_abs_sum"].tolist(),
        "feature_abs_shap_mean": aggregation["feature_abs_mean"].tolist(),
        "time_abs_shap_mean": None,
        "additivity": {
            "applicable": True,
            "max_abs_error": float(additivity_errors.max()),
            "mean_abs_error": float(additivity_errors.mean()),
            "tolerance": args.additivity_tolerance,
        },
        "all_zero_contract_columns": {
            "count": len(all_zero_contract_indices),
            "indices": all_zero_contract_indices,
            "max_abs_shap": all_zero_contract_max_abs,
            "tolerance": args.zero_feature_tolerance,
        },
        "contains_database_ids": False,
    }
    write_json(args.output_dir / "runtime_metrics.json", report)
    write_json(args.output_dir / "validation_report.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(f"SHAP gate failed: {report}")
    return report


def export_full_expert_outputs(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    model, checkpoint = load_checkpoint_model(
        args.model,
        args.checkpoint,
        args.repo_root,
        args.hidden_dim,
        device,
    )
    report: dict[str, Any] = {
        "status": "PASS",
        "model": args.model,
        "data_version": "formal_v2",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "splits": {},
        "contains_database_ids": False,
    }
    for split in ("train", "val", "test"):
        dataset = FixedSequenceDataset(args.data_dir, split)
        split_dir = args.output_root / split / args.model.lower()
        split_dir.mkdir(parents=True, exist_ok=True)
        chunks: list[dict[str, Any]] = []
        for chunk_number, start in enumerate(
            range(0, len(dataset), args.chunk_size)
        ):
            stop = min(start + args.chunk_size, len(dataset))
            output_path = split_dir / f"chunk_{chunk_number:05d}.npz"
            marker_path = split_dir / f"chunk_{chunk_number:05d}.complete.json"
            if output_path.exists() or marker_path.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing chunk {output_path}"
                )
            output = predict_in_batches(
                model,
                dataset.x[start:stop],
                args.batch_size,
                device,
            )
            if not all(np.isfinite(value).all() for value in output.values()):
                raise FloatingPointError(
                    f"{split}/{args.model} expert output contains NaN/inf"
                )
            np.savez_compressed(
                output_path,
                anonymous_sample_indices=np.arange(start, stop, dtype=np.int64),
                labels=dataset.y[start:stop].astype(np.int64),
                **output,
            )
            marker = {
                "status": "PASS",
                "split": split,
                "model": args.model,
                "start": start,
                "stop": stop,
                "samples": stop - start,
                "sha256": sha256_file(output_path),
                "bytes": output_path.stat().st_size,
                "contains_database_ids": False,
            }
            write_json(marker_path, marker)
            chunks.append(marker)
        report["splits"][split] = {
            "samples": len(dataset),
            "chunks": chunks,
            "finite": True,
        }
    write_json(
        args.output_root / f"{args.model.lower()}_export_manifest.json",
        report,
    )
    return report


def align_exports(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "PASS",
        "data_version": "formal_v2",
        "models": list(MODEL_NAMES),
        "splits": {},
        "contains_database_ids": False,
    }
    for split, (expected_count, _) in EXPECTED_SPLITS.items():
        reference: np.ndarray | None = None
        model_counts: dict[str, int] = {}
        for model_name in MODEL_NAMES:
            model_dir = args.output_root / split / model_name.lower()
            paths = sorted(model_dir.glob("chunk_*.npz"))
            indices = np.concatenate(
                [np.load(path)["anonymous_sample_indices"] for path in paths]
            )
            if len(indices) != expected_count:
                raise ValueError(
                    f"{split}/{model_name} exported {len(indices)} samples"
                )
            if reference is None:
                reference = indices
            elif not np.array_equal(reference, indices):
                raise ValueError(f"{split}/{model_name} sample order mismatch")
            model_counts[model_name] = len(indices)
        report["splits"][split] = {
            "expected": expected_count,
            "model_counts": model_counts,
            "sample_order_aligned": True,
        }
    write_json(args.output_root / "alignment_report.json", report)
    return report


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run formal-v2 MIMIC-IV expert training and explanation gates."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    train_parser = subparsers.add_parser("train")
    add_common_model_args(train_parser)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--batch-size", type=int, default=128)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.add_argument("--max-epochs", type=int, default=50)
    train_parser.add_argument("--patience", type=int, default=10)
    train_parser.add_argument("--bootstrap-iterations", type=int, default=100)

    invariant_parser = subparsers.add_parser("invariance")
    add_common_model_args(invariant_parser)
    invariant_parser.add_argument("--checkpoint", type=Path, required=True)
    invariant_parser.add_argument("--output", type=Path, required=True)

    manifest_parser = subparsers.add_parser("sample-manifest")
    manifest_parser.add_argument("--data-dir", type=Path, required=True)
    manifest_parser.add_argument("--output", type=Path, required=True)
    manifest_parser.add_argument("--sample-count", type=int, default=32)
    manifest_parser.add_argument("--seed", type=int, default=42)

    expert_parser = subparsers.add_parser("expert-smoke")
    add_common_model_args(expert_parser)
    expert_parser.add_argument("--checkpoint", type=Path, required=True)
    expert_parser.add_argument("--test-predictions", type=Path, required=True)
    expert_parser.add_argument("--sample-manifest", type=Path, required=True)
    expert_parser.add_argument("--output-dir", type=Path, required=True)
    expert_parser.add_argument("--batch-size", type=int, default=32)

    shap_parser = subparsers.add_parser("shap-smoke")
    add_common_model_args(shap_parser)
    shap_parser.add_argument("--checkpoint", type=Path, required=True)
    shap_parser.add_argument("--sample-manifest", type=Path, required=True)
    shap_parser.add_argument("--output-dir", type=Path, required=True)
    shap_parser.add_argument("--background-size", type=int, default=64)
    shap_parser.add_argument("--kmeans-clusters", type=int, default=32)
    shap_parser.add_argument("--predict-batch-size", type=int, default=512)
    shap_parser.add_argument("--shap-nsamples", type=int)
    shap_parser.add_argument("--additivity-tolerance", type=float, default=1e-4)
    shap_parser.add_argument("--zero-feature-tolerance", type=float, default=1e-8)

    export_parser = subparsers.add_parser("export")
    add_common_model_args(export_parser)
    export_parser.add_argument("--checkpoint", type=Path, required=True)
    export_parser.add_argument("--output-root", type=Path, required=True)
    export_parser.add_argument("--batch-size", type=int, default=128)
    export_parser.add_argument("--chunk-size", type=int, default=1024)

    align_parser = subparsers.add_parser("align")
    align_parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    handlers = {
        "train": train_model,
        "invariance": run_invariance_tests,
        "sample-manifest": make_sample_manifest,
        "expert-smoke": export_expert_smoke,
        "shap-smoke": run_shap_smoke,
        "export": export_full_expert_outputs,
        "align": align_exports,
    }
    report = handlers[args.action](args)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
