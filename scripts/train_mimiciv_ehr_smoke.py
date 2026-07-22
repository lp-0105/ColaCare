from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
import threading
import time
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class PeakRSSMonitor:
    """Sample process RSS without changing the training environment."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self.peak_rss_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        process = psutil.Process()
        while not self._stop.is_set():
            self.peak_rss_bytes = max(
                self.peak_rss_bytes, int(process.memory_info().rss)
            )
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        self.peak_rss_bytes = int(psutil.Process().memory_info().rss)
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_rss_bytes = max(
            self.peak_rss_bytes, int(psutil.Process().memory_info().rss)
        )


class PickleSequenceDataset(Dataset):
    def __init__(self, data_dir: Path, mode: str) -> None:
        self.x = pd.read_pickle(data_dir / f"{mode}_x.pkl")
        self.y = pd.read_pickle(data_dir / f"{mode}_y.pkl")
        if len(self.x) != len(self.y):
            raise ValueError(f"{mode} x/y length mismatch")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.x[index], dtype=np.float32), np.asarray(
            self.y[index], dtype=np.float32
        )


def pad_collate(
    batch: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y = zip(*batch)
    lengths = torch.tensor([len(item) for item in x], dtype=torch.long)
    x_pad = pad_sequence([torch.from_numpy(item) for item in x], batch_first=True)
    y_pad = pad_sequence([torch.from_numpy(item) for item in y], batch_first=True)
    return x_pad.float(), y_pad.float(), lengths


def cuda_device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(f"Expected CUDA device, got {device}")
    return 0 if device.index is None else int(device.index)


def initialize_cuda_for_memory_stats(device: torch.device) -> int:
    """Initialize the selected CUDA context before calling memory-stat APIs."""
    index = cuda_device_index(device)
    torch.cuda.set_device(index)
    return index


def _load_retain_class(repo_root: Path):
    model_path = repo_root / "pyehr" / "models" / "retain.py"
    spec = importlib.util.spec_from_file_location("colacare_retain_smoke", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load RETAIN implementation from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RETAIN


class RetainOutcomeModel(nn.Module):
    def __init__(self, retain_class, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = retain_class(input_dim=input_dim, hidden_dim=hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.to(
            x.device
        ).unsqueeze(1)
        embedding, _ = self.encoder(x, mask)
        return self.head(embedding).squeeze(-1)


def _last_outcome(y: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    indexes = (lengths - 1).to(y.device)
    return y[torch.arange(y.size(0), device=y.device), indexes, 0]


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    criterion = nn.BCELoss()
    probabilities: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    losses: list[float] = []
    with torch.no_grad():
        for x, y, lengths in loader:
            x = x.to(device)
            y = y.to(device)
            target = _last_outcome(y, lengths)
            probability = model(x, lengths)
            losses.append(float(criterion(probability, target).item()))
            probabilities.append(probability.detach().cpu())
            labels.append(target.detach().cpu())
    probability = torch.cat(probabilities)
    label = torch.cat(labels)
    prediction = (probability >= 0.5).float()
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float((prediction == label).float().mean().item()),
        "probability_min": float(probability.min().item()),
        "probability_max": float(probability.max().item()),
        "probability_mean": float(probability.mean().item()),
        "valid_probability": bool(
            torch.isfinite(probability).all()
            and (probability >= 0).all()
            and (probability <= 1).all()
        ),
        "samples": int(len(label)),
    }


def _run_smoke_impl(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 16,
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
    require_cuda: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test but is unavailable")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_load_started = time.perf_counter()
    rss_before_data = int(psutil.Process().memory_info().rss)
    train_dataset = PickleSequenceDataset(data_dir, "train")
    val_dataset = PickleSequenceDataset(data_dir, "val")
    test_dataset = PickleSequenceDataset(data_dir, "test")
    data_loading_seconds = time.perf_counter() - data_load_started
    rss_after_data = int(psutil.Process().memory_info().rss)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pad_collate,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=pad_collate,
    )

    first_x = np.asarray(train_dataset.x[0])
    first_y = np.asarray(train_dataset.y[0])
    input_dim = int(first_x.shape[1])
    repo_root = Path(__file__).resolve().parents[1]
    retain_class = _load_retain_class(repo_root)

    if device.type == "cuda":
        gpu_index = initialize_cuda_for_memory_stats(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(gpu_index)
        memory_before = int(torch.cuda.memory_allocated(gpu_index))
    else:
        gpu_index = 0
        memory_before = 0
    model = RetainOutcomeModel(retain_class, input_dim, hidden_dim).to(device)
    memory_after_model = (
        int(torch.cuda.memory_allocated(gpu_index)) if device.type == "cuda" else 0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    epoch_losses: list[float] = []
    epoch_seconds: list[float] = []
    for _ in range(epochs):
        epoch_started = time.perf_counter()
        model.train()
        batch_losses: list[float] = []
        for x, y, lengths in train_loader:
            x = x.to(device)
            y = y.to(device)
            target = _last_outcome(y, lengths)
            optimizer.zero_grad(set_to_none=True)
            probability = model(x, lengths)
            loss = criterion(probability, target)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        epoch_losses.append(float(np.mean(batch_losses)))
        epoch_seconds.append(time.perf_counter() - epoch_started)

    validation = _evaluate(model, val_loader, device)
    test = _evaluate(model, test_loader, device)
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(gpu_index)) if device.type == "cuda" else 0
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(gpu_index)) if device.type == "cuda" else 0
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "retain_outcome_smoke.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model": "RETAIN",
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "task": "outcome",
            "seed": seed,
        },
        checkpoint_path,
    )
    reloaded_model = RetainOutcomeModel(retain_class, input_dim, hidden_dim).to(device)
    checkpoint_payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    reloaded_model.load_state_dict(checkpoint_payload["model_state_dict"])
    reloaded_model.eval()
    reload_x, _, reload_lengths = next(iter(test_loader))
    with torch.no_grad():
        reload_probability = reloaded_model(
            reload_x.to(device), reload_lengths
        ).detach().cpu()
    reload_valid = bool(
        torch.isfinite(reload_probability).all()
        and (reload_probability >= 0).all()
        and (reload_probability <= 1).all()
    )
    elapsed = time.perf_counter() - started
    result = {
        "purpose": "local data/model plumbing smoke test; not a paper metric",
        "model": "RETAIN",
        "task": "MIMIC-IV in-hospital outcome",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(gpu_index) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "learning_rate": learning_rate,
        "data_loading_seconds": round(data_loading_seconds, 3),
        "epoch_seconds": [round(value, 3) for value in epoch_seconds],
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "first_x_shape": list(first_x.shape),
        "first_y_shape": list(first_y.shape),
        "epoch_train_loss": epoch_losses,
        "loss_decreased": bool(epoch_losses[-1] <= epoch_losses[0]),
        "validation": validation,
        "test": test,
        "cuda_memory_bytes": {
            "before_model": memory_before,
            "after_model": memory_after_model,
            "peak_allocated": peak_allocated,
            "peak_reserved": peak_reserved,
        },
        "system_memory_bytes": {
            "rss_before_data": rss_before_data,
            "rss_after_data": rss_after_data,
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_reload": {
            "success": True,
            "valid_probability": reload_valid,
            "batch_probabilities": int(reload_probability.numel()),
        },
        "elapsed_seconds": round(elapsed, 3),
    }
    metrics_path = output_dir / "retain_outcome_smoke_metrics.json"
    result["metrics_path"] = str(metrics_path)
    return result


def run_smoke(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 16,
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
    require_cuda: bool = True,
) -> dict[str, Any]:
    with PeakRSSMonitor() as monitor:
        result = _run_smoke_impl(
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            seed=seed,
            require_cuda=require_cuda,
        )
    result["system_memory_bytes"]["peak_rss"] = monitor.peak_rss_bytes
    metrics_path = Path(result["metrics_path"])
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal RETAIN outcome smoke test.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("ehr_datasets/mimic-iv/processed/fold_1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ehr_datasets/mimic-iv/processed/smoke_results"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_smoke(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        seed=args.seed,
        require_cuda=not args.allow_cpu,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
