from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
PYEHR_ROOT = REPO_ROOT / "pyehr"
if str(PYEHR_ROOT) not in sys.path:
    sys.path.insert(0, str(PYEHR_ROOT))

from models.adacare import AdaCare


class AdaCareOutcome(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = AdaCare(
            input_dim=61,
            hidden_dim=16,
            kernel_size=2,
            kernel_num=4,
            dropout=0.0,
        )
        self.head = nn.Linear(16, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        embedding, importance = self.encoder(x, mask)
        logits = self.head(embedding).squeeze(-1)
        return logits, torch.sigmoid(logits), embedding, importance


def _model_and_data(samples: int = 40) -> tuple[AdaCareOutcome, torch.Tensor]:
    torch.manual_seed(20260729)
    model = AdaCareOutcome()
    generator = torch.Generator().manual_seed(20260730)
    x = torch.randn(samples, 48, 61, generator=generator)
    return model, x


def _predict(
    model: AdaCareOutcome, x: torch.Tensor, batch_size: int
) -> np.ndarray:
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            values.append(model(x[start : start + batch_size])[1].cpu())
    return torch.cat(values).numpy()


def test_adacare_predictions_are_invariant_to_batch_permutation():
    model, x = _model_and_data(32)
    reference = _predict(model, x, batch_size=32)
    permutation = torch.randperm(len(x), generator=torch.Generator().manual_seed(17))
    restored = np.empty_like(reference)
    restored[permutation.numpy()] = _predict(model, x[permutation], batch_size=32)

    assert np.max(np.abs(reference - restored)) <= 1e-5


def test_adacare_target_is_independent_of_batch_companions():
    model, x = _model_and_data(65)
    target = x[32:33]
    alone = _predict(model, target, batch_size=1)
    with_first_companions = _predict(model, torch.cat([x[:31], target]), 32)[-1:]
    with_other_companions = _predict(model, torch.cat([x[33:64], target]), 32)[-1:]

    assert np.max(np.abs(alone - with_first_companions)) <= 1e-5
    assert np.max(np.abs(alone - with_other_companions)) <= 1e-5


def test_adacare_predictions_are_invariant_to_batch_size():
    model, x = _model_and_data(139)
    predictions = {
        batch_size: _predict(model, x, batch_size)
        for batch_size in (1, 2, 8, 32, 128)
    }
    reference = predictions[1]

    assert max(
        float(np.max(np.abs(reference - prediction)))
        for prediction in predictions.values()
    ) <= 1e-5


def test_adacare_checkpoint_reload_is_identical(tmp_path: Path):
    model, x = _model_and_data(8)
    reference = _predict(model, x, batch_size=8)
    checkpoint = tmp_path / "adacare.pt"
    torch.save(model.state_dict(), checkpoint)

    reloaded, _ = _model_and_data(8)
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    actual = _predict(reloaded, x, batch_size=8)

    assert np.max(np.abs(reference - actual)) <= 1e-7


def test_adacare_shapes_for_single_and_multi_sample_batches():
    model, x = _model_and_data(5)
    model.eval()
    with torch.inference_mode():
        for size in (1, 5):
            logits, probabilities, embeddings, importance = model(x[:size])
            assert logits.shape == (size,)
            assert probabilities.shape == (size,)
            assert embeddings.shape == (size, 16)
            assert importance.shape == (size, 48, 61)
            assert torch.isfinite(logits).all()
            assert torch.isfinite(probabilities).all()
            assert torch.isfinite(embeddings).all()
            assert torch.isfinite(importance).all()


def test_adacare_backward_has_finite_gradients_and_optimizer_step():
    model, x = _model_and_data(4)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    logits, _, _, _ = model(x)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    optimizer.step()
