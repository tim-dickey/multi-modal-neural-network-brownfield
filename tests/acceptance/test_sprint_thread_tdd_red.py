"""ATDD RED tests for Sprint thread: must fail before implementation.

These tests codify required behavior before feature work:
- Trainer passes double-loop signals into model forward
- Trainer applies adaptive LR using model meta output
- Dataset bootstraps bert-base-uncased when tokenizer is omitted
- Attention blocks use scaled_dot_product_attention fast-path
- Train CLI check mode validates environment without full training
- Epoch profiling writes memory/timing evidence and metrics include consumer GPU assertions
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, cast

import pytest
import torch
import torch.nn as nn

from src.data.dataset import MultiModalDataset
from src.models.text_encoder import TextMultiHeadAttention
from src.models.vision_encoder import MultiHeadAttention
from src.training.trainer import Trainer


class _SpyAdaptiveLR:
    def __init__(self) -> None:
        self.called = False
        self.received_scale: Optional[torch.Tensor] = None

    def update_lr(
        self,
        _optimizer: torch.optim.Optimizer,
        lr_scale: torch.Tensor,
    ) -> None:
        self.called = True
        self.received_scale = lr_scale


class _SpyModel(nn.Module):
    def __init__(self, *, use_double_loop: bool, with_meta_info: bool = False) -> None:
        super().__init__()
        self.use_double_loop = use_double_loop
        self.with_meta_info = with_meta_info
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(4, 3))
        self.seen_forward_kwargs: Dict[str, Any] = {}

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        current_loss: Optional[torch.Tensor] = None,
        current_accuracy: Optional[torch.Tensor] = None,
        gradient_norm: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        del input_ids, attention_mask
        self.seen_forward_kwargs = {
            "current_loss": current_loss,
            "current_accuracy": current_accuracy,
            "gradient_norm": gradient_norm,
        }
        out: Dict[str, Any] = {"logits": self.net(images)}
        if self.with_meta_info:
            out["meta_info"] = {
                "lr_scale": torch.tensor(0.5),
                "meta_loss": torch.tensor(0.1),
            }
        return out


def _make_minimal_trainer(tmp_path: Path, model: nn.Module) -> Trainer:
    trainer = object.__new__(Trainer)
    trainer.config = {
        "training": {
            "log_interval": 100,
            "gradient_clip": 0.0,
            "mixed_precision": "bf16",
        }
    }
    trainer.device = torch.device("cpu")
    trainer.logger = __import__("logging").getLogger("tests.atdd")
    trainer.model = cast(Any, model)
    trainer.criterion = nn.CrossEntropyLoss()
    trainer.meta_criterion = nn.Identity()
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer.scheduler = None
    trainer.scheduler_update_freq = "step"
    trainer.adaptive_lr = _SpyAdaptiveLR()
    trainer.wandb_logger = None
    trainer.output_dir = tmp_path
    trainer.checkpoint_dir = tmp_path
    trainer.global_step = 0
    trainer.current_epoch = 0
    trainer.best_val_loss = float("inf")
    trainer.use_amp = False
    trainer.scaler = None

    batch = {
        "images": torch.randn(2, 1, 2, 2),
        "input_ids": torch.randint(0, 10, (2, 4)),
        "attention_mask": torch.ones(2, 4),
        "labels": torch.tensor([0, 1], dtype=torch.long),
    }
    trainer.train_loader = [batch]
    return trainer


@pytest.mark.acceptance
@pytest.mark.tdd_red
@pytest.mark.training
def test_train_epoch_passes_double_loop_signals_to_model(tmp_path: Path) -> None:
    trainer = _make_minimal_trainer(tmp_path, _SpyModel(use_double_loop=True))

    trainer.train_epoch(0)

    seen = cast(Dict[str, Any], trainer.model.seen_forward_kwargs)
    assert seen["current_loss"] is not None
    assert seen["current_accuracy"] is not None
    assert seen["gradient_norm"] is not None


@pytest.mark.acceptance
@pytest.mark.tdd_red
@pytest.mark.training
def test_train_epoch_applies_adaptive_lr_from_meta_info(tmp_path: Path) -> None:
    trainer = _make_minimal_trainer(
        tmp_path,
        _SpyModel(use_double_loop=True, with_meta_info=True),
    )

    trainer.train_epoch(0)

    assert trainer.adaptive_lr.called is True
    assert trainer.adaptive_lr.received_scale is not None


@pytest.mark.acceptance
@pytest.mark.tdd_red
@pytest.mark.data
def test_dataset_bootstraps_bert_tokenizer_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    temp_data_dir: Path,
) -> None:
    calls = {"count": 0}

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_name: str) -> Any:
            calls["count"] += 1

            class _Tok:
                def __call__(
                    self,
                    _text: str,
                    max_length: int,
                    _padding: str,
                    _truncation: bool,
                    _return_tensors: str,
                ) -> Dict[str, torch.Tensor]:
                    return {
                        "input_ids": torch.zeros((1, max_length), dtype=torch.long),
                        "attention_mask": torch.ones((1, max_length), dtype=torch.long),
                    }

            return _Tok()

    class _FakeTransformers:
        AutoTokenizer = _FakeAutoTokenizer

    import sys

    monkeypatch.setitem(sys.modules, "transformers", _FakeTransformers)

    dataset = MultiModalDataset(
        data_path=str(temp_data_dir),
        split="train",
        img_size=32,
        max_text_length=16,
        tokenizer=None,
    )
    sample = dataset[0]

    assert "input_ids" in sample
    assert calls["count"] == 1


@pytest.mark.acceptance
@pytest.mark.tdd_red
@pytest.mark.model
def test_vision_attention_uses_scaled_dot_product_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"count": 0}

    def _fake_sdpa(
        q: torch.Tensor,
        _k: torch.Tensor,
        _v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        del attn_mask, dropout_p, is_causal
        called["count"] += 1
        return torch.zeros_like(q)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", _fake_sdpa)

    attn = MultiHeadAttention(hidden_dim=32, num_heads=4, dropout=0.0)
    x = torch.randn(2, 8, 32)
    _ = attn(x)

    assert called["count"] >= 1


@pytest.mark.acceptance
@pytest.mark.tdd_red
@pytest.mark.model
def test_text_attention_uses_scaled_dot_product_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"count": 0}

    def _fake_sdpa(
        q: torch.Tensor,
        _k: torch.Tensor,
        _v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        del attn_mask, dropout_p, is_causal
        called["count"] += 1
        return torch.zeros_like(q)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", _fake_sdpa)

    attn = TextMultiHeadAttention(hidden_dim=32, num_heads=4, dropout=0.0)
    x = torch.randn(2, 8, 32)
    _ = attn(x)

    assert called["count"] >= 1


@pytest.mark.acceptance
@pytest.mark.tdd_red
@pytest.mark.integration
def test_train_cli_check_mode_validates_without_full_training(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """P5 RED: --check mode should validate and exit successfully without training."""
    import importlib
    import types

    class _FakeTrainer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.train_called = False

        def train(self) -> None:
            self.train_called = True

    fake_training_module = types.SimpleNamespace(Trainer=_FakeTrainer)

    import sys

    monkeypatch.setitem(sys.modules, "src.training", fake_training_module)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--check", "--config", "configs/default.yaml"],
    )

    train_module = importlib.import_module("train")
    train_module = importlib.reload(train_module)

    train_module.main()
    output = capsys.readouterr().out
    assert "check" in output.lower()
    assert "config" in output.lower()
    assert "model" in output.lower()
    assert "data" in output.lower()


@pytest.mark.acceptance
@pytest.mark.tdd_red
@pytest.mark.training
def test_train_epoch_saves_gpu_profile_artifact_with_vram_and_step_time(
    tmp_path: Path,
) -> None:
    """P6 RED: epoch run should persist GPU profile evidence artifact."""
    trainer = _make_minimal_trainer(tmp_path, _SpyModel(use_double_loop=False))

    trainer.train_epoch(0)

    profile_file = tmp_path / "profiling" / "epoch_0000_profile.json"
    assert profile_file.exists()

    import json

    profile = json.loads(profile_file.read_text(encoding="utf-8"))
    assert "peak_vram_gb" in profile
    assert "avg_step_time_ms" in profile
    assert "run_command" in profile


@pytest.mark.acceptance
@pytest.mark.tdd_red
@pytest.mark.integration
def test_first_epoch_consumer_gpu_path_reports_memory_assertions(tmp_path: Path) -> None:
    """P8 RED: first epoch metrics should include CI-compatible memory assertions."""
    trainer = _make_minimal_trainer(tmp_path, _SpyModel(use_double_loop=False))

    metrics = trainer.train_epoch(0)

    assert "peak_vram_gb" in metrics
    assert metrics["peak_vram_gb"] <= 11.5
    assert "oom" in metrics
    assert metrics["oom"] is False


@pytest.mark.acceptance
@pytest.mark.tdd_red
@pytest.mark.training
def test_train_epoch_uses_train_step_unified_path(tmp_path: Path) -> None:
    """P1 RED: epoch loop should execute via the unified train_step path."""
    trainer = _make_minimal_trainer(tmp_path, _SpyModel(use_double_loop=False))
    calls = {"count": 0}

    def _fake_train_step(_batch: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        calls["count"] += 1
        loss = torch.tensor(1.0, requires_grad=True)
        acc = torch.tensor(0.5)
        return loss, acc

    trainer.train_step = cast(Any, _fake_train_step)
    trainer.train_epoch(0)

    assert calls["count"] == len(trainer.train_loader)


@pytest.mark.acceptance
@pytest.mark.tdd_red
@pytest.mark.training
def test_train_epoch_enters_bf16_autocast_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """P2 RED: bf16 autocast context must be active in train_epoch."""
    import contextlib

    trainer = _make_minimal_trainer(tmp_path, _SpyModel(use_double_loop=False))
    trainer.use_amp = True

    seen: Dict[str, Any] = {"called": False, "dtype": None, "device_type": None}

    def _fake_autocast(*, device_type: str, dtype: torch.dtype) -> Any:
        seen["called"] = True
        seen["dtype"] = dtype
        seen["device_type"] = device_type
        return contextlib.nullcontext()

    monkeypatch.setattr(torch, "autocast", _fake_autocast)
    trainer.train_epoch(0)

    assert seen["called"] is True
    assert seen["dtype"] == torch.bfloat16


@pytest.mark.acceptance
@pytest.mark.tdd_red
@pytest.mark.model
def test_text_attention_passes_mask_to_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """P3 RED: text attention should route attention_mask through SDPA."""
    seen = {"called": False, "attn_mask": None}

    def _fake_sdpa(
        q: torch.Tensor,
        _k: torch.Tensor,
        _v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        del dropout_p, is_causal
        seen["called"] = True
        seen["attn_mask"] = attn_mask
        return torch.zeros_like(q)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", _fake_sdpa)

    attn = TextMultiHeadAttention(hidden_dim=32, num_heads=4, dropout=0.0)
    x = torch.randn(2, 8, 32)
    mask = torch.ones(2, 8)
    _ = attn(x, attention_mask=mask)

    assert seen["called"] is True
    assert seen["attn_mask"] is not None


@pytest.mark.acceptance
@pytest.mark.tdd_red
@pytest.mark.data
def test_dataset_bootstrapped_tokenizer_is_reused_across_samples(
    monkeypatch: pytest.MonkeyPatch,
    temp_data_dir: Path,
) -> None:
    """P4 RED: tokenizer should be loaded once and reused for batch/sample calls."""
    calls = {"count": 0}

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_name: str) -> Any:
            calls["count"] += 1

            class _Tok:
                def __call__(
                    self,
                    _text: str,
                    max_length: int,
                    _padding: str,
                    _truncation: bool,
                    _return_tensors: str,
                ) -> Dict[str, torch.Tensor]:
                    return {
                        "input_ids": torch.ones((1, max_length), dtype=torch.long),
                        "attention_mask": torch.ones((1, max_length), dtype=torch.long),
                    }

            return _Tok()

    class _FakeTransformers:
        AutoTokenizer = _FakeAutoTokenizer

    import sys

    monkeypatch.setitem(sys.modules, "transformers", _FakeTransformers)

    dataset = MultiModalDataset(
        data_path=str(temp_data_dir),
        split="train",
        img_size=32,
        max_text_length=16,
        tokenizer=None,
    )
    _ = dataset[0]
    _ = dataset[0]

    assert calls["count"] == 1


@pytest.mark.acceptance
@pytest.mark.tdd_red
@pytest.mark.training
def test_train_epoch_initializes_controller_inputs_as_zero_scalars(tmp_path: Path) -> None:
    """P9 RED: first-step controller inputs should initialize as zero scalar tensors."""
    trainer = _make_minimal_trainer(tmp_path, _SpyModel(use_double_loop=True))

    trainer.train_epoch(0)

    seen = cast(Dict[str, Any], trainer.model.seen_forward_kwargs)
    assert torch.is_tensor(seen["current_loss"])
    assert torch.is_tensor(seen["current_accuracy"])
    assert torch.is_tensor(seen["gradient_norm"])
    assert float(cast(torch.Tensor, seen["current_loss"]).item()) == 0.0
    assert float(cast(torch.Tensor, seen["current_accuracy"]).item()) == 0.0
    assert float(cast(torch.Tensor, seen["gradient_norm"]).item()) == 0.0


@pytest.mark.acceptance
@pytest.mark.tdd_red
@pytest.mark.training
def test_train_epoch_applies_meta_loss_from_controller_output(tmp_path: Path) -> None:
    """P10 RED: meta_criterion must be applied when controller meta_info is returned."""

    class _SpyMetaCriterion:
        def __init__(self) -> None:
            self.called = False
            self.meta_info: Optional[Dict[str, Any]] = None

        def __call__(self, task_loss: torch.Tensor, meta_info: Dict[str, Any]) -> torch.Tensor:
            self.called = True
            self.meta_info = meta_info
            return task_loss + torch.tensor(0.1)

    trainer = _make_minimal_trainer(
        tmp_path,
        _SpyModel(use_double_loop=True, with_meta_info=True),
    )
    spy_meta = _SpyMetaCriterion()
    trainer.meta_criterion = cast(Any, spy_meta)

    trainer.train_epoch(0)

    assert spy_meta.called is True
    assert spy_meta.meta_info is not None
    assert "meta_loss" in cast(Dict[str, Any], spy_meta.meta_info)
