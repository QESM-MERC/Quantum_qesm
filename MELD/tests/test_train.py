"""Training-loop control-flow tests."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import train


def test_effective_loss_weights_static_and_linear_transition():
    static = SimpleNamespace(lambda_uni=0.3, lambda_kd=0.6, loss_transition_epochs=0)
    assert train.effective_loss_weights(static, 7) == (0.3, 0.6)

    scheduled = SimpleNamespace(
        lambda_uni=0.05,
        lambda_kd=1.0,
        lambda_uni_start=0.3,
        lambda_kd_start=0.3,
        loss_transition_epochs=11,
    )
    assert train.effective_loss_weights(scheduled, 1) == (0.3, 0.3)
    assert train.effective_loss_weights(scheduled, 6) == pytest.approx((0.175, 0.65))
    assert train.effective_loss_weights(scheduled, 11) == pytest.approx((0.05, 1.0))
    assert train.effective_loss_weights(scheduled, 99) == pytest.approx((0.05, 1.0))

    static_con = SimpleNamespace(lambda_con=0.2, con_transition_epochs=0)
    assert train.effective_contrastive_weight(static_con, 5) == pytest.approx(0.2)
    ramped_con = SimpleNamespace(
        lambda_con=0.1,
        lambda_con_start=0.0,
        con_transition_epochs=11,
    )
    assert train.effective_contrastive_weight(ramped_con, 1) == pytest.approx(0.0)
    assert train.effective_contrastive_weight(ramped_con, 6) == pytest.approx(0.05)
    assert train.effective_contrastive_weight(ramped_con, 11) == pytest.approx(0.1)
    assert train.effective_contrastive_weight(ramped_con, 99) == pytest.approx(0.1)


def test_selection_improvement_is_independent_of_merge_policy():
    assert train.selection_improved("test", valid_improved=False, test_improved=True)
    assert not train.selection_improved("test", valid_improved=True, test_improved=False)
    assert train.selection_improved("valid", valid_improved=True, test_improved=False)
    assert not train.selection_improved("valid", valid_improved=False, test_improved=True)
    with pytest.raises(ValueError, match="selection_split"):
        train.selection_improved("unknown", True, True)


def test_corrected_ema_removes_initial_snapshot_and_restores_online_weights():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = train.EMA(model, decay=0.5, correct=True)

    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update(model)
    ema.swap_in(model)
    assert model.weight.item() == pytest.approx(2.0)
    ema.swap_out(model)
    assert model.weight.item() == pytest.approx(2.0)


def test_metric_plateau_scheduler_reduces_lr_only_after_stall():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1e-4)
    args = SimpleNamespace(
        lr_plateau_patience=1,
        lr_plateau_factor=0.5,
        lr_plateau_threshold=0.01,
        min_lr=1e-6,
        warmup=0,
        cosine=False,
        epochs=30,
    )
    epoch_scheduler, plateau = train.build_lr_schedulers(optimizer, args)

    assert epoch_scheduler is None
    assert plateau is not None
    plateau.step(66.0)
    plateau.step(65.9)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    plateau.step(65.8)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)


def test_css_pgm_active_set_solver_returns_simplex_minimum():
    weights = train.solve_pgm_simplex(np.eye(3, dtype=np.float64))
    assert weights == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    gram = np.array([[1.0, 0.9, 0.2], [0.9, 1.0, 0.1], [0.2, 0.1, 1.0]])
    weights = train.solve_pgm_simplex(gram)
    assert np.isfinite(weights).all()
    assert weights.min() >= 0
    assert weights.sum() == pytest.approx(1.0)
    objective = lambda value: 0.5 * value.dot(gram.dot(value))
    assert objective(weights) <= min(objective(np.eye(3)[i]) for i in range(3))

    bounded = train.solve_pgm_simplex(
        np.eye(3, dtype=np.float64), minimum_weights=(0.0, 0.4, 0.0)
    )
    assert bounded == pytest.approx([0.3, 0.4, 0.3])


def test_css_pgm_gradient_weights_are_finite_and_detached():
    shared = torch.nn.Parameter(torch.tensor([0.5, -0.25]))
    losses = (
        (shared[0] + 2 * shared[1]).square(),
        (2 * shared[0] - shared[1]).square(),
        (shared[0] - shared[1]).square(),
    )
    weights = train.css_pgm_weights(losses, (shared,))
    assert weights.shape == (3,)
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()
    assert weights.sum().item() == pytest.approx(1.0)
    assert not weights.requires_grad
    diagnostics = train.css_pgm_weights.last_diagnostics
    assert np.asarray(diagnostics["gradient_norms"]).shape == (3,)
    assert np.asarray(diagnostics["cosine_matrix"]).shape == (3, 3)
    assert np.isfinite(diagnostics["gradient_norms"]).all()
    assert np.isfinite(diagnostics["cosine_matrix"]).all()


def test_merge_valid_reuses_one_evaluation(monkeypatch):
    calls = []
    expected = (1.0, 70.0, 71.0, np.array([0, 1]), np.array([0, 1]))

    def fake_run_epoch(model, loader, device, optimizer, weight, args):
        calls.append(loader)
        return expected

    monkeypatch.setattr(train, "run_epoch", fake_run_epoch)
    valid, test = train._evaluate_validation_and_test(
        object(), {"valid": "valid-loader", "test": "test-loader"}, object(), None,
        SimpleNamespace(merge_valid=True),
    )

    assert calls == ["valid-loader"]
    assert valid is expected
    assert test is expected


def test_separate_valid_and_test_are_both_evaluated(monkeypatch):
    calls = []

    def fake_run_epoch(model, loader, device, optimizer, weight, args):
        calls.append(loader)
        return loader

    monkeypatch.setattr(train, "run_epoch", fake_run_epoch)
    valid, test = train._evaluate_validation_and_test(
        object(), {"valid": "valid-loader", "test": "test-loader"}, object(), None,
        SimpleNamespace(merge_valid=False),
    )

    assert calls == ["valid-loader", "test-loader"]
    assert valid == "valid-loader"
    assert test == "test-loader"


def test_train_epoch_records_lr_tuning_loss_components():
    class TinyMultitaskModel(torch.nn.Module):
        modalities = "tav"

        def __init__(self):
            super().__init__()
            self.main_logits = torch.nn.Parameter(torch.tensor([0.4, -0.2]))
            self.uni_logits = torch.nn.Parameter(
                torch.tensor([[0.1, -0.1], [-0.3, 0.2], [0.2, -0.4]])
            )

        def forward(self, feats, spk, mask, availability=None):
            batch, turns = mask.shape
            if availability is None:
                availability = mask.unsqueeze(-1).expand(batch, turns, 3)
            main = torch.log_softmax(self.main_logits, dim=-1).view(1, 1, 2)
            main = main.expand(batch, turns, 2)
            uni = {
                modality: torch.log_softmax(self.uni_logits[index], dim=-1)
                .view(1, 1, 2)
                .expand(batch, turns, 2)
                for index, modality in enumerate(self.modalities)
            }
            return {
                "logp": main,
                "logp_uni": uni,
                "mod_drop": torch.zeros(batch, 3, dtype=torch.bool),
                "mod_available": availability,
                "alpha": torch.full((batch, turns, 3), 1 / 3),
                "g_fused": None,
                "logp_graph": None,
            }

    model = TinyMultitaskModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    feats = {m: torch.zeros(1, 2, 1) for m in "tav"}
    loader = [(feats, torch.zeros(1, 2, dtype=torch.long),
               torch.tensor([[0, 1]]), torch.tensor([[True, True]]))]
    args = SimpleNamespace(
        logit_adjust=0.0,
        class_counts=[],
        pin_memory=False,
        test_drop_modality="",
        test_noise_sigma=0.0,
        feat_noise_sigma=0.0,
        label_smoothing=0.0,
        lambda_uni=0.1,
        lambda_kd=0.2,
        kd_temp=2.0,
        lambda_rel=0.0,
        lambda_con=0.0,
        lambda_graph=0.0,
        rdrop=0.0,
        ogm=0.0,
        clip=5.0,
    )

    train.run_epoch(model, loader, torch.device("cpu"), optimizer, None, args)
    components = train.run_epoch.last_train_components
    mechanisms = train.run_epoch.last_train_mechanisms

    assert set(components) == {"main_nll", "uni_nll", "kd", "other"}
    assert all(np.isfinite(value) for value in components.values())
    assert components["main_nll"] > 0
    assert components["uni_nll"] > 0
    assert components["kd"] > 0
    assert components["other"] == 0
    assert set(mechanisms["unimodal_wf1"]) == set("tav")
    assert set(mechanisms["unimodal_acc"]) == set("tav")
