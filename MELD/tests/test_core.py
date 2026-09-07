"""Numerical and integration tests for the QESM core model."""

import torch

from qotoc.complex_ops import init_frequencies, make_rotation, unit_normalize
from qotoc.model import QOTOCModel


def _random_state(dim):
    state = torch.complex(torch.randn(dim), torch.randn(dim))
    return state / torch.linalg.vector_norm(state)


def _householder(state):
    dim = state.shape[0]
    identity = torch.eye(dim, dtype=torch.complex64)
    return identity - 2.0 * torch.outer(state, state.conj())


def _toy_model(unimodal_path="post-cross", mask_missing_modalities=False):
    return QOTOCModel(
        feat_dims={"t": 12, "a": 10, "v": 8},
        n_speakers=2,
        n_classes=6,
        d_model=32,
        n_heads=4,
        n_temporal_layers=1,
        n_cross_layers=1,
        born_rank=8,
        gate_hidden=16,
        input_dropout=0.0,
        attn_dropout=0.0,
        state_dropout=0.0,
        unimodal_path=unimodal_path,
        mask_missing_modalities=mask_missing_modalities,
    )


def _toy_inputs(batch_size=2, sequence_length=6):
    features = {
        "t": torch.randn(batch_size, sequence_length, 12),
        "a": torch.randn(batch_size, sequence_length, 10),
        "v": torch.randn(batch_size, sequence_length, 8),
    }
    speakers = torch.zeros(batch_size, sequence_length, 2)
    speakers[:, :, 0] = 1
    mask = torch.ones(batch_size, sequence_length, dtype=torch.bool)
    return features, speakers, mask


def test_otoc_closed_form_matches_explicit_commutator():
    torch.manual_seed(0)
    dim = 16
    for _ in range(20):
        w, v = _random_state(dim), _random_state(dim)
        phases = torch.randn(dim)
        evolution = torch.diag(torch.complex(torch.cos(phases), torch.sin(phases)))
        evolved_w = evolution @ w
        op_w, op_v = _householder(evolved_w), _householder(v)
        commutator = op_w @ op_v - op_v @ op_w
        explicit = torch.trace(commutator.conj().T @ commutator).real / dim
        overlap = torch.abs(torch.dot(evolved_w.conj(), v)) ** 2
        closed_form = 32.0 / dim * overlap * (1 - overlap)
        assert torch.allclose(explicit, closed_form, atol=1e-4)


def test_rotation_is_unitary():
    omega = init_frequencies(4, 8)
    rotation = make_rotation(torch.rand(2, 5) * 100, omega)
    assert torch.allclose(
        torch.abs(rotation), torch.ones_like(torch.abs(rotation)), atol=1e-5
    )


def test_unit_normalize():
    state = torch.complex(torch.randn(3, 7, 16), torch.randn(3, 7, 16))
    norm = torch.linalg.vector_norm(unit_normalize(state), dim=-1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)


def test_born_probabilities_and_mixture_weights_are_normalized():
    torch.manual_seed(0)
    model = _toy_model().eval()
    features, speakers, mask = _toy_inputs()
    output = model(features, speakers, mask)
    probabilities = output["logp"].exp()
    assert torch.all(probabilities >= 0)
    assert torch.allclose(
        probabilities.sum(-1), torch.ones_like(probabilities.sum(-1)), atol=1e-4
    )
    assert torch.allclose(
        output["alpha"].sum(-1), torch.ones_like(output["alpha"].sum(-1)), atol=1e-5
    )
    assert torch.all(output["purity"] <= 1.0 + 1e-4)
    assert torch.all(output["purity"] > 0)


def test_padding_does_not_change_valid_predictions():
    torch.manual_seed(0)
    model = _toy_model().eval()
    features, speakers, mask = _toy_inputs(batch_size=1, sequence_length=8)
    mask[:, 5:] = False
    original = model(features, speakers, mask)["logp"][:, :5]
    for modality in features:
        features[modality][:, 5:] = 1000.0 * torch.randn_like(features[modality][:, 5:])
    changed = model(features, speakers, mask)["logp"][:, :5]
    assert torch.allclose(original, changed, atol=1e-4)


def test_backward_gradients_are_finite():
    torch.manual_seed(0)
    model = _toy_model().train()
    features, speakers, mask = _toy_inputs()
    output = model(features, speakers, mask)
    target = torch.randint(0, 6, (2, 6))
    loss = torch.nn.functional.nll_loss(output["logp"].reshape(-1, 6), target.reshape(-1))
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


def test_independent_unimodal_heads_do_not_consume_other_modalities():
    torch.manual_seed(0)
    model = _toy_model(unimodal_path="independent").eval()
    features, speakers, mask = _toy_inputs(batch_size=1, sequence_length=5)
    original = model(features, speakers, mask)
    changed_features = {name: value.clone() for name, value in features.items()}
    changed_features["a"] = torch.randn_like(changed_features["a"]) * 100.0
    changed_features["v"] = torch.randn_like(changed_features["v"]) * 100.0
    changed = model(changed_features, speakers, mask)

    assert torch.allclose(original["logp_uni"]["t"], changed["logp_uni"]["t"], atol=1e-6)
    assert not torch.allclose(original["logp"], changed["logp"], atol=1e-5)

    cross_parameter_ids = {id(parameter) for parameter in model.cross.parameters()}
    shared_parameter_ids = {id(parameter) for parameter in model.shared_parameters()}
    assert cross_parameter_ids.isdisjoint(shared_parameter_ids)


def test_missing_modality_values_cannot_affect_masked_model_outputs():
    torch.manual_seed(0)
    model = _toy_model(
        unimodal_path="independent", mask_missing_modalities=True
    ).eval()
    features, speakers, mask = _toy_inputs(batch_size=1, sequence_length=5)
    availability = torch.ones(1, 5, 3, dtype=torch.bool)
    availability[:, 2, 2] = False

    original = model(features, speakers, mask, availability)
    changed_features = {name: value.clone() for name, value in features.items()}
    changed_features["v"][:, 2] = 1000.0 * torch.randn_like(
        changed_features["v"][:, 2]
    )
    changed = model(changed_features, speakers, mask, availability)

    assert torch.allclose(original["logp"], changed["logp"], atol=1e-6)
    assert original["alpha"][0, 2, 2].item() == 0.0
    assert torch.allclose(
        original["alpha"].sum(-1),
        torch.ones_like(original["alpha"].sum(-1)),
        atol=1e-6,
    )
