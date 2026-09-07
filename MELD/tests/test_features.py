"""combined single-pickle loading and checkpoint-inference tests."""

import json
import pickle

import numpy as np
import pytest
import torch

import evaluate as evaluate_module
from dataset import build_datasets, load_combined_pickle
from evaluate import evaluate_checkpoint
from qotoc import build_model


def _dialogue_arrays(keys, audio_dim, meld=False):
    video_ids, speakers, labels = {}, {}, {}
    text_layers = [{}, {}, {}, {}]
    audio, visual, sentences = {}, {}, {}
    for index, key in enumerate(keys):
        length = 2 + index % 2
        video_ids[key] = [f"{key}_{utterance}" for utterance in range(length)]
        if meld:
            speaker = np.zeros((length, 9), dtype=np.float32)
            speaker[:, index % 9] = 1.0
            speakers[key] = speaker
            labels[key] = [(index + utterance) % 7 for utterance in range(length)]
        else:
            speakers[key] = ["M" if utterance % 2 == 0 else "F" for utterance in range(length)]
            labels[key] = [(index + utterance) % 6 for utterance in range(length)]
        for layer, mapping in enumerate(text_layers, start=1):
            mapping[key] = np.full((length, 1024), layer + index, dtype=np.float32)
        audio[key] = np.full((length, audio_dim), index + 1, dtype=np.float32)
        visual[key] = np.full((length, 342), index + 2, dtype=np.float32)
        sentences[key] = [f"sentence {utterance}" for utterance in range(length)]
    return video_ids, speakers, labels, text_layers, audio, visual, sentences


def _write_iemocap(path):
    train = ["Ses01F_a", "Ses02F_a", "Ses03F_a"]
    test = ["Ses05F_a"]
    ids, speakers, labels, text, audio, visual, sentences = _dialogue_arrays(
        train + test, 1582
    )
    payload = [ids, speakers, labels, *text, audio, visual, sentences, train, test]
    with path.open("wb") as feature_file:
        pickle.dump(payload, feature_file)
    return train, test


def _write_meld(path, include_sentiment=False):
    train, test = {2, 0, 1}, {3}
    keys = [0, 1, 2, 3]
    ids, speakers, labels, text, audio, visual, sentences = _dialogue_arrays(
        keys, 300, meld=True
    )
    if include_sentiment:
        sentiments = {key: [value % 3 for value in labels[key]] for key in keys}
        payload = [
            ids, speakers, labels, sentiments, *text, audio, visual, sentences, train, test, None
        ]
    else:
        payload = [ids, speakers, labels, *text, audio, visual, sentences, train, test, None]
    with path.open("wb") as feature_file:
        pickle.dump(payload, feature_file)


def _tiny_config():
    return {
        "dataset": "iemocap",
        "d_model": 16,
        "n_heads": 4,
        "temporal_layers": 1,
        "cross_layers": 1,
        "ffn_mult": 2,
        "born_rank": 4,
        "joint_rank": 8,
        "gate_hidden": 8,
        "input_dropout": 0.0,
        "attn_dropout": 0.0,
        "state_dropout": 0.0,
        "standardize": False,
        "feature_valid_fraction": 0.34,
        "modalities": "tav",
    }


def test_iemocap_combined_pickle_is_the_only_required_data_file(tmp_path):
    feature_path = tmp_path / "iemocap.pkl"
    train, test = _write_iemocap(feature_path)
    datasets = build_datasets(
        "iemocap",
        data_dir=None,
        feature_pkl=feature_path,
        feature_valid_fraction=0.34,
        standardize=False,
    )

    assert datasets["valid"].keys == train[:1]
    assert datasets["train"].keys == train[1:]
    assert datasets["test"].keys == test
    text, audio, visual, speakers, labels, availability = datasets["test"][0]
    assert text.shape == (3, 1024)
    assert np.all(text == 5.5)  # mean of layers (1..4) plus dialogue index 3
    assert audio.shape == (3, 1582)
    assert visual.shape == (3, 342)
    assert speakers.shape == (3, 2)
    assert labels.dtype == np.int64
    assert availability.shape == (3, 3)
    assert availability.all()


def test_tail_exact_validation_count_overrides_fraction_without_padding(tmp_path):
    feature_path = tmp_path / "iemocap_tail.pkl"
    train, test = _write_iemocap(feature_path)
    datasets = build_datasets(
        "iemocap",
        data_dir=None,
        feature_pkl=feature_path,
        feature_valid_fraction=0.67,
        feature_valid_count=1,
        feature_valid_position="tail",
        standardize=False,
    )

    assert datasets["train"].keys == train[:-1]
    assert datasets["valid"].keys == train[-1:]
    assert datasets["test"].keys == test
    assert datasets["train"].feat_dims == {"t": 1024, "a": 1582, "v": 342}
    text, audio, visual, _, _, availability = datasets["valid"][0]
    assert text.shape == (2, 1024)
    assert audio.shape == (2, 1582)
    assert visual.shape == (2, 342)
    assert np.count_nonzero(text) == text.size
    assert np.count_nonzero(audio) == audio.size
    assert np.count_nonzero(visual) == visual.size
    assert availability.all()


@pytest.mark.parametrize("invalid_count", [-1, 3, 4])
def test_invalid_exact_validation_count_is_rejected(tmp_path, invalid_count):
    feature_path = tmp_path / "iemocap_invalid_count.pkl"
    _write_iemocap(feature_path)
    with pytest.raises(ValueError, match="feature_valid_count"):
        build_datasets(
            "iemocap",
            data_dir=None,
            feature_pkl=feature_path,
            feature_valid_count=invalid_count,
            standardize=False,
        )


@pytest.mark.parametrize("include_sentiment,arity", [(False, 13), (True, 14)])
def test_meld_combined_schemas_and_set_ids_are_supported(tmp_path, include_sentiment, arity):
    feature_path = tmp_path / f"meld_{arity}.pkl"
    _write_meld(feature_path, include_sentiment=include_sentiment)
    features = load_combined_pickle(feature_path, "meld")
    assert features["train_ids"] == [0, 1, 2]
    assert features["feat_dims"] == {"t": 1024, "a": 300, "v": 342}
    if include_sentiment:
        assert features["sentiments"] is not None
    else:
        with pytest.raises(ValueError, match="does not contain native sentiment"):
            build_datasets(
                "meld",
                data_dir=None,
                feature_pkl=feature_path,
                feature_valid_fraction=0.34,
                standardize=False,
                sentiment=True,
            )


def test_raw_pickle_mode_rejects_external_feature_replacements(tmp_path):
    feature_path = tmp_path / "iemocap.pkl"
    _write_iemocap(feature_path)
    with pytest.raises(ValueError, match="forbids external feature replacements"):
        build_datasets(
            "iemocap",
            data_dir=None,
            feature_pkl=feature_path,
            feature_valid_fraction=0.34,
            standardize=False,
            audio_feats="replacement.pkl",
        )


def test_raw_pickle_mode_rejects_non_native_text_concatenation(tmp_path):
    feature_path = tmp_path / "iemocap.pkl"
    _write_iemocap(feature_path)
    with pytest.raises(ValueError, match="native 1024-D"):
        build_datasets(
            "iemocap",
            data_dir=None,
            feature_pkl=feature_path,
            feature_valid_fraction=0.34,
            standardize=False,
            text_mode="concat",
        )


def test_missing_aware_standardization_excludes_and_preserves_zero_rows(tmp_path):
    feature_path = tmp_path / "meld_missing.pkl"
    _write_meld(feature_path)
    with feature_path.open("rb") as feature_file:
        payload = pickle.load(feature_file, encoding="latin1")
    audio, visual = payload[7], payload[8]
    audio[0][:] = 0.0
    visual[0][:] = 0.0
    for key in (1, 2):
        audio[key][:] = 5.0
        visual[key][:] = 5.0
    with feature_path.open("wb") as feature_file:
        pickle.dump(payload, feature_file)

    aware = build_datasets(
        "meld",
        data_dir=None,
        feature_pkl=feature_path,
        standardize=True,
        merge_valid=True,
        missing_aware_standardize=True,
    )
    _, missing_audio, missing_visual, _, _, availability = aware["train"][0]
    assert np.all(missing_audio == 0.0)
    assert np.all(missing_visual == 0.0)
    assert np.allclose(aware["train"].std.stats[1][0], 5.0)
    assert np.allclose(aware["train"].std.stats[2][0], 5.0)
    assert availability[:, 0].all()
    assert not availability[:, 1].any()
    assert not availability[:, 2].any()

    legacy = build_datasets(
        "meld",
        data_dir=None,
        feature_pkl=feature_path,
        standardize=True,
        merge_valid=True,
    )
    _, legacy_audio, legacy_visual, _, _, legacy_availability = legacy["train"][0]
    assert np.any(legacy_audio != 0.0)
    assert np.any(legacy_visual != 0.0)
    assert not legacy_availability[:, 1].any()
    assert not legacy_availability[:, 2].any()


def test_training_data_sources_are_mutually_exclusive(tmp_path):
    feature_path = tmp_path / "iemocap.pkl"
    _write_iemocap(feature_path)
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_datasets(
            "iemocap",
            data_dir="legacy-data",
            feature_pkl=feature_path,
            feature_valid_fraction=0.34,
            standardize=False,
        )


def test_checkpoint_evaluation_reconstructs_recorded_exact_tail_split(tmp_path, monkeypatch):
    feature_path = tmp_path / "iemocap.pkl"
    _write_iemocap(feature_path)
    config = _tiny_config()
    config["feature_valid_count"] = 1
    config["feature_valid_position"] = "tail"
    datasets = build_datasets(
        "iemocap",
        data_dir=None,
        feature_pkl=feature_path,
        feature_valid_fraction=config["feature_valid_fraction"],
        feature_valid_count=config["feature_valid_count"],
        feature_valid_position=config["feature_valid_position"],
        standardize=False,
    )
    model = build_model(config, datasets["train"])
    checkpoint_path = tmp_path / "raw.pt"
    torch.save(model.state_dict(), checkpoint_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    observed_split = {}
    real_build_datasets = evaluate_module.build_datasets

    def recording_build_datasets(*args, **kwargs):
        observed_split["count"] = kwargs["feature_valid_count"]
        observed_split["position"] = kwargs["feature_valid_position"]
        return real_build_datasets(*args, **kwargs)

    monkeypatch.setattr(evaluate_module, "build_datasets", recording_build_datasets)
    result = evaluate_module.evaluate_checkpoint(
        checkpoint_path,
        config_path,
        feature_path,
        batch_size=1,
        device="cpu",
    )
    assert observed_split == {"count": 1, "position": "tail"}
    assert result["dataset"] == "iemocap"
    assert result["n_utterances"] == 3
    for metric in ("weighted_f1", "macro_f1", "accuracy"):
        assert 0.0 <= result[metric] <= 100.0


def test_checkpoint_dimension_mismatch_fails_before_forward(tmp_path):
    feature_path = tmp_path / "iemocap.pkl"
    _write_iemocap(feature_path)
    config = _tiny_config()
    datasets = build_datasets(
        "iemocap",
        data_dir=None,
        feature_pkl=feature_path,
        feature_valid_fraction=config["feature_valid_fraction"],
        standardize=False,
    )
    state = build_model(config, datasets["train"]).state_dict()
    state["encoders.a.input_norm.weight"] = torch.zeros(2606)
    checkpoint_path = tmp_path / "enhanced.pt"
    torch.save(state, checkpoint_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint=2606, raw combined=1582"):
        evaluate_checkpoint(checkpoint_path, config_path, feature_path, device="cpu")


def test_standardized_legacy_config_cannot_guess_feature_split(tmp_path):
    config = _tiny_config()
    config["standardize"] = True
    config_path = tmp_path / "legacy_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="does not record its combined validation split"):
        evaluate_checkpoint(
            tmp_path / "not-read.pt",
            config_path,
            tmp_path / "not-read.pkl",
            device="cpu",
        )
