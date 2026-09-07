"""Evaluate a QESM checkpoint directly on one raw combined feature pickle."""

import argparse
import json
import os
import pickle

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from dataset import build_datasets, collate_dialogues
from qotoc import build_model


RAW_ONLY_FIELDS = ("audio_extra", "text_feats", "visual_feats", "audio_feats")


def load_config(path):
    """Load either a training config object or a ``metrics.json`` snapshot."""
    try:
        with open(path, encoding="utf-8") as config_file:
            payload = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read config {path!r}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("config must contain one JSON object")
    config = payload.get("args", payload)
    if not isinstance(config, dict):
        raise ValueError("metrics.json field 'args' must be a JSON object")
    return dict(config)


def load_state_dict(path):
    """Load a plain or commonly wrapped state dict without executing pickle code."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, pickle.UnpicklingError) as exc:
        raise ValueError(f"cannot read checkpoint {path!r}: {exc}") from exc
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict) or not payload:
        raise ValueError("checkpoint does not contain a non-empty state dict")
    if all(key.startswith("module.") for key in payload):
        payload = {key.removeprefix("module."): value for key, value in payload.items()}
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in payload.items()):
        raise ValueError("checkpoint state dict must map string names to tensors")
    return payload


def _raw_mode_errors(config, state_dict, dataset):
    errors = []
    replacements = [field for field in RAW_ONLY_FIELDS if config.get(field) is not None]
    if config.get("audio_replace"):
        replacements.append("audio_replace")
    if replacements:
        errors.append(
            "config requests external feature replacements: " + ", ".join(replacements)
        )

    modalities = "".join(dict.fromkeys(config.get("modalities") or "tav"))
    checkpoint_modalities = {
        key.split(".")[1]
        for key in state_dict
        if key.startswith("encoders.") and len(key.split(".")) > 2
    }
    if checkpoint_modalities != set(modalities):
        errors.append(
            "checkpoint modalities " + "".join(sorted(checkpoint_modalities))
            + " do not match config modalities " + modalities
        )
    for modality in modalities:
        key = f"encoders.{modality}.input_norm.weight"
        if key not in state_dict:
            errors.append(f"checkpoint is missing {key}")
            continue
        checkpoint_dim = int(state_dict[key].shape[0])
        raw_dim = int(dataset.feat_dims[modality])
        if checkpoint_dim != raw_dim:
            errors.append(
                f"{modality} input: checkpoint={checkpoint_dim}, raw combined={raw_dim}"
            )
        speaker_key = f"encoders.{modality}.spk_emb.weight"
        if speaker_key in state_dict and int(state_dict[speaker_key].shape[1]) != dataset.n_speakers:
            errors.append(
                f"speaker input: checkpoint={int(state_dict[speaker_key].shape[1])}, "
                f"raw combined={dataset.n_speakers}"
            )
    return errors


def _load_checkpoint_strict(model, state_dict):
    model_state = model.state_dict()
    mismatches = [
        (key, tuple(value.shape), tuple(model_state[key].shape))
        for key, value in state_dict.items()
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape)
    ]
    if mismatches:
        detail = "; ".join(
            f"{key}: checkpoint={checkpoint}, model={current}"
            for key, checkpoint, current in mismatches[:8]
        )
        raise ValueError(f"checkpoint tensor shapes do not match the configured model: {detail}")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"checkpoint keys do not match the configured model: {exc}") from exc


def evaluate_checkpoint(checkpoint, config_path, feature_pkl, dataset_name=None, batch_size=8,
                        device="auto", valid_fraction=None, valid_count=None,
                        valid_position=None):
    """Run labelled test inference and return JSON-serialisable metrics."""
    config = load_config(config_path)
    dataset_name = dataset_name or config.get("dataset")
    if dataset_name not in ("iemocap", "meld"):
        raise ValueError("dataset must be 'iemocap' or 'meld' (in --dataset or config)")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")

    if config.get("standardize", True) and not config.get("merge_valid", False):
        recorded_raw_split = config.get("feature_pkl") is not None and (
            "feature_valid_fraction" in config or config.get("feature_valid_count", 0) > 0
        )
        supplied_raw_split = valid_fraction is not None or (
            valid_count is not None and valid_count > 0
        )
        if not recorded_raw_split and not supplied_raw_split:
            raise ValueError(
                "standardized checkpoint config does not record its combined validation split; "
                "use a raw-mode metrics/config file containing the feature split arguments, "
                "or explicitly pass --valid-fraction/--feature-valid-count if the original split "
                "is known"
            )

    fraction = config.get("feature_valid_fraction", 0.1) if valid_fraction is None else valid_fraction
    count = config.get("feature_valid_count", 0) if valid_count is None else valid_count
    position = (
        config.get("feature_valid_position", "head")
        if valid_position is None
        else valid_position
    )
    datasets = build_datasets(
        dataset_name,
        data_dir=None,
        feature_pkl=feature_pkl,
        feature_valid_fraction=fraction,
        feature_valid_count=count,
        feature_valid_position=position,
        standardize=config.get("standardize", True),
        merge_valid=config.get("merge_valid", False),
        text_mode=config.get("text_mode", "mean"),
        sentiment=config.get("sentiment", False),
        missing_aware_standardize=config.get("missing_aware_standardize", False),
    )
    test_dataset = datasets["test"]
    state_dict = load_state_dict(checkpoint)
    raw_errors = _raw_mode_errors(config, state_dict, test_dataset)
    if raw_errors:
        raise ValueError(
            "checkpoint is not compatible with raw combined-only inference: "
            + "; ".join(raw_errors)
            + ". Use a checkpoint trained on the original combined dimensions."
        )

    model = build_model(config, datasets["train"]).to(torch_device)
    _load_checkpoint_strict(model, state_dict)
    model.eval()
    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_dialogues,
    )
    all_true, all_pred = [], []
    all_unimodal_pred = {modality: [] for modality in model.modalities}
    all_unimodal_true = {modality: [] for modality in model.modalities}
    availability_total = np.zeros(len(model.modalities), dtype=np.float64)
    alpha_total = np.zeros(len(model.modalities), dtype=np.float64)
    observed_alpha_total = np.zeros(len(model.modalities), dtype=np.float64)
    observed_alpha_count = np.zeros(len(model.modalities), dtype=np.float64)
    alpha_count = 0
    purity_total = 0.0
    with torch.inference_mode():
        for features, speakers, labels, mask, availability in loader:
            features = {name: value.to(torch_device) for name, value in features.items()}
            speakers = speakers.to(torch_device)
            device_mask = mask.to(torch_device)
            device_availability = availability.to(torch_device)
            output = model(features, speakers, device_mask, device_availability)
            prediction = output["logp"].argmax(-1)
            all_true.append(labels[mask].numpy())
            all_pred.append(prediction[device_mask].cpu().numpy())
            for modality_index, (modality, logp) in enumerate(output["logp_uni"].items()):
                modality_valid = device_mask & output["mod_available"][:, :, modality_index]
                all_unimodal_pred[modality].append(
                    logp.argmax(-1)[modality_valid].cpu().numpy()
                )
                all_unimodal_true[modality].append(labels[modality_valid.cpu()].numpy())
                availability_total[modality_index] += int(modality_valid.sum().item())
            valid_alpha = output["alpha"][device_mask]
            alpha_total += valid_alpha.sum(0).to(
                device="cpu", dtype=torch.float64
            ).numpy()
            alpha_count += int(valid_alpha.size(0))
            for modality_index in range(valid_alpha.size(-1)):
                modality_valid = (
                    device_mask & output["mod_available"][:, :, modality_index]
                )
                observed_alpha_total[modality_index] += float(
                    output["alpha"][:, :, modality_index][modality_valid].sum().item()
                )
                observed_alpha_count[modality_index] += int(
                    modality_valid.sum().item()
                )
            purity_total += float(output["purity"][device_mask].sum().item())
    true = np.concatenate(all_true)
    pred = np.concatenate(all_pred)
    labels = list(range(test_dataset.n_classes))
    per_class = f1_score(
        true, pred, average=None, labels=labels, zero_division=0
    ) * 100
    readout = model.readout
    readout_diagnostics = {}
    if hasattr(readout, "joint_weight"):
        readout_diagnostics["joint_weight"] = float(
            torch.sigmoid(readout.joint_weight.detach()).cpu().item()
        )
    if hasattr(readout, "log_tau"):
        readout_diagnostics["tau"] = float(
            torch.exp(readout.log_tau.detach()).cpu().item()
        )
    if getattr(readout, "coherence", None) is not None:
        readout_diagnostics["coherence"] = float(
            torch.sigmoid(readout.coherence.detach()).cpu().item()
        )

    return {
        "dataset": dataset_name,
        "checkpoint": os.path.abspath(checkpoint),
        "feature_pkl": os.path.abspath(feature_pkl),
        "device": str(torch_device),
        "n_utterances": int(len(true)),
        "weighted_f1": float(
            f1_score(true, pred, average="weighted", labels=labels, zero_division=0) * 100
        ),
        "macro_f1": float(
            f1_score(true, pred, average="macro", labels=labels, zero_division=0) * 100
        ),
        "accuracy": float(accuracy_score(true, pred) * 100),
        "per_class_f1": per_class.tolist(),
        "unimodal_weighted_f1": {
            modality: float(
                f1_score(
                    np.concatenate(all_unimodal_true[modality]),
                    np.concatenate(predictions),
                    average="weighted",
                    labels=labels,
                    zero_division=0,
                ) * 100
            )
            for modality, predictions in all_unimodal_pred.items()
        },
        "unimodal_accuracy": {
            modality: float(
                accuracy_score(
                    np.concatenate(all_unimodal_true[modality]),
                    np.concatenate(predictions),
                ) * 100
            )
            for modality, predictions in all_unimodal_pred.items()
        },
        "fusion_alpha_mean": {
            modality: float(alpha_total[index] / max(alpha_count, 1))
            for index, modality in enumerate(model.modalities)
        },
        "fusion_alpha_when_available": {
            modality: float(
                observed_alpha_total[index] / max(observed_alpha_count[index], 1)
            )
            for index, modality in enumerate(model.modalities)
        },
        "availability_rate": {
            modality: float(availability_total[index] / max(len(true), 1))
            for index, modality in enumerate(model.modalities)
        },
        "mean_purity": float(purity_total / max(alpha_count, 1)),
        "readout": readout_diagnostics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a raw-feature QESM checkpoint on one combined combined pickle"
    )
    parser.add_argument("--checkpoint", required=True, help="raw-feature-trained .pt state dict")
    parser.add_argument("--config", required=True, help="training JSON or metrics.json")
    parser.add_argument("--feature-pkl", required=True, help="combined combined feature pickle")
    parser.add_argument("--dataset", choices=["iemocap", "meld"], default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--valid-fraction",
        type=float,
        default=None,
        help="override feature_valid_fraction from config for normalization reconstruction",
    )
    parser.add_argument(
        "--feature-valid-count",
        type=int,
        default=None,
        help="override feature_valid_count from config; 0 uses the validation fraction",
    )
    parser.add_argument(
        "--feature-valid-position",
        choices=["head", "tail"],
        default=None,
        help="override feature_valid_position from config",
    )
    args = parser.parse_args()
    try:
        result = evaluate_checkpoint(
            args.checkpoint,
            args.config,
            args.feature_pkl,
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            device=args.device,
            valid_fraction=args.valid_fraction,
            valid_count=args.feature_valid_count,
            valid_position=args.feature_valid_position,
        )
    except (ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
