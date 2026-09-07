"""Shared construction of QESM models from JSON-compatible configurations."""

from collections.abc import Mapping

from .model import QOTOCModel


def build_model(config, dataset):
    """Build a :class:`QOTOCModel` from training args or a plain mapping.

    ``dataset`` supplies the feature, speaker, and class dimensions. Keeping
    this factory shared by training and inference prevents checkpoint-loading
    code from drifting away from the architecture used during training.
    """
    if isinstance(config, Mapping):
        values = config
    else:
        values = vars(config)

    def option(name, default):
        value = values.get(name, default)
        return default if value is None else value

    modalities = "".join(dict.fromkeys(option("modalities", "tav")))
    unknown = [modality for modality in modalities if modality not in "tav"]
    if not modalities or unknown:
        raise ValueError("modalities must be a non-empty subset of 'tav'")
    ablate = option("ablate", "")
    if isinstance(ablate, str):
        ablate = tuple(item for item in ablate.split(",") if item)
    else:
        ablate = tuple(ablate)

    return QOTOCModel(
        feat_dims={modality: dataset.feat_dims[modality] for modality in modalities},
        n_speakers=dataset.n_speakers,
        n_classes=dataset.n_classes,
        d_model=option("d_model", 256),
        n_heads=option("n_heads", 8),
        n_temporal_layers=option("temporal_layers", 2),
        n_cross_layers=option("cross_layers", 1),
        ffn_mult=option("ffn_mult", 2),
        input_dropout=option("input_dropout", 0.3),
        attn_dropout=option("attn_dropout", 0.1),
        state_dropout=option("state_dropout", 0.15),
        born_rank=option("born_rank", 32),
        joint_rank=option("joint_rank", 64),
        joint_init=option("joint_init", 0.0),
        joint_freeze=option("joint_freeze", False),
        per_class_tau=option("per_class_tau", False),
        gate_hidden=option("gate_hidden", 128),
        window=option("window", 0),
        modrelu_bias=option("modrelu_bias", 0.0),
        phase_scale=option("phase_scale", 0.1),
        enc_hidden=option("enc_hidden", 0),
        enc_layers=option("enc_layers", 1),
        coherence_init=values.get("coherence_init"),
        interleave=option("interleave", False),
        rope_base=option("rope_base", 10000.0),
        gru_context=option("gru_context", False),
        otoc_poly=option("otoc_poly", False),
        otoc_ops=option("otoc_ops", 1),
        ablate=ablate,
        rel_buckets=option("rel_buckets", 16),
        gate_text_bias=option("gate_text_bias", 0.0),
        graph_branch=option("graph_branch", False),
        d_graph=option("d_graph", 256),
        graph_layers=option("graph_layers", 2),
        graph_window=option("graph_window", 10),
        graph_combine=option("graph_combine", "sum"),
        temporal_model=option("temporal_model", "quantum"),
        cross_model=option("cross_model", "quantum"),
        readout=option("readout", "born"),
        classical_param_match=option("classical_param_match", "arch"),
        mlp_readout_hidden=option("mlp_readout_hidden", 0),
        relation_distill=option("relation_distill", option("lambda_rel", 0.0) > 0),
        unimodal_path=option("unimodal_path", "post-cross"),
        mask_missing_modalities=option("mask_missing_modalities", False),
    )
