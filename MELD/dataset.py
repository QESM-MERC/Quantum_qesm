"""Dialogue-level datasets for IEMOCAP / MELD feature pickles.

The preferred input is the single combined pickle released with combined. The
legacy split M3Net files remain supported for old experiment configurations.

Raw combined feature protocol:
  IEMOCAP: text = mean of 4 RoBERTa-large layers (1024), audio = openSMILE (1582),
           visual = DenseNet (342); 6 classes; 120 train-pool / 31 test dialogues.
  MELD:    text = mean of 4 RoBERTa-large layers (1024), audio (300), visual (342);
           7 classes; 1152 train-pool / 280 test dialogues.
"""
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


def _text_feat(r1, r2, r3, r4, vid, mode="mean"):
    if mode == "r1":
        # the SDT/M3Net-family 'videoText': RoBERTa last layer alone
        return np.asarray(r1[vid], dtype=np.float32)
    rs = [np.asarray(r[vid], dtype=np.float32) for r in (r1, r2, r3, r4)]
    if mode == "concat":
        return np.concatenate(rs, axis=-1)
    return sum(rs) / 4.0


# Emotion->sentiment merge (combined scheme). IEMOCAP is DERIVED from emotion; MELD uses NATIVE
# sentiment (data/meld_sentiment.pkl). Sentiment idx: negative=0, neutral=1, positive=2.
# IEMOCAP order [happy,sad,neutral,angry,excited,frustrated]: sad/angry/frustrated->negative,
# happy/excited->positive, neutral->neutral.
IE_EMO2SENT = {0: 2, 1: 0, 2: 1, 3: 0, 4: 2, 5: 0}

RAW_FEATURE_DIMS = {
    "iemocap": {"t": 1024, "a": 1582, "v": 342},
    "meld": {"t": 1024, "a": 300, "v": 342},
}


def _ordered_ids(values):
    """Return a deterministic list while preserving meaningful pickle order."""
    if isinstance(values, (set, frozenset)):
        return sorted(values)
    return list(values)


def load_combined_pickle(path, name):
    """Load and validate one combined combined feature pickle.

    IEMOCAP uses the public 12-field schema. MELD releases in circulation use
    either 13 fields (emotion labels only) or 14 fields (native sentiment labels
    included); both are accepted. Pickle is code-executing input, so callers
    must only pass files obtained from a trusted source.
    """
    if name not in RAW_FEATURE_DIMS:
        raise ValueError(f"unsupported dataset {name!r}; expected 'iemocap' or 'meld'")
    try:
        with open(path, "rb") as feature_file:
            payload = pickle.load(feature_file, encoding="latin1")
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError,
            IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot read combined pickle {path!r}: {exc}") from exc

    if not isinstance(payload, (list, tuple)):
        raise ValueError(
            f"invalid combined {name} pickle: expected a list/tuple, got {type(payload).__name__}"
        )

    sentiments = None
    if name == "iemocap":
        if len(payload) != 12:
            raise ValueError(
                f"invalid combined IEMOCAP schema: expected 12 fields, got {len(payload)}"
            )
        (video_ids, speakers, labels, r1, r2, r3, r4, audio, visual, sentences,
         train_ids, test_ids) = payload
        n_speakers, n_classes = 2, 6
    else:
        if len(payload) == 13:
            (video_ids, speakers, labels, r1, r2, r3, r4, audio, visual, sentences,
             train_ids, test_ids, _) = payload
        elif len(payload) == 14:
            (video_ids, speakers, labels, sentiments, r1, r2, r3, r4, audio, visual,
             sentences, train_ids, test_ids, _) = payload
        else:
            raise ValueError(
                f"invalid combined MELD schema: expected 13 or 14 fields, got {len(payload)}"
            )
        n_speakers, n_classes = 9, 7

    features = {
        "video_ids": video_ids,
        "speakers": speakers,
        "labels": labels,
        "sentiments": sentiments,
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "r4": r4,
        "audio": audio,
        "visual": visual,
        "sentences": sentences,
        "train_ids": _ordered_ids(train_ids),
        "test_ids": _ordered_ids(test_ids),
        "n_speakers": n_speakers,
        "n_classes": n_classes,
        "feat_dims": dict(RAW_FEATURE_DIMS[name]),
        "name": name,
    }
    _validate_combined_features(features)
    return features


def _validate_combined_features(features):
    """Validate IDs, sequence lengths, labels, speakers, and raw dimensions."""
    name = features["name"]
    train_ids, test_ids = features["train_ids"], features["test_ids"]
    if not train_ids or not test_ids:
        raise ValueError(f"combined {name} pickle must contain non-empty train and test IDs")
    overlap = set(train_ids).intersection(test_ids)
    if overlap:
        raise ValueError(f"combined {name} train/test IDs overlap (example: {next(iter(overlap))!r})")

    mappings = ("speakers", "labels", "r1", "r2", "r3", "r4", "audio", "visual")
    expected_dims = features["feat_dims"]
    for vid in train_ids + test_ids:
        missing = [field for field in mappings if vid not in features[field]]
        if missing:
            raise ValueError(f"combined {name} dialogue {vid!r} is missing fields: {', '.join(missing)}")
        labels = np.asarray(features["labels"][vid])
        length = len(labels)
        if length == 0:
            raise ValueError(f"combined {name} dialogue {vid!r} has no utterances")
        arrays = {
            "r1": np.asarray(features["r1"][vid]),
            "r2": np.asarray(features["r2"][vid]),
            "r3": np.asarray(features["r3"][vid]),
            "r4": np.asarray(features["r4"][vid]),
            "audio": np.asarray(features["audio"][vid]),
            "visual": np.asarray(features["visual"][vid]),
            "speakers": np.asarray(features["speakers"][vid]),
        }
        wrong_lengths = [field for field, value in arrays.items() if len(value) != length]
        if wrong_lengths:
            raise ValueError(
                f"combined {name} dialogue {vid!r} has sequence-length mismatch in "
                f"{', '.join(wrong_lengths)}"
            )
        for field in ("r1", "r2", "r3", "r4"):
            if arrays[field].ndim != 2 or arrays[field].shape[1] != 1024:
                raise ValueError(
                    f"combined {name} {field} for {vid!r} must have shape (T, 1024), "
                    f"got {arrays[field].shape}"
                )
        for field, modality in (("audio", "a"), ("visual", "v")):
            expected = expected_dims[modality]
            if arrays[field].ndim != 2 or arrays[field].shape[1] != expected:
                raise ValueError(
                    f"combined {name} {field} for {vid!r} must have shape (T, {expected}), "
                    f"got {arrays[field].shape}"
                )
        if labels.min() < 0 or labels.max() >= features["n_classes"]:
            raise ValueError(f"combined {name} dialogue {vid!r} contains an out-of-range label")
        if name == "iemocap":
            if arrays["speakers"].ndim != 1:
                raise ValueError(f"combined IEMOCAP speakers for {vid!r} must be a 1-D M/F sequence")
            unknown_speakers = set(arrays["speakers"].tolist()) - {"M", "F"}
            if unknown_speakers:
                raise ValueError(
                    f"combined IEMOCAP dialogue {vid!r} contains unknown speakers: "
                    f"{sorted(unknown_speakers)!r}"
                )
        elif arrays["speakers"].ndim != 2 or arrays["speakers"].shape[1] != 9:
            raise ValueError(
                f"combined MELD speakers for {vid!r} must have shape (T, 9), "
                f"got {arrays['speakers'].shape}"
            )


class CombinedFeatureDataset(Dataset):
    """A dialogue view over one already-loaded combined combined pickle."""

    def __init__(self, features, keys, text_mode="mean", sentiment=False):
        if text_mode not in ("mean", "r1"):
            raise ValueError(
                "raw combined mode requires a native 1024-D text representation; "
                "text_mode must be 'mean' or 'r1'"
            )
        self.features = features
        self.keys = list(keys)
        self.text_mode = text_mode
        self.sentiment = sentiment
        self.n_speakers = features["n_speakers"]
        self.n_classes = 3 if sentiment else features["n_classes"]
        self.feat_dims = dict(features["feat_dims"])
        if sentiment and features["name"] == "meld" and features["sentiments"] is None:
            raise ValueError("this combined MELD pickle does not contain native sentiment labels")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        vid = self.keys[idx]
        features = self.features
        text = _text_feat(
            features["r1"], features["r2"], features["r3"], features["r4"], vid,
            self.text_mode,
        )
        audio = np.asarray(features["audio"][vid], dtype=np.float32)
        visual = np.asarray(features["visual"][vid], dtype=np.float32)
        if features["name"] == "iemocap":
            spk = np.asarray(
                [[1, 0] if speaker == "M" else [0, 1] for speaker in features["speakers"][vid]],
                dtype=np.float32,
            )
        else:
            spk = np.asarray(features["speakers"][vid], dtype=np.float32)
        if self.sentiment and features["name"] == "iemocap":
            labels = np.asarray(
                [IE_EMO2SENT[int(x)] for x in features["labels"][vid]], dtype=np.int64
            )
        elif self.sentiment:
            labels = np.asarray(features["sentiments"][vid], dtype=np.int64)
        else:
            labels = np.asarray(features["labels"][vid], dtype=np.int64)
        availability = np.stack(
            [
                np.any(text != 0.0, axis=1),
                np.any(audio != 0.0, axis=1),
                np.any(visual != 0.0, axis=1),
            ],
            axis=-1,
        )
        return text, audio, visual, spk, labels, availability


class IEMOCAPDataset(Dataset):
    n_classes = 6
    n_speakers = 2
    feat_dims = {"t": 1024, "a": 1582, "v": 342}

    def __init__(self, base_path, roberta_path, split="train", text_mode="mean", audio_extra=None,
                 audio_replace=False, text_feats=None, visual_feats=None, audio_feats=None,
                 sentiment=False, keys=None):
        self.sentiment = sentiment
        if sentiment:
            self.n_classes = 3
        self.text_mode = text_mode
        self.audio_extra = audio_extra  # dict utterance-id -> np.ndarray, appended to audio
        self.audio_replace = audio_replace  # use ONLY the extra features as the audio modality
        self.text_feats = text_feats  # dict vid -> (n_utt, dim): replaces the RoBERTa text feature
        self.visual_feats = visual_feats  # dict vid -> (n_utt, dim): replaces the DenseNet visual feature
        self.audio_feats = audio_feats  # dict vid -> (n_utt, dim): replaces the openSMILE audio feature
        self.feat_dims = dict(IEMOCAPDataset.feat_dims)
        if text_mode == "concat":
            self.feat_dims["t"] = 4096
        if text_feats is not None:
            self.feat_dims["t"] = next(iter(text_feats.values())).shape[-1]
        if visual_feats is not None:
            self.feat_dims["v"] = next(iter(visual_feats.values())).shape[-1]
        if audio_feats is not None:
            self.feat_dims["a"] = next(iter(audio_feats.values())).shape[-1]
        with open(base_path, "rb") as f:
            (self.ids, self.speakers, self.labels, _, self.audio, self.visual, _, _, _) = pickle.load(
                f, encoding="latin1"
            )
        if audio_extra is not None:
            extra_dim = next(iter(audio_extra.values())).shape[-1]
            self.feat_dims["a"] = extra_dim if audio_replace else self.feat_dims["a"] + extra_dim
        with open(roberta_path, "rb") as f:
            (_, _, self.r1, self.r2, self.r3, self.r4, _, train_vid, test_vid, valid_vid) = pickle.load(
                f, encoding="latin1"
            )
        if keys is None:
            if split == "train_full":
                keys = list(train_vid) + list(valid_vid)
            elif split == "all":
                keys = list(train_vid) + list(valid_vid) + list(test_vid)
            else:
                keys = {"train": list(train_vid), "valid": list(valid_vid), "test": list(test_vid)}[split]
        self.keys = list(keys)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        vid = self.keys[idx]
        if self.text_feats is not None:
            text = np.asarray(self.text_feats[vid], dtype=np.float32)
        else:
            text = _text_feat(self.r1, self.r2, self.r3, self.r4, vid, self.text_mode)
        if self.audio_feats is not None:
            audio = np.asarray(self.audio_feats[vid], dtype=np.float32)
        else:
            audio = np.asarray(self.audio[vid], dtype=np.float32)
        if self.audio_extra is not None:
            extra_dim = next(iter(self.audio_extra.values())).shape[-1]
            extra = np.stack(
                [
                    np.asarray(self.audio_extra.get(uid, np.zeros(extra_dim)), dtype=np.float32)
                    for uid in self.ids[vid]
                ]
            )
            audio = extra if self.audio_replace else np.concatenate([audio, extra], axis=-1)
        if self.visual_feats is not None:
            visual = np.asarray(self.visual_feats[vid], dtype=np.float32)
        else:
            visual = np.asarray(self.visual[vid], dtype=np.float32)
        spk = np.asarray([[1, 0] if s == "M" else [0, 1] for s in self.speakers[vid]], dtype=np.float32)
        if self.sentiment:
            labels = np.asarray([IE_EMO2SENT[int(x)] for x in self.labels[vid]], dtype=np.int64)
        else:
            labels = np.asarray(self.labels[vid], dtype=np.int64)
        return text, audio, visual, spk, labels


class MELDDataset(Dataset):
    n_classes = 7
    n_speakers = 9
    feat_dims = {"t": 1024, "a": 300, "v": 342}

    def __init__(self, base_path, roberta_path, split="train", text_mode="mean", text_feats=None,
                 visual_feats=None, audio_feats=None, sentiment=False):
        self.sentiment = sentiment
        if sentiment:
            self.n_classes = 3
            with open(os.path.join(os.path.dirname(__file__), "data", "meld_sentiment.pkl"), "rb") as _f:
                self.sent = pickle.load(_f)
        self.text_mode = text_mode
        self.text_feats = text_feats  # dict vid -> (n_utt, dim): replaces the RoBERTa text feature
        self.visual_feats = visual_feats  # dict vid -> (n_utt, dim): replaces the DenseNet visual feature
        self.audio_feats = audio_feats  # dict vid -> (n_utt, dim): replaces the openSMILE audio feature
        self.feat_dims = dict(MELDDataset.feat_dims)
        if text_mode == "concat":
            self.feat_dims["t"] = 4096
        if text_feats is not None:
            self.feat_dims["t"] = next(iter(text_feats.values())).shape[-1]
        if visual_feats is not None:
            self.feat_dims["v"] = next(iter(visual_feats.values())).shape[-1]
        if audio_feats is not None:
            self.feat_dims["a"] = next(iter(audio_feats.values())).shape[-1]
        with open(base_path, "rb") as f:
            (_, self.speakers, self.labels, _, self.audio, self.visual, _, _, _, _) = pickle.load(
                f, encoding="latin1"
            )
        with open(roberta_path, "rb") as f:
            (_, _, _, self.r1, self.r2, self.r3, self.r4, _, train_ids, test_ids, valid_ids) = pickle.load(
                f, encoding="latin1"
            )
        if split == "train_full":
            keys = list(train_ids) + list(valid_ids)
        else:
            keys = {"train": list(train_ids), "valid": list(valid_ids), "test": list(test_ids)}[split]
        self.keys = keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        vid = self.keys[idx]
        if self.text_feats is not None:
            text = np.asarray(self.text_feats[vid], dtype=np.float32)
        else:
            text = _text_feat(self.r1, self.r2, self.r3, self.r4, vid, self.text_mode)
        if self.audio_feats is not None:
            audio = np.asarray(self.audio_feats[vid], dtype=np.float32)
        else:
            audio = np.asarray(self.audio[vid], dtype=np.float32)
        if self.visual_feats is not None:
            visual = np.asarray(self.visual_feats[vid], dtype=np.float32)
        else:
            visual = np.asarray(self.visual[vid], dtype=np.float32)
        spk = np.asarray(self.speakers[vid], dtype=np.float32)
        if self.sentiment:
            labels = np.asarray(self.sent[vid], dtype=np.int64)
        else:
            labels = np.asarray(self.labels[vid], dtype=np.int64)
        return text, audio, visual, spk, labels


def collate_dialogues(batch):
    """Pad a list of variable-length dialogues (text,audio,visual,spk,labels) into batch tensors."""
    B = len(batch)
    T = max(item[0].shape[0] for item in batch)
    d_t, d_a, d_v = (batch[0][i].shape[1] for i in range(3))
    n_spk = batch[0][3].shape[1]

    text = torch.zeros(B, T, d_t)
    audio = torch.zeros(B, T, d_a)
    visual = torch.zeros(B, T, d_v)
    spk = torch.zeros(B, T, n_spk)
    labels = torch.full((B, T), -100, dtype=torch.long)
    mask = torch.zeros(B, T, dtype=torch.bool)
    availability = torch.zeros(B, T, 3, dtype=torch.bool)

    for b, item in enumerate(batch):
        tx, au, vi, sp, lb = item[0], item[1], item[2], item[3], item[4]
        L = tx.shape[0]
        text[b, :L] = torch.from_numpy(tx)
        audio[b, :L] = torch.from_numpy(au)
        visual[b, :L] = torch.from_numpy(vi)
        spk[b, :L] = torch.from_numpy(sp)
        labels[b, :L] = torch.from_numpy(lb)
        mask[b, :L] = True
        if len(item) > 5:
            observed = np.asarray(item[5], dtype=np.bool_)
            if observed.shape != (L, 3):
                raise ValueError(
                    f"availability must have shape ({L}, 3), got {observed.shape}"
                )
            availability[b, :L] = torch.from_numpy(observed)
        else:
            availability[b, :L] = True

    feats = {"t": text, "a": audio, "v": visual}
    return feats, spk, labels, mask, availability


class Standardizer:
    """Per-feature z-score fitted on the training split (audio/visual only).

    ``missing_aware`` is intentionally opt-in. Some combined pickles encode a
    missing modality as an all-zero utterance. Treating those sentinels as
    observations biases the fitted moments and turns them into large non-zero
    vectors after z-scoring. In missing-aware mode, zero rows are excluded from
    the fitted moments and remain zero after transformation.
    """

    def __init__(self, train_ds, indices=(1, 2), missing_aware=False):
        self.indices = indices
        self.missing_aware = missing_aware
        self.stats = {}
        for i in indices:
            feats = np.concatenate([train_ds[k][i] for k in range(len(train_ds))], axis=0)
            fit_feats = feats
            if missing_aware:
                observed = np.any(feats != 0.0, axis=1)
                fit_feats = feats[observed]
            if len(fit_feats) == 0:
                mu = np.zeros(feats.shape[1], dtype=np.float32)
                sd = np.ones(feats.shape[1], dtype=np.float32)
            else:
                mu = fit_feats.mean(axis=0)
                sd = fit_feats.std(axis=0)
            sd[sd < 1e-6] = 1.0
            self.stats[i] = (mu.astype(np.float32), sd.astype(np.float32))

    def __call__(self, item):
        item = list(item)
        for i in self.indices:
            mu, sd = self.stats[i]
            feats = item[i]
            transformed = (feats - mu) / sd
            if self.missing_aware:
                missing = np.all(feats == 0.0, axis=1)
                transformed[missing] = 0.0
            item[i] = transformed
        return tuple(item)


class StandardizedView(Dataset):
    def __init__(self, ds, std):
        self.ds, self.std = ds, std
        self.n_classes, self.n_speakers = ds.n_classes, ds.n_speakers
        self.feat_dims = dict(ds.feat_dims)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.std(self.ds[idx])


class CachedDataset(Dataset):
    """Materialise a dataset once to avoid repeated pickle-backed transforms per epoch."""

    def __init__(self, ds):
        self.items = [ds[i] for i in range(len(ds))]
        self.n_classes, self.n_speakers = ds.n_classes, ds.n_speakers
        self.feat_dims = dict(ds.feat_dims)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def load_audio_extra(path):
    """Load utterance-id -> feature dict (emotion2vec etc.) from .pkl or a dir of .npy."""
    import os
    if path is None:
        return None
    if os.path.isdir(path):
        out = {}
        for fn in os.listdir(path):
            if fn.endswith(".npy"):
                out[fn[:-4]] = np.load(os.path.join(path, fn))
        return out
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def build_combined_datasets(name, pkl_path, standardize=True, merge_valid=False, text_mode="mean",
                          valid_fraction=0.1, sentiment=False,
                          missing_aware_standardize=False, valid_count=0,
                          valid_position="head"):
    """Build train/valid/test views directly from one raw combined pickle.

    combined stores a train pool and test split but no validation IDs. Following
    its released loaders, validation defaults to the first
    ``int(valid_fraction * N)`` train-pool IDs. A positive ``valid_count``
    overrides the fractional size, and ``valid_position='tail'`` selects the
    final IDs instead. Set ``merge_valid`` to use the full train pool for
    training and the test split for selection.
    """
    if isinstance(valid_count, bool) or not isinstance(valid_count, (int, np.integer)):
        raise ValueError("feature_valid_count must be an integer")
    valid_count = int(valid_count)
    if valid_count < 0:
        raise ValueError("feature_valid_count must be >= 0")
    if valid_position not in ("head", "tail"):
        raise ValueError("feature_valid_position must be 'head' or 'tail'")
    features = load_combined_pickle(pkl_path, name)
    pool = features["train_ids"]
    if valid_count:
        if valid_count >= len(pool):
            raise ValueError(
                "feature_valid_count must be smaller than the combined train-pool size "
                f"({len(pool)})"
            )
        n_valid = valid_count
    else:
        if not 0.0 <= valid_fraction < 1.0:
            raise ValueError("feature_valid_fraction must satisfy 0 <= value < 1")
        n_valid = int(valid_fraction * len(pool))
    if not merge_valid and n_valid == 0:
        raise ValueError(
            "feature_valid_fraction creates an empty validation split; increase it or use merge_valid"
        )
    if merge_valid:
        train_ids = pool
        valid_ids = features["test_ids"]
    elif valid_position == "tail":
        train_ids = pool[:-n_valid]
        valid_ids = pool[-n_valid:]
    else:
        train_ids = pool[n_valid:]
        valid_ids = pool[:n_valid]
    split_ids = {"train": train_ids, "valid": valid_ids, "test": features["test_ids"]}
    sets = {
        split: CombinedFeatureDataset(features, keys, text_mode=text_mode, sentiment=sentiment)
        for split, keys in split_ids.items()
    }
    if standardize:
        std = Standardizer(sets["train"], missing_aware=missing_aware_standardize)
        sets = {split: StandardizedView(dataset, std) for split, dataset in sets.items()}
    return sets


def build_datasets(name, data_dir=None, standardize=True, merge_valid=False, text_mode="mean",
                   audio_extra=None,
                   audio_replace=False, text_feats=None, visual_feats=None, audio_feats=None,
                   sentiment=False, loso_test_session=0, loso_valid_session=0,
                   loso_internal_valid_frac=0.0, loso_split_seed=1, feature_pkl=None,
                   feature_valid_fraction=0.1, missing_aware_standardize=False,
                   feature_valid_count=0, feature_valid_position="head"):
    if feature_pkl is not None:
        if data_dir is not None:
            raise ValueError("feature_pkl and data_dir are mutually exclusive")
        replacements = {
            "audio_extra": audio_extra,
            "text_feats": text_feats,
            "visual_feats": visual_feats,
            "audio_feats": audio_feats,
        }
        active = [key for key, value in replacements.items() if value is not None]
        if audio_replace:
            active.append("audio_replace")
        if active:
            raise ValueError(
                "combined raw-pickle mode forbids external feature replacements: " + ", ".join(active)
            )
        if loso_test_session or loso_valid_session or loso_internal_valid_frac:
            raise ValueError("combined raw-pickle mode currently supports the fixed train/test split only")
        return build_combined_datasets(
            name,
            feature_pkl,
            standardize=standardize,
            merge_valid=merge_valid,
            text_mode=text_mode,
            valid_fraction=feature_valid_fraction,
            sentiment=sentiment,
            missing_aware_standardize=missing_aware_standardize,
            valid_count=feature_valid_count,
            valid_position=feature_valid_position,
        )

    if missing_aware_standardize:
        raise ValueError("missing-aware standardization is only supported with feature_pkl")

    if data_dir is None:
        raise ValueError("provide either feature_pkl or data_dir")
    extra = load_audio_extra(audio_extra)
    tfeats = load_audio_extra(text_feats)  # same loader (pkl dict vid -> ndarray)
    vfeats = load_audio_extra(visual_feats)
    afeats = load_audio_extra(audio_feats)
    if name == "iemocap":
        base = f"{data_dir}/IEMOCAP_features.pkl"
        rob = f"{data_dir}/iemocap_features_roberta.pkl"

        def cls(b, r, s, t, keys=None):
            return IEMOCAPDataset(b, r, s, t, audio_extra=extra, audio_replace=audio_replace,
                                  text_feats=tfeats, visual_feats=vfeats, audio_feats=afeats,
                                  sentiment=sentiment, keys=keys)

    elif name == "meld":
        base = f"{data_dir}/MELD_features_raw1.pkl"
        rob = f"{data_dir}/meld_features_roberta.pkl"

        def cls(b, r, s, t, keys=None):
            return MELDDataset(b, r, s, t, text_feats=tfeats, visual_feats=vfeats, audio_feats=afeats,
                               sentiment=sentiment)

    else:
        raise ValueError(name)
    if loso_test_session:
        if name != "iemocap":
            raise ValueError("Session-wise LOSO is only defined for IEMOCAP")
        all_keys = cls(base, rob, "all", text_mode).keys
        prefix = lambda session: f"Ses{session:02d}"
        if loso_valid_session:
            if (loso_test_session == loso_valid_session or
                    not (1 <= loso_valid_session <= 5)):
                raise ValueError("LOSO test and validation sessions must be distinct integers in [1, 5]")
            split_keys = {
                "test": [vid for vid in all_keys if vid.startswith(prefix(loso_test_session))],
                "valid": [vid for vid in all_keys if vid.startswith(prefix(loso_valid_session))],
                "train": [vid for vid in all_keys if not vid.startswith((prefix(loso_test_session), prefix(loso_valid_session)))],
            }
        elif 0.0 < loso_internal_valid_frac < 1.0:
            # Standard session-wise LOSO.  Hold out one entire Session for test and
            # split a deterministic dialogue-level validation subset from each of the
            # remaining four Sessions.  This preserves dialogue context and keeps all
            # validation dialogues disjoint from both optimisation and test data.
            test_keys = [vid for vid in all_keys if vid.startswith(prefix(loso_test_session))]
            train_pool = [vid for vid in all_keys if not vid.startswith(prefix(loso_test_session))]
            rng = np.random.default_rng(loso_split_seed + loso_test_session)
            valid_set = set()
            for session in range(1, 6):
                if session == loso_test_session:
                    continue
                session_keys = [vid for vid in train_pool if vid.startswith(prefix(session))]
                n_valid = max(1, int(round(len(session_keys) * loso_internal_valid_frac)))
                selected = rng.choice(len(session_keys), size=n_valid, replace=False)
                valid_set.update(session_keys[i] for i in selected)
            split_keys = {
                "test": test_keys,
                "valid": [vid for vid in train_pool if vid in valid_set],
                "train": [vid for vid in train_pool if vid not in valid_set],
            }
        else:
            raise ValueError(
                "LOSO requires either a distinct --loso-valid-session or "
                "0 < --loso-internal-valid-frac < 1"
            )
        sets = {split: cls(base, rob, split, text_mode, keys=keys) for split, keys in split_keys.items()}
    else:
        sets = {split: cls(base, rob, split, text_mode) for split in ("train", "valid", "test")}
    if merge_valid:
        sets["train"] = cls(base, rob, "train_full", text_mode)
        sets["valid"] = sets["test"]
    if standardize:
        std = Standardizer(sets["train"])
        sets = {k: StandardizedView(v, std) for k, v in sets.items()}
    return sets
