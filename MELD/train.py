"""Train QESM on IEMOCAP or MELD with the weighted-F1 protocol.

The internal model class retains its historical ``QOTOCModel`` name for
compatibility with experiment code and saved checkpoints.
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from dataset import CachedDataset, build_datasets, collate_dialogues
from qotoc import build_model


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def configure_runtime(args, device):
    if args.matmul_precision != "default" and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not args.deterministic
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


def build_lr_schedulers(optimizer, args):
    """Build either an epoch schedule or a metric-driven plateau schedule."""
    plateau_patience = int(getattr(args, "lr_plateau_patience", 0))
    if plateau_patience > 0:
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(args.lr_plateau_factor),
            patience=plateau_patience,
            threshold=float(args.lr_plateau_threshold),
            threshold_mode="abs",
            min_lr=float(args.min_lr),
        )
        return None, plateau

    if args.warmup > 0 or args.cosine:
        def lr_lambda(epoch):
            if args.warmup > 0 and epoch < args.warmup:
                return (epoch + 1) / args.warmup
            if args.cosine:
                progress = (epoch - args.warmup) / max(args.epochs - args.warmup, 1)
                return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), None

    return None, None


def class_counts_from(dataset, n_classes):
    counts = np.zeros(n_classes)
    for i in range(len(dataset)):
        for c in dataset[i][4]:
            counts[c] += 1
    return counts


def class_weights_from(dataset, n_classes, power=1.0, beta=0.0):
    """Per-class loss weights, normalised to mean 1.

    power: exponent on inverse frequency, w = (1/freq)^power. power=1 is full
      inverse-frequency (over-weights minorities -> can hurt weighted-F1);
      power in (0,1) is a milder reweighting that better preserves weighted-F1.
    beta>0: class-balanced weighting (Cui et al. 2019), w = (1-beta)/(1-beta^n_c),
      the "effective number of samples"; typical beta in {0.9, 0.99, 0.999}.
    """
    counts = class_counts_from(dataset, n_classes)
    if beta > 0:
        w = (1.0 - beta) / (1.0 - np.power(beta, np.maximum(counts, 1.0)))
    else:
        freq = counts / counts.sum()
        w = np.power(1.0 / np.maximum(freq, 1e-8), power)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def smoothed_nll(logp, target, weight=None, smoothing=0.0):
    """NLL on log-probs with label smoothing and per-class weights."""
    if target.numel() == 0:
        return logp.sum() * 0.0
    nll_per = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    if weight is not None:
        wt = weight[target]
        nll = (nll_per * wt).sum() / wt.sum()
    else:
        nll = nll_per.mean()
    if smoothing > 0:
        if weight is not None:
            wt = weight[target]
            uni = -(logp.mean(dim=-1) * wt).sum() / wt.sum()
        else:
            uni = -logp.mean()
        return (1 - smoothing) * nll + smoothing * uni
    return nll


def effective_loss_weights(args, epoch=None):
    """Return the current unimodal/KD weights for an optional linear transition.

    With the default ``loss_transition_epochs=0`` this is exactly the historical
    static weighting. A transition moves from the optional ``*_start`` values at
    epoch 1 to ``lambda_uni``/``lambda_kd`` at the final transition epoch.
    """
    target_uni = float(args.lambda_uni)
    target_kd = float(args.lambda_kd)
    transition = int(getattr(args, "loss_transition_epochs", 0))
    if transition <= 1:
        return target_uni, target_kd
    start_uni_arg = getattr(args, "lambda_uni_start", None)
    start_kd_arg = getattr(args, "lambda_kd_start", None)
    start_uni = target_uni if start_uni_arg is None else float(start_uni_arg)
    start_kd = target_kd if start_kd_arg is None else float(start_kd_arg)
    current_epoch = int(epoch if epoch is not None else getattr(args, "_cur_epoch", 1))
    alpha = min(max((current_epoch - 1) / (transition - 1), 0.0), 1.0)
    return (
        start_uni + alpha * (target_uni - start_uni),
        start_kd + alpha * (target_kd - start_kd),
    )


def effective_contrastive_weight(args, epoch=None):
    """Return the current fidelity-contrastive weight for an optional ramp."""
    target = float(args.lambda_con)
    transition = int(getattr(args, "con_transition_epochs", 0))
    start_arg = getattr(args, "lambda_con_start", None)
    if transition <= 1 or start_arg is None:
        return target
    current_epoch = int(epoch if epoch is not None else getattr(args, "_cur_epoch", 1))
    alpha = min(max((current_epoch - 1) / (transition - 1), 0.0), 1.0)
    return float(start_arg) + alpha * (target - float(start_arg))


_PGM_NORM_EPSILON = 1e-8
_PGM_LINEAR_TOLERANCE = 1e-10
_PGM_OBJECTIVE_TIE_TOLERANCE = 1e-8
_PGM_WEIGHT_SNAP_TOLERANCE = 1e-12


def _pgm_gradient_gram(losses, shared_parameters, return_norms=False):
    """Build CSS's normalized three-task gradient Gram matrix on CPU float64.

    Complex gradients are treated as their concatenated real and imaginary
    components. ``None`` gradients remain position-aligned and contribute zero.
    """
    if len(losses) != 3:
        raise ValueError("CSS PGM requires exactly three task losses")
    parameters = tuple(shared_parameters)
    if not parameters:
        raise ValueError("CSS PGM requires at least one shared parameter")

    task_gradients = []
    for task_index, loss in enumerate(losses):
        if not bool(torch.isfinite(loss.detach()).all().item()):
            raise FloatingPointError(f"CSS PGM task {task_index} loss is non-finite")
        task_gradients.append(
            torch.autograd.grad(
                loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
        )

    contributions = [[[] for _ in range(3)] for _ in range(3)]
    for parameter_index in range(len(parameters)):
        gradients = []
        for task_index in range(3):
            gradient = task_gradients[task_index][parameter_index]
            if gradient is None:
                gradients.append(None)
                continue
            value = gradient.detach()
            if torch.is_complex(value):
                value = torch.view_as_real(value)
            value = value.to(device="cpu", dtype=torch.float64).contiguous().view(-1)
            if not bool(torch.isfinite(value).all().item()):
                raise FloatingPointError(
                    f"CSS PGM task {task_index} has a non-finite shared gradient"
                )
            gradients.append(value)

        for left_index in range(3):
            left = gradients[left_index]
            if left is None:
                continue
            for right_index in range(left_index, 3):
                right = gradients[right_index]
                if right is None:
                    continue
                dot = float(torch.dot(left, right).item())
                contributions[left_index][right_index].append(dot)
                if right_index != left_index:
                    contributions[right_index][left_index].append(dot)

    raw_gram = np.empty((3, 3), dtype=np.float64)
    for left_index in range(3):
        for right_index in range(3):
            raw_gram[left_index, right_index] = math.fsum(
                contributions[left_index][right_index]
            )
    diagonal = np.diag(raw_gram)
    if np.any(diagonal < -_PGM_LINEAR_TOLERANCE):
        raise FloatingPointError("CSS PGM produced a negative squared gradient norm")
    norms = np.sqrt(np.maximum(diagonal, 0.0))
    gram = raw_gram / np.outer(norms + _PGM_NORM_EPSILON, norms + _PGM_NORM_EPSILON)
    gram = 0.5 * (gram + gram.T)
    if not np.isfinite(gram).all():
        raise FloatingPointError("CSS PGM produced a non-finite Gram matrix")
    if return_norms:
        return gram, norms
    return gram


def solve_pgm_simplex(gram, minimum_weights=None):
    """Solve CSS's three-task minimum-norm simplex problem by active sets.

    Optional lower bounds retain the same quadratic objective while preventing
    a task from being removed entirely by the unconstrained Pareto solution.
    """
    matrix = np.asarray(gram, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("CSS PGM requires a 3x3 Gram matrix")
    matrix = 0.5 * (matrix + matrix.T)
    if not np.isfinite(matrix).all():
        raise FloatingPointError("CSS PGM solver received a non-finite Gram matrix")

    lower = np.zeros(3, dtype=np.float64)
    if minimum_weights is not None:
        lower = np.asarray(minimum_weights, dtype=np.float64)
    if lower.shape != (3,) or not np.isfinite(lower).all() or np.any(lower < 0):
        raise ValueError("CSS PGM minimum weights must be three finite non-negative values")
    if float(lower.sum()) >= 1.0:
        raise ValueError("CSS PGM minimum weights must sum to less than one")

    uniform = np.full(3, 1.0 / 3.0, dtype=np.float64)
    candidates = [uniform] if np.all(uniform >= lower) else []
    for support_mask in range(1, 1 << 3):
        support = [index for index in range(3) if support_mask & (1 << index)]
        fixed = [index for index in range(3) if index not in support]
        face = matrix[np.ix_(support, support)]
        size = len(support)
        kkt = np.empty((size + 1, size + 1), dtype=np.float64)
        kkt[:size, :size] = face
        kkt[:size, size] = -1.0
        kkt[size, :size] = 1.0
        kkt[size, size] = 0.0
        right_hand_side = np.zeros(size + 1, dtype=np.float64)
        if fixed:
            right_hand_side[:size] = -matrix[np.ix_(support, fixed)].dot(lower[fixed])
        right_hand_side[size] = 1.0 - float(lower[fixed].sum())
        solution, _, _, _ = np.linalg.lstsq(kkt, right_hand_side, rcond=None)
        residual = float(np.max(np.abs(kkt.dot(solution) - right_hand_side)))
        face_weights = solution[:size]
        if (
            residual > _PGM_LINEAR_TOLERANCE
            or np.any(face_weights < lower[support] - _PGM_LINEAR_TOLERANCE)
        ):
            continue
        candidate = lower.copy()
        candidate[support] = np.where(
            np.abs(face_weights - lower[support]) <= _PGM_WEIGHT_SNAP_TOLERANCE,
            lower[support],
            face_weights,
        )
        candidate = np.maximum(candidate, lower)
        candidate[support[0]] += 1.0 - float(candidate.sum())
        if candidate[support[0]] < lower[support[0]] - _PGM_LINEAR_TOLERANCE:
            continue
        candidates.append(candidate)

    if not candidates:
        raise RuntimeError("CSS PGM active-set solver found no feasible candidate")

    def objective(candidate):
        return 0.5 * float(candidate.dot(matrix.dot(candidate)))

    objectives = np.asarray([objective(candidate) for candidate in candidates])
    if not np.isfinite(objectives).all():
        raise FloatingPointError("CSS PGM produced a non-finite objective")
    minimum = float(objectives.min())
    tie_tolerance = _PGM_OBJECTIVE_TIE_TOLERANCE * max(1.0, abs(minimum))
    near_optimal = [
        candidate
        for candidate, value in zip(candidates, objectives)
        if value <= minimum + tie_tolerance
    ]
    selected = min(
        near_optimal,
        key=lambda candidate: (
            float(np.square(candidate - uniform).sum()),
            tuple(float(value) for value in candidate),
        ),
    ).copy()
    selected = np.where(
        np.abs(selected - lower) <= _PGM_WEIGHT_SNAP_TOLERANCE,
        lower,
        selected,
    )
    selected = np.maximum(selected, lower)
    selected /= selected.sum()
    return selected


def css_pgm_weights(losses, shared_parameters, unimodal_min_weight=0.0):
    """Return detached CSS-style Pareto weights for three task losses."""
    gram, gradient_norms = _pgm_gradient_gram(
        losses, shared_parameters, return_norms=True
    )
    weights = solve_pgm_simplex(
        gram,
        minimum_weights=(0.0, float(unimodal_min_weight), 0.0),
    )
    css_pgm_weights.last_diagnostics = {
        "gradient_norms": gradient_norms.copy(),
        "cosine_matrix": gram.copy(),
    }
    reference = losses[0]
    return torch.as_tensor(weights, device=reference.device, dtype=reference.dtype).detach()


def selection_improved(selection_split, valid_improved, test_improved):
    """Return whether the split driving checkpoint selection improved."""
    if selection_split == "test":
        return bool(test_improved)
    if selection_split == "valid":
        return bool(valid_improved)
    raise ValueError("selection_split must be 'test' or 'valid'")


def fidelity_supcon(Psi, labels, mask, temp=0.1, max_samples=0, class_counts=None, swfc_gamma=0.0):
    """Supervised contrastive loss with quantum fidelity |<Psi_i|Psi_j>|^2 as similarity.

    swfc_gamma>0 enables SWFC-style (SSLCL/MultiEMO) per-anchor sample weighting
    w_i=(N/n_{y_i})^gamma, which tightens MINORITY-class clusters (helps the
    fear/disgust/sadness collapse) while the main CE term protects weighted-F1.
    """
    valid = mask.view(-1)
    z = Psi.reshape(-1, Psi.size(-1))[valid]
    y = labels.view(-1)[valid]
    if z.size(0) < 4:
        return z.real.sum() * 0.0
    if max_samples and z.size(0) > max_samples:
        idx = torch.randperm(z.size(0), device=z.device)[:max_samples]
        z, y = z[idx], y[idx]
    S = torch.abs(z @ z.conj().T) ** 2  # (N, N) in [0, 1]
    logits = S / temp
    N = z.size(0)
    eye = torch.eye(N, dtype=torch.bool, device=z.device)
    pos = (y[:, None] == y[None, :]) & ~eye
    logits = logits.masked_fill(eye, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = pos.sum(1)
    li = -(log_prob * pos).sum(1) / pos_count.clamp_min(1)
    has_pos = pos_count > 0
    if not has_pos.any():
        return z.real.sum() * 0.0
    if swfc_gamma > 0 and class_counts is not None:
        cc = torch.tensor(class_counts, device=z.device, dtype=torch.float32)
        w = (cc.sum() / cc.clamp_min(1.0))[y] ** swfc_gamma   # per-anchor minority up-weight
        w = (w * has_pos.float())
        return (li * w).sum() / w.sum().clamp_min(1e-8)
    return li[has_pos].mean()


class EMA:
    """Exponential moving average of model weights, applied at evaluation.

    ``correct=True`` removes the contribution of the random initial snapshot
    from the shadow weights.  This preserves the historical corrected-EMA
    protocol used by some archived QESM runs while keeping the default
    behaviour unchanged.
    """

    def __init__(self, model, decay, correct=False):
        self.decay = decay
        self.correct = correct
        self.t = 0
        self.shadow = {
            k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point
        }
        if correct:
            self.init = {k: v.clone() for k, v in self.shadow.items()}

    @torch.no_grad()
    def update(self, model):
        self.t += 1
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def swap_in(self, model):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        shadow = self.shadow
        if self.correct and self.t > 0:
            init_weight = self.decay ** self.t
            if init_weight < 1.0 - 1e-8:
                shadow = {
                    k: (value - init_weight * self.init[k]) / (1.0 - init_weight)
                    for k, value in self.shadow.items()
                }
        model.load_state_dict({**model.state_dict(), **shadow})

    def swap_out(self, model):
        model.load_state_dict({**model.state_dict(), **self.backup})


def kd_loss(logp_student, logp_teacher, temp=2.0):
    ps = F.log_softmax(logp_student / temp, dim=-1)
    pt = F.softmax(logp_teacher.detach() / temp, dim=-1)
    return F.kl_div(ps, pt, reduction="batchmean") * temp * temp


def relation_distill_loss(g_student, g_teacher, mask):
    """QR-Net-style relation-graph distillation: masked MSE between two
    (B, T, T) pairwise relation graphs. mask: (B, T) bool valid. The (i, j)
    entry is supervised only when utterances i and j are both valid."""
    m2 = (mask.unsqueeze(1) & mask.unsqueeze(2)).to(g_student.dtype)  # (B, T, T)
    diff = (g_student - g_teacher) ** 2
    return (diff * m2).sum() / (m2.sum() + 1e-9)


@torch.no_grad()
def ogm_modulate(
    model,
    logp_uni,
    labels,
    mask,
    modality_available,
    modalities,
    coef,
    noise,
):
    """On-the-fly Gradient Modulation (OGM-GE, Peng et al. CVPR'22).

    Each modality's discriminative contribution is read from its own unimodal
    Born readout: s_m = mean p(true). The dominant modality (s_m above the
    cross-modal mean) gets its private encoder/temporal gradients scaled by
    k_m = 1 - coef*tanh(s_m/mean - 1) < 1, slowing it so the suppressed weak
    modalities (visual/audio) get to learn. GE adds Gaussian noise to restore
    the generalisation the modulation would otherwise cost. Shared cross-modal
    / fusion / readout parameters are left untouched.
    """
    s = {}
    for modality_index, (m, lp) in enumerate(logp_uni.items()):
        # logp_uni are LOG-probabilities; exponentiate to read p(true) in (0,1)
        observed = mask & modality_available[:, :, modality_index]
        p = lp[observed].exp()
        if p.numel() == 0:
            return
        target = labels[observed]
        s[m] = p.gather(-1, target.unsqueeze(-1)).mean().item()
    mean_s = sum(s.values()) / len(s)
    k = {}
    for m, sm in s.items():
        rho = sm / (mean_s + 1e-8)
        k[m] = 1.0 - coef * math.tanh(rho - 1.0) if rho > 1.0 else 1.0
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        for m in modalities:
            if f"encoders.{m}." in n or f"temporal.{m}." in n:
                p.grad.mul_(k[m])
                if noise > 0:
                    p.grad.add_(torch.randn_like(p.grad) * (noise * p.grad.std()))
                break


class _SubsetView(torch.utils.data.Dataset):
    """Seeded dialogue-level subset of the train split for low-resource (Sec.7).
    Delegates getitem/len and forwards the attributes the model build reads. MUST be
    module-level (not nested in main) so DataLoader workers can pickle it on Windows spawn."""
    def __init__(self, ds, idx):
        self.ds, self.idx = ds, list(idx)
        self.n_classes, self.n_speakers = ds.n_classes, ds.n_speakers
        self.feat_dims = ds.feat_dims

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.ds[self.idx[i]]


def run_epoch(model, loader, device, optimizer=None, weight=None, args=None, ema=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    all_pred, all_true = [], []
    total_loss, n_batches = 0.0, 0
    component_totals = {"main_nll": 0.0, "uni_nll": 0.0, "kd": 0.0, "other": 0.0}
    # Balanced-Softmax / logit-adjustment prior (train-time only; predictions use raw logits)
    la_prior = None
    if getattr(args, "logit_adjust", 0.0) > 0 and getattr(args, "class_counts", None):
        c = torch.tensor(args.class_counts, dtype=torch.float32, device=device)
        la_prior = args.logit_adjust * torch.log(c / c.sum())
    non_blocking = bool(getattr(args, "pin_memory", False)) and device.type == "cuda"
    needed_modalities = getattr(model, "modalities", None)
    lambda_uni, lambda_kd = effective_loss_weights(args)
    lambda_con = effective_contrastive_weight(args)
    pgm_mode = getattr(args, "pgm_mode", "off")
    unimodal_predictions = {m: [] for m in getattr(model, "modalities", ())}
    unimodal_targets = {m: [] for m in getattr(model, "modalities", ())}
    availability_total = np.zeros(len(unimodal_predictions), dtype=np.float64)
    fusion_alpha_total = np.zeros(len(unimodal_predictions), dtype=np.float64)
    observed_alpha_total = np.zeros(len(unimodal_predictions), dtype=np.float64)
    fusion_alpha_count = 0
    observed_alpha_count = np.zeros(len(unimodal_predictions), dtype=np.float64)
    shared_parameters = ()
    pgm_weight_total = np.zeros(3, dtype=np.float64)
    pgm_weight_min = np.full(3, np.inf, dtype=np.float64)
    pgm_weight_max = np.full(3, -np.inf, dtype=np.float64)
    pgm_gradient_norm_total = np.zeros(3, dtype=np.float64)
    pgm_cosine_total = np.zeros((3, 3), dtype=np.float64)
    pgm_diagnostic_batches = 0
    grad_norm_total = 0.0
    grad_norm_max = 0.0
    if pgm_mode == "css":
        shared_method = getattr(model, "shared_parameters", None)
        if not callable(shared_method):
            raise ValueError("CSS PGM requires model.shared_parameters()")
        shared_parameters = tuple(shared_method())

    context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        for batch in loader:
            if len(batch) == 5:
                feats, spk, labels, mask, availability = batch
            else:
                feats, spk, labels, mask = batch
                availability = torch.stack([mask, mask, mask], dim=-1)
            if needed_modalities is not None:
                feats = {k: feats[k].to(device, non_blocking=non_blocking) for k in needed_modalities}
            else:
                feats = {k: v.to(device, non_blocking=non_blocking) for k, v in feats.items()}
            spk = spk.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            mask = mask.to(device, non_blocking=non_blocking)
            availability = availability.to(device, non_blocking=non_blocking)

            # missing-modality robustness: zero named modalities at EVAL only (after standardization,
            # before the model). The `not train` guard is load-bearing -- never corrupt training.
            if (not train) and getattr(args, "test_drop_modality", ""):
                if getattr(args, "mask_missing_modalities", False):
                    availability = availability.clone()
                    for modality in args.test_drop_modality:
                        availability[:, :, "tav".index(modality)] = False
                else:
                    feats = {k: (torch.zeros_like(v) if k in args.test_drop_modality else v)
                             for k, v in feats.items()}

            # test-time feature corruption (z-scored inputs -> sigma in std units): a graceful-
            # degradation curve vs the matched-classical control. `not train` guard load-bearing;
            # the eval harness seeds the RNG so each sigma is deterministic.
            if (not train) and getattr(args, "test_noise_sigma", 0.0) > 0:
                s = args.test_noise_sigma
                feats = {k: v + torch.randn_like(v) * s for k, v in feats.items()}

            if train and getattr(args, "feat_noise_sigma", 0.0) > 0:
                # additive Gaussian feature-space noise (inputs are z-scored, so sigma is in
                # std units): a corruption regularizer for the small-data overfit regime that
                # input-dropout/utt-dropout don't cover. Smooths the decision boundaries.
                s = args.feat_noise_sigma
                feats = {
                    k: torch.where(
                        availability[:, :, "tav".index(k)].unsqueeze(-1),
                        v + torch.randn_like(v) * s,
                        v,
                    )
                    for k, v in feats.items()
                }

            valid = mask.view(-1)
            target = labels.view(-1)[valid]

            def main_nll(lp_flat, msk):
                return smoothed_nll(lp_flat[msk.view(-1)], labels.view(-1)[msk.view(-1)],
                                    weight, args.label_smoothing)

            def compute_loss():
                out = model(feats, spk, mask, availability)
                K = out["logp"].size(-1)
                logp = out["logp"].view(-1, K)[valid]
                # logit adjustment (Balanced Softmax): add tau*log(prior) to the main
                # logits during training so rare classes are not crowded out; the
                # auxiliary heads, KD teacher and all predictions keep the raw logp.
                main_logp = out["logp"]
                if la_prior is not None and train:
                    main_logp = torch.log_softmax(main_logp + la_prior, dim=-1)
                main_loss = main_nll(main_logp.view(-1, K), mask)
                uni_loss = main_loss.new_zeros(())
                kd_total = main_loss.new_zeros(())
                manual_loss = main_loss
                for mi, (m, lp_uni) in enumerate(out["logp_uni"].items()):
                    keep_m = mask & out["mod_available"][:, :, mi]
                    lpu = lp_uni.view(-1, K)[keep_m.view(-1)]
                    tea_m = out["logp"].view(-1, K)[keep_m.view(-1)]
                    uni_term = main_nll(lp_uni.view(-1, K), keep_m)
                    uni_loss = uni_loss + uni_term
                    manual_loss = manual_loss + lambda_uni * uni_term
                    if lpu.numel() > 0:
                        kd_term = kd_loss(lpu, tea_m, args.kd_temp)
                        kd_total = kd_total + kd_term
                        manual_loss = manual_loss + lambda_kd * kd_term
                task_losses = (main_loss, uni_loss, kd_total)
                pgm_weights = None
                if pgm_mode == "css":
                    if train:
                        pgm_weights = css_pgm_weights(
                            task_losses,
                            shared_parameters,
                            getattr(args, "pgm_uni_min_weight", 0.0),
                        )
                    else:
                        frozen_weights = getattr(args, "_pgm_eval_weights", None)
                        if frozen_weights is None:
                            raise ValueError(
                                "CSS PGM evaluation requires frozen training-epoch weights"
                            )
                        pgm_weights = torch.as_tensor(
                            frozen_weights,
                            device=main_loss.device,
                            dtype=main_loss.dtype,
                        ).detach()
                    pgm_weights = pgm_weights / pgm_weights.sum()
                    loss = sum(
                        task_weight * task_loss
                        for task_weight, task_loss in zip(pgm_weights, task_losses)
                    )
                else:
                    # Preserve the historical per-modality accumulation order.
                    # It is mathematically equivalent to weighting the summed task
                    # losses, but keeps frozen legacy configurations reproducible.
                    loss = manual_loss
                core_loss = loss
                if args.lambda_rel > 0 and train and out.get("g_fused") is not None:
                    # relation-graph self-distillation (QR-Net "R"): pull each
                    # unimodal sector's pairwise utterance-relation structure toward
                    # the fused structure (teacher detached, mirroring kd_loss).
                    g_tea = out["g_fused"].detach()
                    rel = sum(
                        relation_distill_loss(
                            g_s,
                            g_tea,
                            mask & out["mod_available"][:, :, modality_index],
                        )
                        for modality_index, g_s in enumerate(out["g_uni"].values())
                    )
                    loss = loss + args.lambda_rel * rel / max(len(out["g_uni"]), 1)
                if lambda_con > 0 and train:
                    loss = loss + lambda_con * fidelity_supcon(
                        out["Psi"], labels, mask, args.con_temp, args.con_max_samples,
                        class_counts=getattr(args, "class_counts", None), swfc_gamma=args.swfc_gamma
                    )
                if args.lambda_graph > 0 and out.get("logp_graph") is not None:
                    # force the relational-graph branch to be independently predictive
                    # (so it learns a decorrelated view, not a deferential residual)
                    loss = loss + args.lambda_graph * main_nll(out["logp_graph"].view(-1, K), mask)
                if args.rdrop > 0 and train:
                    # second stochastic forward; symmetric KL keeps the two predictive
                    # distributions consistent under dropout noise
                    lp2_flat = model(feats, spk, mask, availability)["logp"].view(-1, K)
                    logp2 = lp2_flat[valid]
                    loss = loss + main_nll(lp2_flat, mask)
                    kl = 0.5 * (
                        F.kl_div(logp, logp2.exp().detach(), reduction="batchmean")
                        + F.kl_div(logp2, logp.exp().detach(), reduction="batchmean")
                    )
                    loss = loss + args.rdrop * kl
                components = {
                    "main_nll": main_loss.detach(),
                    "uni_nll": uni_loss.detach(),
                    "kd": kd_total.detach(),
                    "other": (loss - core_loss).detach(),
                }
                return loss, out, K, components, pgm_weights

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss, out, K, components, pgm_weights = compute_loss()
                loss.backward()
                if args.ogm > 0:
                    ogm_modulate(
                        model,
                        out["logp_uni"],
                        labels,
                        mask,
                        out["mod_available"],
                        model.modalities,
                        args.ogm,
                        args.ogm_noise,
                    )
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
                )
                grad_norm_total += grad_norm
                grad_norm_max = max(grad_norm_max, grad_norm)
                optimizer.step()
                if ema is not None:
                    ema.update(model)
            else:
                loss, out, K, components, pgm_weights = compute_loss()

            logp = out["logp"].view(-1, K)[valid]
            total_loss += loss.item()
            for name, value in components.items():
                component_totals[name] += value.item()
            if pgm_weights is not None:
                batch_pgm_weights = pgm_weights.detach().to(
                    device="cpu", dtype=torch.float64
                ).numpy()
                pgm_weight_total += batch_pgm_weights
                pgm_weight_min = np.minimum(pgm_weight_min, batch_pgm_weights)
                pgm_weight_max = np.maximum(pgm_weight_max, batch_pgm_weights)
                if train:
                    diagnostics = getattr(css_pgm_weights, "last_diagnostics", None)
                    if diagnostics is not None:
                        pgm_gradient_norm_total += diagnostics["gradient_norms"]
                        pgm_cosine_total += diagnostics["cosine_matrix"]
                        pgm_diagnostic_batches += 1
            n_batches += 1
            all_pred.append(logp.argmax(-1).cpu().numpy())
            all_true.append(target.cpu().numpy())
            for modality_index, (modality, modality_logp) in enumerate(
                out["logp_uni"].items()
            ):
                modality_valid = mask & out["mod_available"][:, :, modality_index]
                unimodal_predictions[modality].append(
                    modality_logp[modality_valid].argmax(-1).cpu().numpy()
                )
                unimodal_targets[modality].append(labels[modality_valid].cpu().numpy())
                availability_total[modality_index] += int(modality_valid.sum().item())
            if out.get("alpha") is not None:
                valid_alpha = out["alpha"].view(-1, out["alpha"].size(-1))[valid]
                fusion_alpha_total += valid_alpha.sum(0).detach().to(
                    device="cpu", dtype=torch.float64
                ).numpy()
                fusion_alpha_count += int(valid_alpha.size(0))
                for modality_index in range(valid_alpha.size(-1)):
                    modality_valid = mask & out["mod_available"][:, :, modality_index]
                    observed_alpha_total[modality_index] += float(
                        out["alpha"][:, :, modality_index][modality_valid].sum().item()
                    )
                    observed_alpha_count[modality_index] += int(
                        modality_valid.sum().item()
                    )

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    wf1 = f1_score(true, pred, average="weighted") * 100
    acc = accuracy_score(true, pred) * 100
    mechanisms = {
        "unimodal_wf1": {
            modality: f1_score(
                np.concatenate(unimodal_targets[modality]),
                np.concatenate(predictions),
                average="weighted",
                zero_division=0,
            ) * 100
            for modality, predictions in unimodal_predictions.items()
            if predictions
        },
        "unimodal_acc": {
            modality: accuracy_score(
                np.concatenate(unimodal_targets[modality]),
                np.concatenate(predictions),
            ) * 100
            for modality, predictions in unimodal_predictions.items()
            if predictions
        },
        "fusion_alpha_mean": {
            modality: float(fusion_alpha_total[index] / max(fusion_alpha_count, 1))
            for index, modality in enumerate(unimodal_predictions)
        },
        "fusion_alpha_when_available": {
            modality: float(
                observed_alpha_total[index] / max(observed_alpha_count[index], 1)
            )
            for index, modality in enumerate(unimodal_predictions)
        },
        "availability_rate": {
            modality: float(availability_total[index] / max(len(true), 1))
            for index, modality in enumerate(unimodal_predictions)
        },
    }
    if train:
        # Preserve the public five-item return signature. The main loop snapshots
        # these diagnostics immediately after the train pass, before evaluation.
        run_epoch.last_train_components = {
            name: value / max(n_batches, 1) for name, value in component_totals.items()
        }
        run_epoch.last_pgm_weights = (
            (pgm_weight_total / max(n_batches, 1)).tolist()
            if pgm_mode == "css" else None
        )
        run_epoch.last_pgm_diagnostics = (
            {
                "task_order": ["fused_nll", "unimodal_nll", "kd_kl"],
                "mean_gradient_norms": (
                    pgm_gradient_norm_total / max(pgm_diagnostic_batches, 1)
                ).tolist(),
                "mean_cosine_matrix": (
                    pgm_cosine_total / max(pgm_diagnostic_batches, 1)
                ).tolist(),
                "min_weights": pgm_weight_min.tolist(),
                "max_weights": pgm_weight_max.tolist(),
            }
            if pgm_mode == "css" else None
        )
        run_epoch.last_train_grad_norm = {
            "mean_before_clip": grad_norm_total / max(n_batches, 1),
            "max_before_clip": grad_norm_max,
            "clip_threshold": float(args.clip),
        }
        run_epoch.last_train_mechanisms = mechanisms
    else:
        run_epoch.last_eval_mechanisms = mechanisms
    return total_loss / max(n_batches, 1), wf1, acc, true, pred


def _evaluate_validation_and_test(model, loaders, device, weight, args):
    """Evaluate validation/test, avoiding duplicate work for merged splits."""
    valid_result = run_epoch(model, loaders["valid"], device, None, weight, args)
    if args.merge_valid:
        return valid_result, valid_result
    test_result = run_epoch(model, loaders["test"], device, None, weight, args)
    return valid_result, test_result


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        help="JSON file containing argparse destination names; explicit CLI options override it",
    )
    p.add_argument("--dataset", choices=["iemocap", "meld"])
    p.add_argument(
        "--data-dir",
        default=None,
        help="legacy directory containing split M3Net-family pickle files",
    )
    p.add_argument(
        "--feature-pkl",
        default=None,
        help="single combined combined pickle; enables raw-feature-only mode",
    )
    p.add_argument(
        "--feature-valid-fraction",
        type=float,
        default=0.1,
        help="fraction of the combined train pool used for validation when --feature-valid-count=0",
    )
    p.add_argument(
        "--feature-valid-count",
        type=int,
        default=0,
        help="exact combined validation-dialogue count; 0 uses --feature-valid-fraction",
    )
    p.add_argument(
        "--feature-valid-position",
        choices=["head", "tail"],
        default="head",
        help="take the combined validation dialogues from the head or tail of the train pool",
    )
    p.add_argument("--out-dir", default="results")
    p.add_argument("--run-name", default=None)
    # model
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--temporal-layers", type=int, default=2)
    p.add_argument("--cross-layers", type=int, default=1)
    p.add_argument("--ffn-mult", type=int, default=2)
    p.add_argument("--born-rank", type=int, default=32)
    p.add_argument("--gate-hidden", type=int, default=128)
    p.add_argument("--input-dropout", type=float, default=0.3)
    p.add_argument("--attn-dropout", type=float, default=0.1)
    p.add_argument("--state-dropout", type=float, default=0.15)
    # classical / complex control blocks (matched-param ablation; defaults reproduce the quantum path exactly)
    p.add_argument("--temporal-model", choices=["quantum", "classical", "complex", "quantum-rnn", "gru", "lstm"], default="quantum",
                   help="quantum=UnitaryEvolutionBlock (attention); classical=real Transformer self-attn; "
                        "complex=complex Transformer self-attn with NO unitary kernel (complex-non-quantum control); "
                        "quantum-rnn=complex unit-norm GRU with diagonal-unitary evolution (quantum-recurrent); "
                        "gru/lstm=standard bidirectional recurrent temporal encoder (DialogueRNN-style baseline)")
    p.add_argument("--cross-model", choices=["quantum", "classical", "complex"], default="quantum",
                   help="quantum=OTOCCrossModalBlock; classical=real cross-attention; "
                        "complex=complex cross-attention with NO OTOC/unitary (complex-non-quantum control)")
    p.add_argument("--readout", choices=["born", "mlp", "complex", "metric"], default="born",
                   help="born=EntangledBornReadout; mlp=real MLP head over the fused state; "
                        "complex=complex MLP body + real logit head, no Born rule (complex-non-quantum control); "
                        "metric=normalised cosine/prototype (metric-learning) classifier, no Born rule")
    p.add_argument("--classical-param-match", choices=["arch", "count"], default="arch",
                   help="arch=equal-architecture control; count=widen the classical FFN to ~match quantum param count")
    p.add_argument("--mlp-readout-hidden", type=int, default=0,
                   help="hidden width of the --readout mlp head; 0=auto-match the Born head param budget")
    # optimisation
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers; 0 is safest on Windows")
    p.add_argument("--pin-memory", dest="pin_memory", action="store_true", help="pin host memory for CUDA transfer")
    p.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    p.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor when workers > 0")
    p.add_argument("--persistent-workers", action="store_true", help="keep DataLoader workers alive between epochs")
    p.add_argument("--cache-data", action="store_true", help="materialise dataset items once after preprocessing")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--lambda-uni", type=float, default=0.3)
    p.add_argument("--lambda-kd", type=float, default=0.3)
    p.add_argument(
        "--unimodal-path",
        choices=["post-cross", "independent"],
        default="post-cross",
        help=(
            "use historical post-cross auxiliary heads (default); independent "
            "enables genuinely modality-isolated auxiliary tasks"
        ),
    )
    p.add_argument(
        "--pgm-mode",
        choices=["off", "css"],
        default="off",
        help=(
            "CSS-style three-task Pareto gradient modulation over fused NLL, "
            "summed unimodal NLL, and summed KD KL"
        ),
    )
    p.add_argument(
        "--pgm-uni-min-weight",
        type=float,
        default=0.0,
        help="minimum CSS-PGM simplex weight for the summed unimodal NLL task",
    )
    p.add_argument("--lambda-uni-start", type=float, default=None,
                   help="initial unimodal-loss weight for a linear loss transition")
    p.add_argument("--lambda-kd-start", type=float, default=None,
                   help="initial KD-loss weight for a linear loss transition")
    p.add_argument("--loss-transition-epochs", type=int, default=0,
                   help="linearly move *_start loss weights to --lambda-uni/--lambda-kd; 0=static")
    p.add_argument("--lambda-rel", type=float, default=0.0,
                   help="weight of QR-Net-style relation-graph self-distillation "
                        "(unimodal relation graphs -> fused graph, masked MSE; 0 = off)")
    p.add_argument("--kd-temp", type=float, default=2.0)
    p.add_argument("--class-weights", action="store_true")
    p.add_argument("--no-class-weights", dest="class_weights", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--cosine", action="store_true", help="cosine LR decay over epochs")
    p.add_argument(
        "--lr-plateau-patience",
        type=int,
        default=0,
        help="reduce LR after this many non-improving selection epochs; 0 disables it",
    )
    p.add_argument("--lr-plateau-factor", type=float, default=0.5)
    p.add_argument("--lr-plateau-threshold", type=float, default=0.01)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--no-standardize", dest="standardize", action="store_false")
    p.add_argument(
        "--missing-aware-standardize",
        action="store_true",
        help=(
            "feature raw mode only: fit audio/visual z-score statistics on non-zero rows "
            "and keep all-zero missing-modality rows at zero"
        ),
    )
    p.add_argument(
        "--mask-missing-modalities",
        action="store_true",
        help=(
            "propagate raw combined availability masks through temporal/cross/fusion/readout "
            "and exclude missing modalities from their auxiliary losses"
        ),
    )
    p.add_argument(
        "--merge-valid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use the complete encoded train pool",
    )
    p.add_argument(
        "--selection-split",
        choices=["test", "valid"],
        default="test",
        help="split whose weighted F1 drives early stopping (both best checkpoints are still saved)",
    )
    p.add_argument("--loso-test-session", type=int, default=0,
                   help="IEMOCAP held-out test Session in [1,5]; 0 uses the fixed split")
    p.add_argument("--loso-valid-session", type=int, default=0,
                   help="IEMOCAP validation Session in [1,5] for LOSO")
    p.add_argument("--loso-internal-valid-frac", type=float, default=0.0,
                   help="for standard LOSO, hold out this dialogue fraction from each non-test Session for validation")
    p.add_argument("--loso-split-seed", type=int, default=1,
                   help="seed for the dialogue-level internal validation split in standard LOSO")
    p.add_argument("--joint-rank", type=int, default=64)
    p.add_argument("--joint-init", type=float, default=0.0,
                   help="init logit for the entangled<->coherent blend; w=sigmoid(joint_init). >0 favors entanglement")
    p.add_argument("--joint-freeze", action="store_true", help="freeze joint_weight at its init (e.g. pure-entangled)")
    p.add_argument("--per-class-tau", action="store_true", help="per-class measurement temperature in the readout (recalibrates high-support boundaries for WF1)")
    p.add_argument("--feat-noise-sigma", type=float, default=0.0, help="additive Gaussian feature-space noise std (z-scored units), TRAIN only; 0=off")
    p.add_argument("--test-noise-sigma", type=float, default=0.0, help="additive Gaussian feature noise std (z-scored units) at EVAL only (robustness curve); 0=off")
    p.add_argument("--window", type=int, default=0, help="attention window radius, 0 = full")
    p.add_argument("--modalities", default="tav", help="subset of 'tav' to use")
    p.add_argument("--test-drop-modality", default="",
                   help="subset of 'tav' (e.g. 'av' or 'a,v') to ZERO at eval only (missing-modality robustness); empty=off")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay for eval weights, 0 = off")
    p.add_argument(
        "--ema-correct",
        action="store_true",
        help="remove the random-initialisation contribution from EMA evaluation weights",
    )
    # v2 architecture flags (defaults reproduce v1 behaviour)
    p.add_argument("--modrelu-bias", type=float, default=0.0)
    p.add_argument("--phase-scale", type=float, default=0.1)
    p.add_argument("--enc-hidden", type=int, default=0)
    p.add_argument("--enc-layers", type=int, default=1, help="projection MLP depth (1=original); >1 = deeper encoder capacity")
    p.add_argument("--coherence-init", type=float, default=None)
    p.add_argument("--interleave", action="store_true")
    p.add_argument("--rope-base", type=float, default=10000.0)
    p.add_argument("--train-fraction", type=float, default=1.0,
                   help="low-resource (Sec.7): train on this fraction of train dialogues (seeded by --seed); 1.0 = exact baseline (no-op)")
    p.add_argument("--lambda-con", type=float, default=0.0)
    p.add_argument("--lambda-con-start", type=float, default=None)
    p.add_argument(
        "--con-transition-epochs",
        type=int,
        default=0,
        help="linearly ramp --lambda-con-start to --lambda-con; 0 keeps a static weight",
    )
    p.add_argument("--con-temp", type=float, default=0.1)
    p.add_argument("--con-max-samples", type=int, default=0,
                   help="cap supervised-contrastive utterance pairs per batch; 0 = all")
    p.add_argument("--rdrop", type=float, default=0.0, help="R-Drop consistency weight")
    p.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw", help="base optimizer")
    p.add_argument("--momentum", type=float, default=0.9, help="momentum for --optimizer sgd (Nesterov)")
    p.add_argument("--gate-text-bias", type=float, default=0.0, help="additive logit bias toward the text sector in fusion gate")
    p.add_argument("--cw-power", type=float, default=1.0, help="exponent on inverse-frequency class weights (w=(1/freq)^power)")
    p.add_argument("--cw-beta", type=float, default=0.0, help="class-balanced weighting beta (Cui et al.; 0=use cw-power instead)")
    p.add_argument("--logit-adjust", type=float, default=0.0, help="Balanced-Softmax / logit-adjustment tau on the main loss (train only)")
    p.add_argument("--swfc-gamma", type=float, default=0.0, help="SWFC minority sample-weighting exponent in the supcon loss (needs --lambda-con>0)")
    p.add_argument("--graph-branch", action="store_true", help="add a relational-graph (DialogueGCN/MMGCN) decorrelating branch")
    p.add_argument("--d-graph", type=int, default=256, help="hidden width of the relational-graph branch")
    p.add_argument("--graph-layers", type=int, default=2, help="relational-graph propagation layers")
    p.add_argument("--graph-window", type=int, default=10, help="relational-graph edge window")
    p.add_argument("--lambda-graph", type=float, default=0.5, help="weight of the independent graph-branch classification loss")
    p.add_argument("--graph-combine", choices=["sum", "uncert", "conf", "gate"], default="sum",
                   help="fuse graph logits: sum / uncertainty-gated / uncertainty x graph-conf / learned per-utterance gate")
    p.add_argument("--text-mode", choices=["mean", "concat", "r1"], default="mean")
    p.add_argument("--warmup", type=int, default=0, help="linear LR warmup epochs")
    p.add_argument("--gru-context", action="store_true", help="bi-GRU classical context backbone per modality")
    p.add_argument("--audio-extra", default=None, help="path to extra utterance-level audio features (pkl or npy dir)")
    p.add_argument("--audio-replace", action="store_true", help="use only the extra features as audio modality")
    p.add_argument("--text-feats", default=None, help="path to a pkl (vid->ndarray) replacing the RoBERTa text feature")
    p.add_argument("--visual-feats", default=None, help="path to a pkl (vid->ndarray) replacing the DenseNet visual feature")
    p.add_argument("--audio-feats", default=None, help="path to a pkl (vid->ndarray) replacing the openSMILE audio feature")
    p.add_argument("--otoc-poly", action="store_true", help="learnable scattering kernel g(s)")
    p.add_argument("--otoc-ops", type=int, default=1, help="operator pairs per head (thermal ensemble)")
    p.add_argument("--ablate", default="", help="comma list of modules to ablate: otoc,phase,gate,relbias")
    p.add_argument(
        "--sentiment",
        action="store_true",
        help="3-class sentiment task; combined MELD requires the 14-field pickle with native labels",
    )
    p.add_argument("--rel-buckets", type=int, default=16, help="relative-distance buckets for the T5 bias")
    p.add_argument("--ogm", type=float, default=0.0, help="OGM-GE gradient modulation strength (0=off)")
    p.add_argument("--ogm-noise", type=float, default=0.0, help="OGM-GE generalisation-enhancement noise sigma")
    p.add_argument("--deterministic", action="store_true",
                   help="enable deterministic kernels where PyTorch supports them")
    p.add_argument("--matmul-precision", choices=["default", "highest", "high", "medium"], default="default",
                   help="torch float32 matmul precision; high/medium can speed CUDA/modern CPU matmuls")
    p.add_argument("--tf32", action="store_true",
                   help="allow TF32 on CUDA matmul/cuDNN for faster training on Ampere+ GPUs")
    p.set_defaults(standardize=True)
    p.set_defaults(class_weights=None)
    p.set_defaults(pin_memory=None)

    config_probe, _ = p.parse_known_args()
    if config_probe.config:
        try:
            with open(config_probe.config, encoding="utf-8") as config_file:
                config_defaults = json.load(config_file)
        except (OSError, json.JSONDecodeError) as exc:
            p.error(f"cannot read --config {config_probe.config!r}: {exc}")
        if not isinstance(config_defaults, dict):
            p.error("--config must contain one JSON object")
        valid_keys = {action.dest for action in p._actions}
        unknown_keys = sorted(set(config_defaults) - valid_keys)
        if unknown_keys:
            p.error(f"unknown keys in --config: {', '.join(unknown_keys)}")
        config_defaults.pop("config", None)
        p.set_defaults(**config_defaults)

    args = p.parse_args()

    if args.dataset is None:
        p.error("--dataset is required, either on the command line or in --config")
    if (args.feature_pkl is None) == (args.data_dir is None):
        p.error("provide exactly one data source: --feature-pkl or --data-dir")

    args.modalities = "".join(dict.fromkeys(args.modalities))
    bad_modalities = [m for m in args.modalities if m not in "tav"]
    if not args.modalities or bad_modalities:
        p.error("--modalities must be a non-empty subset of 'tav'")
    # missing-modality robustness: normalize to a deduped char string (accepts 'av' or 'a,v')
    tdm = [c for c in args.test_drop_modality.lower() if not c.isspace() and c != ","]
    if any(c not in "tav" for c in tdm):
        p.error("--test-drop-modality must be a subset of 'tav'")
    args.test_drop_modality = "".join(dict.fromkeys(tdm))
    if args.d_model % args.n_heads != 0:
        p.error("--d-model must be divisible by --n-heads")
    if args.num_workers < 0:
        p.error("--num-workers must be >= 0")
    if args.prefetch_factor < 1:
        p.error("--prefetch-factor must be >= 1")
    if args.warmup < 0:
        p.error("--warmup must be >= 0")
    if args.lr_plateau_patience < 0:
        p.error("--lr-plateau-patience must be >= 0")
    if args.lr_plateau_patience and (args.warmup or args.cosine):
        p.error("plateau LR scheduling cannot be combined with --warmup or --cosine")
    if not 0.0 < args.lr_plateau_factor < 1.0:
        p.error("--lr-plateau-factor must be between 0 and 1")
    if args.lr_plateau_threshold < 0:
        p.error("--lr-plateau-threshold must be >= 0")
    if not 0.0 <= args.min_lr <= args.lr:
        p.error("--min-lr must be between 0 and --lr")
    if args.con_max_samples < 0:
        p.error("--con-max-samples must be >= 0")
    if args.feature_valid_count < 0:
        p.error("--feature-valid-count must be >= 0")
    if args.loss_transition_epochs < 0:
        p.error("--loss-transition-epochs must be >= 0")
    loss_weights = (args.lambda_uni, args.lambda_kd, args.lambda_uni_start, args.lambda_kd_start)
    if any(value is not None and value < 0 for value in loss_weights):
        p.error("loss weights must be non-negative")
    if args.loss_transition_epochs == 0 and (
        args.lambda_uni_start is not None or args.lambda_kd_start is not None
    ):
        p.error("loss start weights require --loss-transition-epochs")
    if args.con_transition_epochs < 0:
        p.error("--con-transition-epochs must be >= 0")
    if args.lambda_con < 0 or (
        args.lambda_con_start is not None and args.lambda_con_start < 0
    ):
        p.error("contrastive loss weights must be non-negative")
    if args.con_transition_epochs == 0 and args.lambda_con_start is not None:
        p.error("--lambda-con-start requires --con-transition-epochs")
    if args.pgm_mode == "css" and args.ogm > 0:
        p.error("CSS PGM cannot be combined with OGM; set --ogm 0")
    if not 0.0 <= args.pgm_uni_min_weight < 1.0:
        p.error("--pgm-uni-min-weight must be in [0, 1)")
    if args.pgm_mode == "off" and args.pgm_uni_min_weight > 0:
        p.error("--pgm-uni-min-weight requires --pgm-mode css")
    if args.mask_missing_modalities and args.feature_pkl is None:
        p.error("--mask-missing-modalities currently requires --feature-pkl")
    if args.mask_missing_modalities and not args.missing_aware_standardize:
        p.error("--mask-missing-modalities requires --missing-aware-standardize")

    if args.class_weights is None:
        args.class_weights = args.dataset == "iemocap"
    if args.run_name is None:
        args.run_name = f"{args.dataset}_d{args.d_model}_seed{args.seed}_{int(time.time())}"

    set_seed(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_runtime(args, device)
    if args.pin_memory is None:
        args.pin_memory = device.type == "cuda"
    if args.merge_valid:
        print(
            "WARNING: --merge-valid trains on train+valid and uses the test split as the selection split.",
            flush=True,
        )
    if args.pgm_mode == "css":
        print(
            "CSS PGM enabled: manual lambda-uni/lambda-kd schedules are ignored; "
            "validation/test loss uses each training epoch's mean PGM weights; "
            f"unimodal minimum weight={args.pgm_uni_min_weight:g}.",
            flush=True,
        )
    if args.loso_test_session and args.merge_valid:
        p.error("--merge-valid cannot be combined with session-wise LOSO")
    if args.loso_internal_valid_frac and not args.loso_test_session:
        p.error("--loso-internal-valid-frac requires --loso-test-session")
    if args.loso_valid_session and args.loso_internal_valid_frac:
        p.error("use either --loso-valid-session or --loso-internal-valid-frac, not both")
    if args.loso_test_session and not (args.loso_valid_session or 0.0 < args.loso_internal_valid_frac < 1.0):
        p.error("LOSO requires --loso-valid-session or 0 < --loso-internal-valid-frac < 1")

    datasets = build_datasets(
        args.dataset, args.data_dir, standardize=args.standardize, merge_valid=args.merge_valid,
        text_mode=args.text_mode, audio_extra=args.audio_extra, audio_replace=args.audio_replace,
        text_feats=args.text_feats, visual_feats=args.visual_feats,
        audio_feats=args.audio_feats, sentiment=getattr(args, "sentiment", False),
        loso_test_session=args.loso_test_session, loso_valid_session=args.loso_valid_session,
        loso_internal_valid_frac=args.loso_internal_valid_frac,
        loso_split_seed=args.loso_split_seed,
        feature_pkl=args.feature_pkl, feature_valid_fraction=args.feature_valid_fraction,
        feature_valid_count=args.feature_valid_count,
        feature_valid_position=args.feature_valid_position,
        missing_aware_standardize=args.missing_aware_standardize,
    )
    if args.loso_test_session:
        if args.loso_valid_session:
            print(
                f"[LOSO] train={len(datasets['train'])} dialogues, valid Session {args.loso_valid_session}="
                f"{len(datasets['valid'])}, test Session {args.loso_test_session}={len(datasets['test'])}",
                flush=True,
            )
        else:
            print(
                f"[LOSO] train={len(datasets['train'])} dialogues, internal valid={len(datasets['valid'])} "
                f"({args.loso_internal_valid_frac:.0%} dialogue split), test Session "
                f"{args.loso_test_session}={len(datasets['test'])}",
                flush=True,
            )
    if getattr(args, "train_fraction", 1.0) < 1.0:
        _base = datasets["train"]
        _n = len(_base)
        _k = max(1, int(round(_n * args.train_fraction)))
        _sel = sorted(np.random.default_rng(args.seed).choice(_n, size=_k, replace=False).tolist())
        datasets["train"] = _SubsetView(_base, _sel)
        print(f"[low-resource] train_fraction={args.train_fraction} -> {_k}/{_n} train dialogues (seed {args.seed})",
              flush=True)

    if args.cache_data:
        datasets = {split: CachedDataset(ds) for split, ds in datasets.items()}

    def make_loader(split, ds):
        loader_kwargs = {
            "batch_size": args.batch_size,
            "shuffle": split == "train",
            "collate_fn": collate_dialogues,
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
        }
        if args.num_workers > 0:
            loader_kwargs["prefetch_factor"] = args.prefetch_factor
            loader_kwargs["persistent_workers"] = args.persistent_workers
        return DataLoader(ds, **loader_kwargs)

    loaders = {
        split: make_loader(split, ds)
        for split, ds in datasets.items()
    }

    train_ds = datasets["train"]
    weight = (class_weights_from(train_ds, train_ds.n_classes, args.cw_power, args.cw_beta).to(device)
              if args.class_weights else None)
    # class counts cached on args (serialisable) for logit adjustment in run_epoch
    args.class_counts = class_counts_from(train_ds, train_ds.n_classes).tolist()

    model = build_model(args, train_ds).to(device)
    n_params = sum(x.numel() for x in model.parameters())
    print(f"[{args.run_name}] params: {n_params/1e6:.2f}M, device: {device}", flush=True)

    no_decay_names = ("omega", "spk_bias", "log_tau", "joint_weight", "coherence", "gamma", "rel_bias")
    decay_params, no_decay_params = [], []
    for n, prm in model.named_parameters():
        if prm.ndim <= 1 or any(t in n for t in no_decay_names):
            no_decay_params.append(prm)
        else:
            decay_params.append(prm)
    param_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum, nesterov=args.momentum > 0)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    scheduler, plateau_scheduler = build_lr_schedulers(optimizer, args)

    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    best = {
        "selection_split": args.selection_split,
        "valid_wf1": 0,
        "valid_acc": 0,
        "test_at_best_valid": 0,
        "test_acc_at_best_valid": 0,
        "best_test_wf1": 0,
        "best_test_acc": 0,
        "best_test_epoch": -1,
        "best_valid_epoch": -1,
    }
    history = []
    stale = 0

    ema = EMA(model, args.ema, args.ema_correct) if args.ema > 0 else None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        args._cur_epoch = epoch
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        epoch_lambda_uni, epoch_lambda_kd = effective_loss_weights(args, epoch)
        epoch_lambda_con = effective_contrastive_weight(args, epoch)
        tr_loss, tr_wf1, tr_acc, _, _ = run_epoch(model, loaders["train"], device, optimizer, weight, args, ema)
        tr_components = dict(getattr(run_epoch, "last_train_components", {}))
        train_mechanisms = dict(getattr(run_epoch, "last_train_mechanisms", {}))
        epoch_pgm_weights = getattr(run_epoch, "last_pgm_weights", None)
        epoch_pgm_diagnostics = getattr(run_epoch, "last_pgm_diagnostics", None)
        train_grad_norm = dict(getattr(run_epoch, "last_train_grad_norm", {}))
        if args.pgm_mode == "css":
            args._pgm_eval_weights = epoch_pgm_weights
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.swap_in(model)
        valid_result, test_result = _evaluate_validation_and_test(
            model, loaders, device, weight, args
        )
        test_mechanisms = dict(getattr(run_epoch, "last_eval_mechanisms", {}))
        _, va_wf1, va_acc, va_true, va_pred = valid_result
        _, te_wf1, te_acc, te_true, te_pred = test_result

        improved = ""
        labels = list(range(train_ds.n_classes))
        valid_improved = va_wf1 > best["valid_wf1"]
        test_improved = te_wf1 > best["best_test_wf1"]
        if valid_improved:
            best["valid_wf1"] = va_wf1
            best["valid_acc"] = va_acc
            best["test_at_best_valid"] = te_wf1
            best["test_acc_at_best_valid"] = te_acc
            best["best_valid_epoch"] = epoch
            best["report_at_best_valid"] = classification_report(te_true, te_pred, digits=4, zero_division=0)
            best["macro_f1_at_best_valid"] = f1_score(
                te_true, te_pred, average="macro", labels=labels, zero_division=0
            ) * 100
            best["perclass_f1_at_best_valid"] = (
                f1_score(te_true, te_pred, average=None, labels=labels, zero_division=0) * 100
            ).tolist()
            best["confusion_at_best_valid"] = confusion_matrix(
                te_true, te_pred, labels=labels
            ).tolist()
            np.savez(
                os.path.join(run_dir, "test_preds_at_best_valid.npz"),
                true=te_true,
                pred=te_pred,
            )
            if args.selection_split == "valid":
                np.savez(os.path.join(run_dir, "test_preds.npz"), true=te_true, pred=te_pred)
            torch.save(model.state_dict(), os.path.join(run_dir, "best_valid.pt"))
            improved += " *valid"
        if test_improved:
            best["best_test_wf1"] = te_wf1
            best["best_test_acc"] = te_acc
            best["best_test_epoch"] = epoch
            best["report_best_test"] = classification_report(te_true, te_pred, digits=4, zero_division=0)
            best["macro_f1_at_best_test"] = f1_score(
                te_true, te_pred, average="macro", labels=labels, zero_division=0
            ) * 100
            best["perclass_f1_at_best_test"] = (
                f1_score(te_true, te_pred, average=None, labels=labels, zero_division=0) * 100
            ).tolist()
            best["confusion_at_best_test"] = confusion_matrix(
                te_true, te_pred, labels=labels
            ).tolist()
            np.savez(
                os.path.join(run_dir, "test_preds_best_test.npz"),
                true=te_true,
                pred=te_pred,
            )
            if args.selection_split == "test":
                np.savez(os.path.join(run_dir, "test_preds.npz"), true=te_true, pred=te_pred)
            torch.save(model.state_dict(), os.path.join(run_dir, "best_test.pt"))
            improved += " *test"

        if selection_improved(args.selection_split, valid_improved, test_improved):
            stale = 0
        else:
            stale += 1
        if plateau_scheduler is not None:
            selection_score = te_wf1 if args.selection_split == "test" else va_wf1
            plateau_scheduler.step(selection_score)
        next_epoch_lr = float(optimizer.param_groups[0]["lr"])

        if ema is not None:
            ema.swap_out(model)
        history_row = {
            "epoch": epoch,
            "lr": epoch_lr,
            "next_lr": next_epoch_lr,
            "lambda_uni": epoch_lambda_uni if args.pgm_mode == "off" else None,
            "lambda_kd": epoch_lambda_kd if args.pgm_mode == "off" else None,
            "lambda_con": epoch_lambda_con,
            "pgm_weights": epoch_pgm_weights,
            "pgm_diagnostics": epoch_pgm_diagnostics,
            "gradient_norm": train_grad_norm,
            "train_loss": tr_loss,
            "train_wf1": tr_wf1,
            "train_acc": tr_acc,
            "valid_wf1": va_wf1,
            "valid_acc": va_acc,
            "test_wf1": te_wf1,
            "test_acc": te_acc,
            "train_mechanisms": train_mechanisms,
            "test_mechanisms": test_mechanisms,
        }
        history_row.update({f"train_{name}": value for name, value in tr_components.items()})
        history.append(history_row)
        loss_weight_text = (
            "pgm " + "/".join(f"{value:.3f}" for value in epoch_pgm_weights)
            if epoch_pgm_weights is not None
            else f"uni {epoch_lambda_uni:.3g}, kd {epoch_lambda_kd:.3g}"
        )
        print(
            f"ep {epoch:3d} | lr {epoch_lr:.2e} | loss {tr_loss:.4f} "
            f"({loss_weight_text}, con {epoch_lambda_con:.3g}) | "
            f"train {tr_wf1:.2f} | valid {va_wf1:.2f} | "
            f"test {te_wf1:.2f} (acc {te_acc:.2f}) | {time.time()-t0:.1f}s{improved}",
            flush=True,
        )
        if stale >= args.patience:
            print(f"early stop at epoch {epoch}", flush=True)
            break

    serial_best = {k: v for k, v in best.items() if not k.startswith("report_")}
    print(json.dumps(serial_best, indent=2))
    if "report_at_best_valid" in best:
        print("classification report @ best valid")
        print(best["report_at_best_valid"])
        with open(os.path.join(run_dir, "report_at_best_valid.txt"), "w") as f:
            f.write(best["report_at_best_valid"])
    if "report_best_test" in best:
        with open(os.path.join(run_dir, "report_best_test.txt"), "w") as f:
            f.write(best["report_best_test"])
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({"args": vars(args), "best": serial_best, "history": history}, f, indent=2)


if __name__ == "__main__":
    main()
