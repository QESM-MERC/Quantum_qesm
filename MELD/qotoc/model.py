"""QOTOC: OTOC-inspired quantum model for multimodal emotion recognition in conversation.

Information flow (按 特征 -> 时序建模 -> 跨模态交互 -> 融合 -> 读出):

  raw features (t/a/v) --QuantumStateEncoder--> pure states |psi_t^m> in C^d
       --UnitaryEvolutionBlock x L (per modality)--> context-evolved states
       --OTOCCrossModalBlock x Lc--> cross-modally scattered states
       --DensityMixtureFusion--> mixed state rho_t (factored)
       --BornReadout--> log p(emotion | rho_t)
  plus per-modality Born readouts (local measurements) used for the
  self-distillation auxiliary losses.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex_ops import unit_normalize
from .modules import (
    BornReadout,
    ClassicalCrossModalBlock,
    ClassicalTemporalBlock,
    ComplexCrossModalBlock,
    ComplexMLPReadout,
    ComplexTemporalBlock,
    DensityMixtureFusion,
    EntangledBornReadout,
    MetricReadout,
    MLPReadout,
    OTOCCrossModalBlock,
    QuantumRNNBlock,
    RNNTemporalBlock,
    QuantumStateEncoder,
    RelationalGraphBranch,
    UnitaryEvolutionBlock,
)

MODALITIES = ("t", "a", "v")


class QOTOCModel(nn.Module):
    def __init__(
        self,
        feat_dims: dict,          # {"t": 1024, "a": 1582, "v": 342}
        n_speakers: int,
        n_classes: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_temporal_layers: int = 2,
        n_cross_layers: int = 1,
        ffn_mult: int = 2,
        input_dropout: float = 0.3,
        attn_dropout: float = 0.1,
        state_dropout: float = 0.15,
        born_rank: int = 32,
        joint_rank: int = 64,
        joint_init: float = 0.0,
        joint_freeze: bool = False,
        per_class_tau: bool = False,
        gate_hidden: int = 128,
        window: int = 0,
        modrelu_bias: float = 0.0,
        phase_scale: float = 0.1,
        enc_hidden: int = 0,
        enc_layers: int = 1,
        coherence_init: float = None,
        interleave: bool = False,
        rope_base: float = 10000.0,
        gru_context: bool = False,
        otoc_poly: bool = False,
        otoc_ops: int = 1,
        ablate: tuple = (),
        rel_buckets: int = 16,
        gate_text_bias: float = 0.0,
        graph_branch: bool = False,
        d_graph: int = 256,
        graph_layers: int = 2,
        graph_window: int = 10,
        graph_combine: str = "sum",
        temporal_model: str = "quantum",
        cross_model: str = "quantum",
        readout: str = "born",
        classical_param_match: str = "arch",
        mlp_readout_hidden: int = 0,
        relation_distill: bool = False,
        unimodal_path: str = "post-cross",
        mask_missing_modalities: bool = False,
    ):
        super().__init__()
        if unimodal_path not in ("post-cross", "independent"):
            raise ValueError("unimodal_path must be 'post-cross' or 'independent'")
        self.unimodal_path = unimodal_path
        self.mask_missing_modalities = bool(mask_missing_modalities)
        self.graph_combine = graph_combine
        self.relation_distill = relation_distill
        ablate = set(ablate)
        self.modalities = [m for m in MODALITIES if m in feat_dims]
        # streams = modality states
        self.streams = self.modalities
        self.interleave = interleave
        self.encoders = nn.ModuleDict(
            {
                m: QuantumStateEncoder(feat_dims[m], d_model, n_speakers, input_dropout, phase_scale,
                                       enc_hidden, gru_context, ablate_phase="phase" in ablate,
                                       input_norm="layer",
                                       enc_layers=enc_layers)
                for m in self.modalities
            }
        )
        def _make_temporal():
            if temporal_model == "classical":
                return nn.ModuleList(
                    [
                        ClassicalTemporalBlock(
                            d_model, n_heads, ffn_mult, attn_dropout, state_dropout, window,
                            rel_buckets,
                            param_match=classical_param_match, ablate_relbias="relbias" in ablate,
                        )
                        for _ in range(n_temporal_layers)
                    ]
                )
            if temporal_model == "complex":
                # complex-but-non-quantum control: complex Transformer self-attn, NO unitary kernel
                return nn.ModuleList(
                    [
                        ComplexTemporalBlock(
                            d_model, n_heads, ffn_mult, attn_dropout, state_dropout, window,
                            modrelu_bias, rel_buckets,
                            param_match=classical_param_match, ablate_relbias="relbias" in ablate,
                        )
                        for _ in range(n_temporal_layers)
                    ]
                )
            if temporal_model == "quantum-rnn":
                # quantum-recurrent control: complex unit-norm GRU (diagonal-unitary step evolution)
                # replaces self-attention; cross-modal OTOC + Born readout unchanged.
                return nn.ModuleList(
                    [
                        QuantumRNNBlock(
                            d_model, n_heads, ffn_mult, attn_dropout, state_dropout, window, modrelu_bias,
                            rope_base,
                            ablate_relbias="relbias" in ablate, rel_buckets=rel_buckets,
                        )
                        for _ in range(n_temporal_layers)
                    ]
                )
            if temporal_model in ("gru", "lstm"):
                # standard recurrent temporal-encoder baselines (DialogueRNN-style GRU/LSTM):
                # classic NN sequence encoder vs the quantum-attention evolution.
                return nn.ModuleList(
                    [
                        RNNTemporalBlock(
                            d_model, n_heads, ffn_mult, attn_dropout, state_dropout, window, modrelu_bias,
                            rope_base, ablate_relbias="relbias" in ablate, rel_buckets=rel_buckets,
                            kind=temporal_model,
                        )
                        for _ in range(n_temporal_layers)
                    ]
                )
            return nn.ModuleList(
                [
                    UnitaryEvolutionBlock(
                        d_model, n_heads, ffn_mult, attn_dropout, state_dropout, window, modrelu_bias,
                        rope_base,
                        ablate_relbias="relbias" in ablate, rel_buckets=rel_buckets,
                    )
                    for _ in range(n_temporal_layers)
                ]
            )
        self.temporal = nn.ModuleDict({m: _make_temporal() for m in self.streams})
        if cross_model == "classical":
            self.cross = nn.ModuleList(
                [
                    ClassicalCrossModalBlock(
                        self.streams, d_model, n_heads, ffn_mult, attn_dropout, state_dropout, window,
                        rel_buckets,
                        param_match=classical_param_match, ablate_relbias="relbias" in ablate,
                    )
                    for _ in range(n_cross_layers)
                ]
            )
        elif cross_model == "complex":
            # complex-but-non-quantum control: complex cross-attn, NO OTOC, NO unitary kernel
            self.cross = nn.ModuleList(
                [
                    ComplexCrossModalBlock(
                        self.streams, d_model, n_heads, ffn_mult, attn_dropout, state_dropout, window,
                        modrelu_bias, rel_buckets,
                        param_match=classical_param_match, ablate_relbias="relbias" in ablate,
                    )
                    for _ in range(n_cross_layers)
                ]
            )
        else:
            self.cross = nn.ModuleList(
                [
                    OTOCCrossModalBlock(
                        self.streams, d_model, n_heads, ffn_mult, attn_dropout, state_dropout, window, modrelu_bias,
                        rope_base,
                        otoc_poly=otoc_poly, otoc_ops=otoc_ops,
                        ablate_otoc="otoc" in ablate, ablate_relbias="relbias" in ablate,
                        rel_buckets=rel_buckets,
                    )
                    for _ in range(n_cross_layers)
                ]
            )
        # readout sectors: stream states
        self.sectors = self.streams
        self.fusion = DensityMixtureFusion(self.sectors, d_model, gate_hidden, state_dropout,
                                           ablate_gate="gate" in ablate, text_bias=gate_text_bias)
        if readout in ("mlp", "complex", "metric"):
            # matched-param control over the same fused state. born_param_ref = analytic
            # Born-head budget (A_re/A_im + F_re/F_im) so hidden auto-matches param count.
            # mlp = real [Re;Im]->MLP; complex = complex MLP body + real logit head (non-Born);
            # metric = normalised cosine/prototype (metric-learning) classifier (non-Born).
            born_param_ref = (2 * n_classes * born_rank * len(self.sectors) * d_model
                              + 2 * len(self.streams) * n_classes * joint_rank * d_model)
            if readout == "metric":
                self.readout = MetricReadout(d_model, n_classes, len(self.sectors), dropout=state_dropout)
            else:
                ReadoutCls = MLPReadout if readout == "mlp" else ComplexMLPReadout
                self.readout = ReadoutCls(
                    d_model, n_classes, len(self.sectors), hidden=mlp_readout_hidden,
                    dropout=state_dropout, born_param_ref=born_param_ref,
                )
        else:
            self.readout = EntangledBornReadout(
                d_model, n_classes, len(self.sectors), mix_rank=born_rank, joint_rank=joint_rank,
                coherence_init=coherence_init, n_cp=len(self.streams),
                joint_init=joint_init, joint_freeze=joint_freeze,
                per_class_tau=per_class_tau,
            )
        self.uni_readout = nn.ModuleDict({m: BornReadout(d_model, n_classes, born_rank // 2) for m in self.modalities})
        # decorrelating relational-graph branch (DialogueGCN/MMGCN inductive bias):
        # independent emotion-logit stream combined with the quantum log-probs to
        # move the shared decision boundary the forensics identified.
        self.graph_branch = None
        if graph_branch:
            in_dim = sum(feat_dims[m] for m in self.modalities)
            self.graph_branch = RelationalGraphBranch(in_dim, d_graph, n_classes, n_speakers,
                                                      graph_layers, graph_window, input_dropout)
            self.graph_alpha = nn.Parameter(torch.tensor(1.0))  # logit-mix weight (init contributes)
            if graph_combine == "gate":
                # learned per-utterance mix weight from both branches' confidence stats
                # (q_max, g_max, q_entropy, g_entropy, disagree) -> w in (0,1)
                self.graph_gate = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, 1))

    @staticmethod
    def _relation_graph(z: torch.Tensor) -> torch.Tensor:
        """Pairwise utterance relation graph (QR-Net-style).

        z: (B, T, D) complex (or real). Returns (B, T, T) cosine-similarity
        matrix computed on the real [Re; Im] embedding, L2-normalised so the
        graph is a unit-cosine relation structure independent of feature norm.
        """
        x = torch.cat([z.real, z.imag], dim=-1) if torch.is_complex(z) else z
        x = F.normalize(x, p=2, dim=-1)
        return torch.matmul(x, x.transpose(-1, -2))

    def shared_parameters(self):
        """Parameters shared by the fused, unimodal, and KD objectives.

        CSS-style Pareto gradient modulation must compare task gradients on
        the common representation path, not on task-specific readout heads.
        """
        parameters = []
        seen = set()
        modules = (self.encoders, self.temporal)
        if self.unimodal_path == "post-cross":
            modules += (self.cross,)
        for module in modules:
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    parameters.append(parameter)
                    seen.add(id(parameter))
        return tuple(parameters)

    def forward(
        self,
        feats: dict,
        spk: torch.Tensor,
        mask: torch.Tensor,
        availability: torch.Tensor = None,
    ):
        """
        feats: dict modality -> (B, T, d_m) float
        spk:   (B, T, n_speakers) float one-hot
        mask:  (B, T) bool, True for valid utterances

        Returns dict with log-probs and diagnostics.
        """
        B, T = mask.shape
        tpos = torch.arange(T, device=mask.device, dtype=torch.float32)[None, :].expand(B, T)
        if self.mask_missing_modalities:
            if availability is None:
                raise ValueError("mask_missing_modalities requires an availability tensor")
            if availability.shape != (B, T, len(MODALITIES)):
                raise ValueError(
                    "availability must have shape "
                    f"({B}, {T}, {len(MODALITIES)}), got {tuple(availability.shape)}"
                )
            modality_available = {
                modality: mask & availability[:, :, MODALITIES.index(modality)].bool()
                for modality in self.modalities
            }
        else:
            modality_available = {modality: mask for modality in self.modalities}
        available_tensor = torch.stack(
            [modality_available[modality] for modality in self.modalities], dim=-1
        )
        active_modality_available = (
            modality_available if self.mask_missing_modalities else None
        )
        active_available_tensor = (
            available_tensor if self.mask_missing_modalities else None
        )
        model_feats = {
            modality: (
                feats[modality]
                * modality_available[modality].unsqueeze(-1).to(feats[modality].dtype)
                if self.mask_missing_modalities
                else feats[modality]
            )
            for modality in self.modalities
        }

        encoded = {
            m: self.encoders[m](model_feats[m], spk, mask) for m in self.modalities
        }
        encoded_states = {m: encoded[m][0] for m in self.modalities}
        states = dict(encoded_states)
        intens = {m: encoded[m][1] for m in self.modalities}

        otoc_log = {}
        if self.interleave:
            # T-C-T(-C...) alternation: cross-modal scattering between evolution layers
            n_t = len(self.temporal[self.streams[0]])
            n_c = len(self.cross)
            for i in range(max(n_t, n_c)):
                if i < n_t:
                    for m in self.streams:
                        states[m] = self.temporal[m][i](
                            states[m], tpos, spk, modality_available[m]
                        )
                if i < n_c:
                    states, ol, _ = self.cross[i](
                        states, tpos, spk, mask, active_modality_available
                    )
                    otoc_log.update(ol)
        else:
            for m in self.streams:
                for blk in self.temporal[m]:
                    states[m] = blk(states[m], tpos, spk, modality_available[m])
            for blk in self.cross:
                states, ol, _ = blk(
                    states, tpos, spk, mask, active_modality_available
                )
                otoc_log.update(ol)

        psis, alpha, purity = self.fusion({k: states[k] for k in self.sectors},
                                          {k: intens[k] for k in self.sectors},
                                          active_available_tensor)
        if isinstance(self.readout, EntangledBornReadout):
            logp = self.readout(psis, alpha, active_available_tensor)
        else:
            logp = self.readout(psis, alpha)

        # relational-graph branch: independent logit stream combined at the logit
        # level (alpha init 1, so it contributes from step 0 -- unlike a 0-gated
        # residual). logp_graph keeps its own normalised stream for the aux loss
        # that forces the branch to be independently predictive (decorrelated).
        logp_graph = None
        if self.graph_branch is not None:
            g_logits = self.graph_branch(
                torch.cat([model_feats[m] for m in self.modalities], dim=-1), spk, mask
            )
            logp_graph = torch.log_softmax(g_logits, dim=-1)
            if self.graph_combine == "sum":
                logp = torch.log_softmax(logp + self.graph_alpha * g_logits, dim=-1)
            elif self.graph_combine == "gate":
                # learned per-utterance convex mix of the two log-prob streams
                qp, gp = logp.exp(), logp_graph.exp()
                feat = torch.cat([
                    qp.max(-1, keepdim=True).values,
                    gp.max(-1, keepdim=True).values,
                    -(qp * logp).sum(-1, keepdim=True),               # quantum entropy
                    -(gp * logp_graph).sum(-1, keepdim=True),         # graph entropy
                    (logp.argmax(-1, keepdim=True) != logp_graph.argmax(-1, keepdim=True)).float(),
                ], dim=-1)
                w = torch.sigmoid(self.graph_gate(feat))              # (B,T,1)
                logp = torch.log_softmax((1.0 - w) * logp + w * logp_graph, dim=-1)
            else:
                # gate the (decorrelated but weaker) graph stream so it only fires
                # where the quantum model is UNSURE -> captures rescues on the
                # contested set without breaking confident-correct predictions.
                uncert = 1.0 - logp.exp().max(-1, keepdim=True).values   # (B,T,1)
                gate = F.softplus(self.graph_alpha) * uncert
                if self.graph_combine == "conf":
                    gate = gate * logp_graph.exp().max(-1, keepdim=True).values  # AND graph-confident
                logp = torch.log_softmax(logp + gate * g_logits, dim=-1)

        logp_base = logp

        # fused superposition state (unit norm) for fidelity-based contrastive loss
        Psi = (alpha.clamp_min(1e-8).sqrt().unsqueeze(-1) * psis).reshape(B, T, -1)

        if self.unimodal_path == "independent":
            # A genuine unimodal branch must never consume cross-modal states.
            # Reuse the shared temporal blocks on each encoded stream without
            # invoking cross attention, so its NLL/KD gradients still train the
            # common encoder/temporal backbone used by the fused path.
            uni_states = dict(encoded_states)
            for m in self.streams:
                for block in self.temporal[m]:
                    uni_states[m] = block(
                        uni_states[m], tpos, spk, modality_available[m]
                    )
        else:
            uni_states = states
        logp_uni = {
            m: self.uni_readout[m](unit_normalize(uni_states[m]).unsqueeze(2))
            for m in self.modalities
        }
        mod_drop = torch.zeros(B, len(self.modalities), dtype=torch.bool, device=mask.device)

        # QR-Net-style relation-graph self-distillation: pairwise utterance
        # relation structure of each (weaker) unimodal sector (students) is later
        # pulled toward the fused superposition's structure (teacher). Computed
        # only while training so eval/inference forward stays numerically identical.
        g_uni = g_fused = None
        if self.relation_distill and self.training:
            g_uni = {m: self._relation_graph(uni_states[m]) for m in self.modalities}
            g_fused = self._relation_graph(Psi)
        return {
            "logp": logp,            # (B, T, K)
            "logp_base": logp_base,  # (B, T, K) - fused readout
            "logp_uni": logp_uni,    # dict m -> (B, T, K)
            "alpha": alpha,          # (B, T, M)
            "purity": purity,        # (B, T)
            "otoc": otoc_log,        # dict pair -> scalar
            "Psi": Psi,              # (B, T, M*D) complex unit states
            "mod_drop": mod_drop,    # (B, M) bool - modality dropped this forward
            "mod_available": available_tensor,  # (B, T, M) bool - observed raw modality
            "logp_graph": logp_graph,  # (B, T, K) or None - relational-graph branch stream
            "g_uni": g_uni,            # dict m -> (B, T, T) relation graphs (students) or None
            "g_fused": g_fused,        # (B, T, T) fused relation graph (teacher) or None
        }
