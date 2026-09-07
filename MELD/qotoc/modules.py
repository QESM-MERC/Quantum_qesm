"""QOTOC core modules.

Pipeline (per dialogue):
  1. QuantumStateEncoder   - amplitude-phase encoding of utterance features into
                             pure states |psi> in C^d.
  2. UnitaryEvolutionBlock - intra-modal temporal context via complex attention
                             whose relative-position kernel is a diagonal unitary
                             Heisenberg evolution D(dt) = e^{i H dt} (H diagonal).
  3. OTOCCrossModalBlock   - cross-modal interaction gated by an OTOC kernel.
                             Local operators are Householder reflections
                             W = I - 2|w><w|; the squared commutator of the
                             time-evolved pair has the closed form
                                 C = (32/d) * s * (1 - s),  s = |<w(t)|v>|^2,
                             so the normalised OTOC score is 4 s (1-s) in [0,1].
  4. DensityMixtureFusion  - per-utterance mixed state rho = sum_m a_m |psi_m><psi_m|
                             kept in factored (low-rank) form; purity Tr[rho^2]
                             is exposed as an "emotional ambiguity" diagnostic.
  5. BornReadout           - POVM measurement M_k = A_k^dagger A_k, with
                             p(k) = Tr[M_k rho] = sum_m a_m ||A_k psi_m||^2.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex_ops import (
    ComplexFFN,
    ComplexLayerNorm,
    ComplexLinear,
    ModReLU,
    complex_dropout,
    init_frequencies,
    make_rotation,
    unit_normalize,
)

NEG_INF = -1e9


def window_mask(T: int, window: int, device) -> torch.Tensor:
    """(T, T) bool, True where |i-j| <= window. window<=0 means no restriction."""
    if window <= 0:
        return torch.ones(T, T, dtype=torch.bool, device=device)
    pos = torch.arange(T, device=device)
    return (pos[None, :] - pos[:, None]).abs() <= window


def relative_buckets(T: int, device, n_buckets: int = 16, max_dist: int = 64) -> torch.Tensor:
    """T5-style signed log-spaced relative-distance buckets, (T, T) long."""
    pos = torch.arange(T, device=device)
    rel = pos[None, :] - pos[:, None]  # j - i
    n_half = n_buckets // 2
    sign = (rel > 0).long() * n_half
    a = rel.abs().clamp(1)
    n_exact = n_half // 2
    log_bucket = (
        n_exact
        + (torch.log(a.float() / n_exact) / torch.log(torch.tensor(max_dist / n_exact, device=device)) * (n_half - 1 - n_exact)).long()
    ).clamp(max=n_half - 1)
    bucket = torch.where(a < n_exact, a, log_bucket)
    return (sign + bucket).clamp(0, n_buckets - 1)


class RelativeBias(nn.Module):
    def __init__(self, n_heads: int, n_buckets: int = 16):
        # n_buckets is threaded from the attention modules (--rel-buckets)
        super().__init__()
        self.n_buckets = n_buckets
        self.emb = nn.Embedding(n_buckets, n_heads)
        nn.init.zeros_(self.emb.weight)

    def forward(self, T: int, device) -> torch.Tensor:
        """returns (1, H, T, T) additive bias."""
        b = relative_buckets(T, device, self.n_buckets)
        return self.emb(b).permute(2, 0, 1)[None]


class QuantumStateEncoder(nn.Module):
    """Amplitude-phase (dense) encoding of classical features into pure states."""

    def __init__(self, d_in: int, d_model: int, n_speakers: int, input_dropout: float,
                 phase_scale: float = 0.1, enc_hidden: int = 0, gru_context: bool = False,
                 ablate_phase: bool = False, input_norm: str = "layer", enc_layers: int = 1):
        super().__init__()
        self.ablate_phase = ablate_phase
        # "layer": per-utterance LayerNorm (default). "none": rely on the dataset
        # z-score only, preserving the per-utterance scale (loudness/intensity) that
        # LayerNorm strips -- restores a meaningful intensity signal for the gate (3A).
        self.input_norm = nn.LayerNorm(d_in) if input_norm == "layer" else nn.Identity()
        self.drop = nn.Dropout(input_dropout)
        self.spk_emb = nn.Linear(n_speakers, d_in, bias=False)
        feat_dim = d_in
        self.pre = None
        self.gru = None
        if gru_context:
            # classical bidirectional context backbone before quantum state preparation
            self.gru = nn.GRU(d_in, d_model, bidirectional=True, batch_first=True)
            feat_dim = 2 * d_model
        elif enc_hidden > 0:
            # enc_layers=1 (default) == original single Linear->GELU->Dropout projection.
            # enc_layers>1 stacks extra (enc_hidden->enc_hidden) blocks = deeper capacity
            # in the *projection* (the bottleneck), distinct from widening d_model (saturates).
            layers = [nn.Linear(d_in, enc_hidden), nn.GELU(), nn.Dropout(input_dropout)]
            for _ in range(max(0, enc_layers - 1)):
                layers += [nn.Linear(enc_hidden, enc_hidden), nn.GELU(), nn.Dropout(input_dropout)]
            self.pre = nn.Sequential(*layers)
            feat_dim = enc_hidden
        self.mag = nn.Linear(feat_dim, d_model)
        self.phase = nn.Linear(feat_dim, d_model)
        # phase_scale < 1 starts near zero phase (real-network-like warm start)
        with torch.no_grad():
            self.phase.weight.mul_(phase_scale)
            self.phase.bias.zero_()

    def forward(self, x: torch.Tensor, spk: torch.Tensor, mask: torch.Tensor = None):
        """x: (B, T, d_in) real, spk: (B, T, n_speakers) one-hot.

        Returns (B,T,d) complex unit states and the pre-normalisation magnitude
        ||r|| (B,T,1) - the modality "intensity" the unit sphere discards.
        """
        h = self.input_norm(x) + self.spk_emb(spk)
        h = self.drop(h)
        if self.gru is not None:
            lengths = mask.sum(1).clamp_min(1).cpu() if mask is not None else torch.full((x.shape[0],), x.shape[1])
            packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
            h = self.drop(h)
        elif self.pre is not None:
            h = self.pre(h)
        r = self.mag(h)
        if self.ablate_phase:
            theta = torch.zeros_like(r)
        else:
            theta = torch.pi * torch.tanh(self.phase(h))
        psi = torch.complex(r * torch.cos(theta), r * torch.sin(theta))
        norm = torch.linalg.vector_norm(r, dim=-1, keepdim=True)
        return unit_normalize(psi), torch.log1p(norm)


def _split_heads(z: torch.Tensor, n_heads: int) -> torch.Tensor:
    """(B, T, D) -> (B, T, H, Dh)"""
    B, T, D = z.shape
    return z.view(B, T, n_heads, D // n_heads)


def _merge_heads(z: torch.Tensor) -> torch.Tensor:
    """(B, T, H, Dh) -> (B, T, D)"""
    B, T, H, Dh = z.shape
    return z.reshape(B, T, H * Dh)


def _same_speaker(spk: torch.Tensor) -> torch.Tensor:
    """spk: (B, T, S) one-hot -> (B, T, T) 1 where same speaker."""
    return torch.einsum("bis,bjs->bij", spk, spk).clamp(0, 1)


def _speaker_bias(same: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """same: (B, T, T); bias: (H, 2) -> (B, H, T, T) additive attention bias."""
    return bias[None, :, 0, None, None] * same[:, None] + bias[None, :, 1, None, None] * (1 - same[:, None])


def _build_keep(mask, T, device, window):
    """Combined source-validity / window mask -> (B, 1, T, T)."""
    wm = window_mask(T, window, device)[None, None]  # (1,1,T,T)
    return mask[:, None, None, :] & wm


class UnitaryEvolutionAttention(nn.Module):
    """Complex self-attention with a diagonal-unitary Heisenberg evolution kernel.

    score_ij = Re <q_i | D(t_j - t_i) | k_j> / sqrt(Dh) + speaker bias.
    """

    def __init__(self, dim: int, n_heads: int, attn_dropout: float = 0.1, window: int = 0,
                 rope_base: float = 10000.0, ablate_relbias: bool = False, rel_buckets: int = 16):
        super().__init__()
        assert dim % n_heads == 0
        self.h = n_heads
        self.dh = dim // n_heads
        self.window = window
        self.wq = ComplexLinear(dim, dim)
        self.wk = ComplexLinear(dim, dim)
        self.wv = ComplexLinear(dim, dim)
        self.wo = ComplexLinear(dim, dim)
        self.omega = nn.Parameter(init_frequencies(n_heads, self.dh, rope_base))
        self.spk_bias = nn.Parameter(torch.zeros(n_heads, 2))
        self.rel_bias = None if ablate_relbias else RelativeBias(n_heads, rel_buckets)
        self.p_attn = attn_dropout

    def forward(self, z, tpos, spk, mask):
        """z: (B,T,D) complex; tpos: (B,T) real; spk: (B,T,S); mask: (B,T) bool valid."""
        q = _split_heads(self.wq(z), self.h)
        k = _split_heads(self.wk(z), self.h)
        v = _split_heads(self.wv(z), self.h)

        rot = make_rotation(tpos, self.omega)  # (B,T,H,Dh)
        qr = q * rot
        kr = k * rot

        same = _same_speaker(spk)
        scores = torch.einsum("bihd,bjhd->bhij", qr.conj(), kr).real / (self.dh ** 0.5)
        scores = scores + _speaker_bias(same, self.spk_bias)
        if self.rel_bias is not None:
            scores = scores + self.rel_bias(z.shape[1], z.device)
        keep = _build_keep(mask, z.shape[1], z.device, self.window)
        scores = scores.masked_fill(~keep, NEG_INF)
        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, self.p_attn, self.training)

        out = torch.complex(
            torch.einsum("bhij,bjhd->bihd", attn, v.real),
            torch.einsum("bhij,bjhd->bihd", attn, v.imag),
        )
        return self.wo(_merge_heads(out))


class UnitaryEvolutionBlock(nn.Module):
    def __init__(self, dim, n_heads, ffn_mult=2, attn_dropout=0.1, state_dropout=0.1, window=0,
                 modrelu_bias=0.0, rope_base=10000.0, ablate_relbias=False, rel_buckets=16):
        super().__init__()
        self.attn = UnitaryEvolutionAttention(dim, n_heads, attn_dropout, window,
                                              rope_base, ablate_relbias, rel_buckets)
        self.ln1 = ComplexLayerNorm(dim)
        self.ffn = ComplexFFN(dim, ffn_mult, state_dropout, modrelu_bias)
        self.ln2 = ComplexLayerNorm(dim)
        self.p = state_dropout

    def forward(self, z, tpos, spk, mask):
        ctx = self.attn(z, tpos, spk, mask)
        z = self.ln1(z + complex_dropout(ctx, self.p, self.training))
        z = self.ln2(z + complex_dropout(self.ffn(z), self.p, self.training))
        return z


class _RealMHA(nn.Module):
    """Standard real multi-head attention used by the classical control blocks.

    Same validity/window/speaker masking and T5 relative bias as the quantum
    attention modules, so the ONLY difference vs UnitaryEvolution/OTOC attention is
    the kernel: plain scaled-dot-product softmax with NO unitary Heisenberg rotation
    and NO OTOC scattering gate. Operates on real tokens (the [Re;Im] lift of the
    complex state) and supports both self- (kv_in is q_in) and cross-attention.
    """

    def __init__(self, dim, n_heads, attn_dropout=0.1, window=0,
                 rel_buckets=16, ablate_relbias=False):
        super().__init__()
        assert dim % n_heads == 0
        self.h, self.dh = n_heads, dim // n_heads
        self.window = window
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wo = nn.Linear(dim, dim)
        self.spk_bias = nn.Parameter(torch.zeros(n_heads, 2))
        self.rel_bias = None if ablate_relbias else RelativeBias(n_heads, rel_buckets)
        self.p_attn = attn_dropout

    def forward(self, q_in, kv_in, spk, mask):
        B, T, _ = q_in.shape
        q = self.wq(q_in).view(B, T, self.h, self.dh)
        k = self.wk(kv_in).view(B, T, self.h, self.dh)
        v = self.wv(kv_in).view(B, T, self.h, self.dh)
        scores = torch.einsum("bihd,bjhd->bhij", q, k) / (self.dh ** 0.5)
        same = _same_speaker(spk)
        scores = scores + _speaker_bias(same, self.spk_bias)
        if self.rel_bias is not None:
            scores = scores + self.rel_bias(T, q_in.device)
        keep = _build_keep(mask, T, q_in.device, self.window)
        scores = scores.masked_fill(~keep, NEG_INF)
        attn = F.dropout(torch.softmax(scores, dim=-1), self.p_attn, self.training)
        out = torch.einsum("bhij,bjhd->bihd", attn, v).reshape(B, T, self.h * self.dh)
        return self.wo(out)


class ClassicalTemporalBlock(nn.Module):
    """Matched-parameter CLASSICAL control for UnitaryEvolutionBlock.

    A standard real Transformer encoder layer over the [Re;Im] lift of the complex
    state -- no unitary Heisenberg evolution kernel. param_match='arch' works at
    width 2*dim (equal-architecture; ~2x the quantum projection params, so if the
    quantum block wins it wins with FEWER params); 'count' works at width dim with
    in/out projections (~parameter parity with the quantum block). The output is
    re-lifted to a unit complex state so the rest of the (complex) pipeline is
    unchanged.
    """

    def __init__(self, dim, n_heads, ffn_mult=2, attn_dropout=0.1, state_dropout=0.1, window=0,
                 rel_buckets=16, param_match="arch", ablate_relbias=False):
        super().__init__()
        self.dim = dim
        dw = 2 * dim if param_match == "arch" else dim
        self.dw = dw
        self.inp = nn.Identity() if dw == 2 * dim else nn.Linear(2 * dim, dw)
        self.outp = nn.Identity() if dw == 2 * dim else nn.Linear(dw, 2 * dim)
        self.attn = _RealMHA(dw, n_heads, attn_dropout, window, rel_buckets, ablate_relbias)
        self.ln1 = nn.LayerNorm(dw)
        self.ln2 = nn.LayerNorm(dw)
        self.ffn = nn.Sequential(nn.Linear(dw, ffn_mult * dw), nn.GELU(),
                                 nn.Dropout(state_dropout), nn.Linear(ffn_mult * dw, dw))
        self.p = state_dropout

    def forward(self, z, tpos, spk, mask):
        x = self.inp(torch.cat([z.real, z.imag], dim=-1))
        x = self.ln1(x + F.dropout(self.attn(x, x, spk, mask), self.p, self.training))
        x = self.ln2(x + F.dropout(self.ffn(x), self.p, self.training))
        re, im = self.outp(x).chunk(2, dim=-1)
        return unit_normalize(torch.complex(re, im))


class QuantumGRUCell(nn.Module):
    """Quantum-inspired complex GRU cell with a unit-norm (pure-state) hidden state.

    Each step: the hidden state first FREE-EVOLVES under a learned diagonal unitary
    D = diag(e^{i omega}) (norm-preserving, the same Heisenberg-evolution idea the
    attention kernel uses), then a gated complex update mixes it with an input-driven
    candidate, and the result is re-projected onto the unit sphere (|psi>=h/||h||).
    Gates (reset r, update u) are REAL in (0,1), derived from the magnitudes |x|,|h|.
    """

    def __init__(self, dim: int, rope_base: float = 10000.0):
        super().__init__()
        self.dim = dim
        # per-dimension eigen-frequencies of the diagonal unitary step-evolution
        self.omega = nn.Parameter(init_frequencies(1, dim, rope_base).squeeze(0))  # (dim,)
        self.wx = ComplexLinear(dim, dim)   # input -> candidate
        self.wh = ComplexLinear(dim, dim)   # reset-gated evolved hidden -> candidate
        self.gate = nn.Linear(2 * dim, 2 * dim)  # [|x|;|h_evolved|] -> (reset, update)
        self.act = ModReLU(dim, bias_init=-0.1)

    def forward_seq(self, x: torch.Tensor, mask: torch.Tensor, reverse: bool = False):
        """x: (B,T,dim) complex; mask: (B,T) bool. Returns (B,T,dim) complex hidden states."""
        B, T, D = x.shape
        h = x.new_zeros(B, D)
        rot = torch.complex(torch.cos(self.omega), torch.sin(self.omega))  # (dim,) unit modulus
        order = range(T - 1, -1, -1) if reverse else range(T)
        outs = [None] * T
        for t in order:
            xt = x[:, t]                                   # (B,D) complex
            h_ev = h * rot                                 # diagonal-unitary free evolution
            g = torch.sigmoid(self.gate(torch.cat([xt.abs(), h_ev.abs()], dim=-1)))
            r, u = g.chunk(2, dim=-1)                      # (B,D) each, real in (0,1)
            cand = self.act(self.wx(xt) + self.wh(r * h_ev))   # (B,D) complex candidate
            h_new = unit_normalize((1 - u) * h_ev + u * cand)  # gated update, back to unit sphere
            mf = mask[:, t].unsqueeze(-1).to(x.real.dtype)     # (B,1): only advance valid positions
            h = mf * h_new + (1 - mf) * h
            outs[t] = h
        return torch.stack(outs, dim=1)                    # (B,T,dim)


class QuantumRNNBlock(nn.Module):
    """Recurrent counterpart of UnitaryEvolutionBlock: a bidirectional quantum (complex,
    unit-norm) GRU replaces self-attention for intra-modal temporal mixing, then the same
    residual + ComplexFFN + ComplexLayerNorm wrapping. Drop-in for _make_temporal so it is
    directly comparable to the attention temporal block (quantum-recurrent vs quantum-attention).
    Signature mirrors UnitaryEvolutionBlock; n_heads/window/rel_bias are accepted but unused
    (recurrence has no heads/relative-position table)."""

    def __init__(self, dim, n_heads=8, ffn_mult=2, attn_dropout=0.1, state_dropout=0.1, window=0,
                 modrelu_bias=0.0, rope_base=10000.0, ablate_relbias=False, rel_buckets=16):
        super().__init__()
        self.fwd = QuantumGRUCell(dim, rope_base)
        self.bwd = QuantumGRUCell(dim, rope_base)
        self.merge = ComplexLinear(2 * dim, dim)
        self.ln1 = ComplexLayerNorm(dim)
        self.ffn = ComplexFFN(dim, ffn_mult, state_dropout, modrelu_bias)
        self.ln2 = ComplexLayerNorm(dim)
        self.p = state_dropout

    def forward(self, z, tpos, spk, mask):
        hf = self.fwd.forward_seq(z, mask, reverse=False)
        hb = self.bwd.forward_seq(z, mask, reverse=True)
        ctx = self.merge(torch.cat([hf, hb], dim=-1))
        z = self.ln1(z + complex_dropout(ctx, self.p, self.training))
        z = self.ln2(z + complex_dropout(self.ffn(z), self.p, self.training))
        return z


class RNNTemporalBlock(nn.Module):
    """Standard recurrent temporal-encoder baseline (GRU / LSTM) for UnitaryEvolutionBlock.

    A conventional DialogueRNN-style sequence encoder: operate on the [Re;Im] lift of the complex
    state, run a bidirectional GRU/LSTM over the dialogue (packed by mask), then residual + LayerNorm
    + FFN; re-lift the output to a unit complex state so the rest of the (complex) cross-modal/Born
    pipeline is unchanged. Lets us contrast the quantum-attention evolution against the classic
    recurrent encoders (GRU/LSTM) head-to-head. Signature mirrors UnitaryEvolutionBlock (n_heads /
    window / rope / rel args accepted but unused -- recurrence has no heads/relative-position table).
    """

    def __init__(self, dim, n_heads=8, ffn_mult=2, attn_dropout=0.1, state_dropout=0.1, window=0,
                 modrelu_bias=0.0, rope_base=10000.0, ablate_relbias=False, rel_buckets=16, kind="gru"):
        super().__init__()
        self.dim = dim
        rnn_cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn_cls(2 * dim, dim, batch_first=True, bidirectional=True)
        self.outp = nn.Linear(2 * dim, 2 * dim)        # bidirectional hidden (2*dim) -> [Re;Im] (2*dim)
        self.ln1 = nn.LayerNorm(2 * dim)
        self.ffn = nn.Sequential(nn.Linear(2 * dim, ffn_mult * 2 * dim), nn.GELU(),
                                 nn.Dropout(state_dropout), nn.Linear(ffn_mult * 2 * dim, 2 * dim))
        self.ln2 = nn.LayerNorm(2 * dim)
        self.p = state_dropout

    def forward(self, z, tpos, spk, mask):
        x = torch.cat([z.real, z.imag], dim=-1)        # (B,T,2dim)
        lengths = mask.sum(1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        x = self.ln1(x + F.dropout(self.outp(out), self.p, self.training))
        x = self.ln2(x + F.dropout(self.ffn(x), self.p, self.training))
        re, im = x.chunk(2, dim=-1)
        return unit_normalize(torch.complex(re, im))


class RelGraphLayer(nn.Module):
    """One multi-head relational graph-attention layer over a dialogue
    (DialogueGCN/MMGCN family) + residual FFN.

    Edges are typed by speaker relation (same/different) x temporal direction
    (past/future) within a window. Attention q/k is shared but the value
    projection W_r is per-relation, so each relation type propagates a distinct
    message -- the relational inductive bias that decorrelates this branch from
    the temporal-attention quantum pipeline.
    """

    def __init__(self, d: int, n_rel: int = 4, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d % heads == 0
        self.d, self.n_rel, self.h, self.dh = d, n_rel, heads, d // heads
        self.wq = nn.Linear(d, d)
        self.wk = nn.Linear(d, d)
        self.wv = nn.ModuleList([nn.Linear(d, d) for _ in range(n_rel)])
        self.w_self = nn.Linear(d, d)
        self.wo = nn.Linear(d, d)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, h, rel_masks):
        B, T, _ = h.shape
        q = self.wq(h).view(B, T, self.h, self.dh)
        k = self.wk(h).view(B, T, self.h, self.dh)
        scores = torch.einsum("bihd,bjhd->bhij", q, k) / (self.dh ** 0.5)   # (B,H,T,T)
        agg = h.new_zeros(B, T, self.h, self.dh)
        for r in range(self.n_rel):
            mr = rel_masks[r]                                              # (B,T,T)
            s = scores.masked_fill(~mr[:, None], -1e9)
            attn = torch.softmax(s, dim=-1) * mr.any(-1)[:, None, :, None]  # zero no-neighbour rows
            attn = self.drop(attn)
            vr = self.wv[r](h).view(B, T, self.h, self.dh)
            agg = agg + torch.einsum("bhij,bjhd->bihd", attn, vr)
        out = self.w_self(h) + self.wo(agg.reshape(B, T, self.d))
        h = self.ln1(h + self.drop(out))
        h = self.ln2(h + self.ffn(h))
        return h


class RelationalGraphBranch(nn.Module):
    """Speaker-relational graph network producing an INDEPENDENT emotion-logit
    stream with a *different inductive bias* (relational graph + sequential GRU
    context, not temporal/cross-modal unitary attention) to move the shared
    decision boundary that caps the quantum pipeline. Node features carry speaker
    identity and a bi-GRU sequence pre-context (DialogueRNN-style), then multi-head
    relational graph-attention layers, then an independent readout (trained with
    its own loss so the stream is decorrelated, not deferential)."""

    def __init__(self, in_dim: int, d_graph: int, n_classes: int, n_speakers: int,
                 layers: int = 2, window: int = 10, dropout: float = 0.3, heads: int = 4):
        super().__init__()
        self.window = window
        self.norm_in = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, d_graph)
        self.spk_emb = nn.Linear(n_speakers, d_graph, bias=False)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(d_graph, d_graph // 2, bidirectional=True, batch_first=True)
        self.layers = nn.ModuleList([RelGraphLayer(d_graph, 4, heads, dropout) for _ in range(layers)])
        self.readout = nn.Sequential(nn.Linear(d_graph, d_graph), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(d_graph, n_classes))

    def _rel_masks(self, spk, mask, T, device):
        idx = torch.arange(T, device=device)
        diff = idx[None, :] - idx[:, None]                 # j - i
        within = diff.abs() <= self.window
        past = ((diff < 0) & within)[None]                 # (1,T,T)
        future = ((diff > 0) & within)[None]
        same = _same_speaker(spk).bool()                   # (B,T,T)
        validj = mask[:, None, :]                          # (B,1,T)
        m_sp, m_dp = same & validj, (~same) & validj
        return [m_sp & past, m_sp & future, m_dp & past, m_dp & future]

    def forward(self, x, spk, mask):
        """x: (B,T,in_dim) real concatenated modality features. -> (B,T,K) logits."""
        h = self.drop(self.proj(self.norm_in(x)) + self.spk_emb(spk))
        lengths = mask.sum(1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        rel_masks = self._rel_masks(spk, mask, h.shape[1], h.device)
        for gl in self.layers:
            h = gl(h, rel_masks)
        return self.readout(h)


class OTOCCrossAttention(nn.Module):
    """Directed cross-modal attention (source n -> target m) gated by an OTOC kernel.

    For each pair (i in target stream, j in source stream) and head h:
        w_i = normalize(P_w psi_i^m),   v_j = normalize(P_v psi_j^n)
        s_ij = |<w_i | D(t_j - t_i) | v_j>|^2          (in [0, 1])
        otoc_ij = 4 s_ij (1 - s_ij)                    (in [0, 1])
    which is exactly (d/8) * the squared commutator of the time-evolved
    Householder reflections built from w and v.  The attention logits are
        Re<q_i|D|k_j>/sqrt(Dh) + gamma_h * otoc_ij + speaker bias.
    """

    def __init__(self, dim: int, n_heads: int, attn_dropout: float = 0.1, window: int = 0,
                 rope_base: float = 10000.0, otoc_poly: bool = False, otoc_ops: int = 1,
                 ablate_otoc: bool = False, ablate_relbias: bool = False, rel_buckets: int = 16):
        super().__init__()
        assert dim % n_heads == 0
        self.h = n_heads
        self.dh = dim // n_heads
        self.window = window
        self.n_ops = otoc_ops
        self.ablate_otoc = ablate_otoc
        self.wq = ComplexLinear(dim, dim)
        self.wk = ComplexLinear(dim, dim)
        self.wv = ComplexLinear(dim, dim)
        if not ablate_otoc:
            # L operator pairs per head ("thermal ensemble" of local observables)
            self.w_op = ComplexLinear(dim, dim * otoc_ops)
            self.v_op = ComplexLinear(dim, dim * otoc_ops)
            self.gamma = nn.Parameter(torch.ones(n_heads))
            # learnable scattering kernel: g(s) = a*s + b*s^2 + c*4s(1-s); init = pure OTOC
            self.poly = nn.Parameter(torch.tensor([0.0, 0.0, 1.0]).repeat(n_heads, 1)) if otoc_poly else None
        self.wo = ComplexLinear(dim, dim)
        self.omega = nn.Parameter(init_frequencies(n_heads, self.dh, rope_base))
        self.spk_bias = nn.Parameter(torch.zeros(n_heads, 2))
        self.rel_bias = None if ablate_relbias else RelativeBias(n_heads, rel_buckets)
        self.p_attn = attn_dropout

    def forward(self, z_tgt, z_src, tpos, spk, mask):
        B, T, _ = z_tgt.shape
        q = _split_heads(self.wq(z_tgt), self.h)
        k = _split_heads(self.wk(z_src), self.h)
        v = _split_heads(self.wv(z_src), self.h)

        rot = make_rotation(tpos, self.omega)
        qr, kr = q * rot, k * rot

        scores = torch.einsum("bihd,bjhd->bhij", qr.conj(), kr).real / (self.dh ** 0.5)
        keep = _build_keep(mask, z_tgt.shape[1], z_tgt.device, self.window)

        if self.ablate_otoc:
            otoc_mean = scores.new_zeros(())
        else:
            L = self.n_ops
            w = unit_normalize(self.w_op(z_tgt).view(B, T, self.h, L, self.dh))
            u = unit_normalize(self.v_op(z_src).view(B, T, self.h, L, self.dh))
            wr, ur = w * rot.unsqueeze(3), u * rot.unsqueeze(3)
            overlap = torch.einsum("bihld,bjhld->bhlij", wr.conj(), ur)
            s = (overlap.real ** 2 + overlap.imag ** 2).clamp(0.0, 1.0)
            if self.poly is not None:
                g = (
                    self.poly[None, :, 0, None, None, None] * s
                    + self.poly[None, :, 1, None, None, None] * s ** 2
                    + self.poly[None, :, 2, None, None, None] * 4.0 * s * (1.0 - s)
                )
            else:
                g = 4.0 * s * (1.0 - s)  # normalised squared commutator, peak at s = 1/2
            otoc = g.mean(dim=2)  # thermal average over the operator ensemble
            # Diagnostic intervention used by scripts/ocs_intervention.py.  For each target
            # utterance, cyclically permute the OTOC responses only among valid source positions
            # inside its attention window.  This preserves the response distribution and all
            # learned parameters while breaking the source--target correspondence.  The default
            # path is unchanged when the attribute is absent.
            if getattr(self, "_otoc_intervention", "normal") == "shuffle":
                shuffled = otoc.clone()
                for b in range(B):
                    for i in range(T):
                        idx = torch.nonzero(keep[b, 0, i], as_tuple=False).flatten()
                        if idx.numel() > 1:
                            shift = max(1, idx.numel() // 2)
                            shuffled[b, :, i, idx] = otoc[b, :, i, torch.roll(idx, shifts=shift)]
                otoc = shuffled
            scores = scores + self.gamma[None, :, None, None] * otoc

        same = _same_speaker(spk)
        scores = scores + _speaker_bias(same, self.spk_bias)
        if self.rel_bias is not None:
            scores = scores + self.rel_bias(z_tgt.shape[1], z_tgt.device)
        scores = scores.masked_fill(~keep, NEG_INF)
        attn = torch.softmax(scores, dim=-1)
        # diagnostic (default OFF, no params, md5-stable): how much does the OTOC term reshape the
        # attention distribution? Compare attn (with OTOC) vs attn computed WITHOUT the gamma*otoc term.
        if getattr(self, "_diag", False) and not self.ablate_otoc:
            term = self.gamma[None, :, None, None] * otoc          # (B,H,i,j) the OTOC logit contribution
            attn_base = torch.softmax(scores - term, dim=-1)       # attention if OTOC were absent
            # full matrices for the O3 "OTOC reshapes cross-attention" figure (detached side-outputs,
            # diagnostic-only -> no params, md5-stable):
            self._diag_attn = attn.detach()                        # (B,H,i,j) attention WITH OTOC
            self._diag_attn_base = attn_base.detach()              # (B,H,i,j) attention WITHOUT OTOC
            self._diag_s = s.detach()                              # (B,H,L,i,j) overlap s_ij
            tv = 0.5 * (attn - attn_base).abs().sum(-1)            # (B,H,i) per-query total-variation
            qm = mask[:, None, :].to(tv.dtype)
            km = mask[:, None, None, :].to(scores.dtype)
            def _jstd(x):                                          # std across valid keys j, per (B,H,i)
                n = km.sum(-1).clamp_min(1)
                mu = (x * km).sum(-1, keepdim=True) / n.unsqueeze(-1)
                return ((((x - mu) ** 2) * km).sum(-1) / n).clamp_min(0).sqrt()
            self._diag_tv = float((tv * qm).sum() / qm.sum().clamp_min(1))
            self._diag_termstd = float((_jstd(term) * qm).sum() / qm.sum().clamp_min(1))
            self._diag_basestd = float((_jstd(scores - term) * qm).sum() / qm.sum().clamp_min(1))
            # value-vector homogeneity across valid source positions, per head:
            # ||mean_j v_j|| / mean_j||v_j|| in [0,1]. ->1 means all source values point the same way,
            # so reweighting attention barely changes out=sum_j attn_ij v_j (attention becomes inert).
            vn = torch.linalg.vector_norm(v, dim=-1)                 # (B,T,H)
            msrc = mask.to(vn.dtype)
            nsrc = msrc.sum(1).clamp_min(1)
            mv = (v * msrc[:, :, None, None]).sum(1) / nsrc[:, None, None]   # (B,H,dh) mean value
            mvn = torch.linalg.vector_norm(mv, dim=-1)              # (B,H)
            mean_vn = (vn * msrc[:, :, None]).sum(1) / nsrc[:, None]
            self._diag_vhomog = float((mvn / mean_vn.clamp_min(1e-6)).mean())
        # counterfactual: replace learned attention with UNIFORM over the kept (valid+windowed) keys
        if getattr(self, "_uniform", False):
            u = keep.to(attn.dtype)                                  # (B,1,T,T)
            attn = (u / u.sum(-1, keepdim=True).clamp_min(1)).expand(-1, self.h, -1, -1)
        attn = F.dropout(attn, self.p_attn, self.training)

        out = torch.complex(
            torch.einsum("bhij,bjhd->bihd", attn, v.real),
            torch.einsum("bhij,bjhd->bihd", attn, v.imag),
        )
        if not self.ablate_otoc:
            # per-(head, target) mean OTOC over valid source positions (diagnostic / regulariser hook)
            per_pos = (otoc * mask[:, None, None, :]).sum(-1) / mask.sum(-1).clamp_min(1)[:, None, None]  # (B,H,T)
            # stash a per-target-utterance scrambling map (mean over heads), detached side-output for
            # T1.5 OTOC analysis. NOT used in the forward computation -> zero effect on params/results.
            self._otoc_pos = per_pos.mean(1).detach()  # (B, T_target)
            otoc_mean = per_pos.mean()
        return self.wo(_merge_heads(out)), otoc_mean


class OTOCCrossModalBlock(nn.Module):
    """All directed pairs among modalities; messages summed into each target stream."""

    def __init__(self, modalities, dim, n_heads, ffn_mult=2, attn_dropout=0.1, state_dropout=0.1, window=0,
                 modrelu_bias=0.0, rope_base=10000.0, otoc_poly=False, otoc_ops=1,
                 ablate_otoc=False, ablate_relbias=False, rel_buckets=16):
        super().__init__()
        self.modalities = modalities
        self.pairs = [(m, n) for m in modalities for n in modalities if m != n]
        self.attn = nn.ModuleDict(
            {
                f"{m}<{n}": OTOCCrossAttention(dim, n_heads, attn_dropout, window,
                                               rope_base, otoc_poly, otoc_ops,
                                               ablate_otoc, ablate_relbias, rel_buckets)
                for m, n in self.pairs
            }
        )
        self.ln1 = nn.ModuleDict({m: ComplexLayerNorm(dim) for m in modalities})
        self.ffn = nn.ModuleDict({m: ComplexFFN(dim, ffn_mult, state_dropout, modrelu_bias) for m in modalities})
        self.ln2 = nn.ModuleDict({m: ComplexLayerNorm(dim) for m in modalities})
        self.p = state_dropout

    def forward(self, states, tpos, spk, mask, availability=None):
        """states: dict modality -> (B,T,D) complex.

        Returns (new_states, otoc_log, pair_states) where pair_states holds the
        per-pathway scattering outputs (separate "channel sectors" for readout).
        """
        otoc_log = {}
        new_states = {}
        pair_states = {}
        for m in self.modalities:
            msg = None
            for mt, n in self.pairs:
                if mt != m:
                    continue
                source_mask = mask if availability is None else mask & availability[n]
                out, om = self.attn[f"{m}<{n}"](
                    states[m], states[n], tpos, spk, source_mask
                )
                source_present = source_mask.any(dim=1)[:, None, None]
                target_present = (
                    mask if availability is None else mask & availability[m]
                )[:, :, None]
                out = out * source_present * target_present
                pair_states[f"{m}<{n}"] = out
                msg = out if msg is None else msg + out
                otoc_log[f"{m}<{n}"] = om
            if getattr(self, "_diag", False) and msg is not None:
                # relative magnitude of the cross-modal message vs the residual state it is added to
                mm = torch.linalg.vector_norm(msg, dim=-1)
                sm = torch.linalg.vector_norm(states[m], dim=-1).clamp_min(1e-6)
                vmask = mask.to(mm.dtype)
                if not hasattr(self, "_diag_msgratio"):
                    self._diag_msgratio = {}
                self._diag_msgratio[m] = float((mm / sm * vmask).sum() / vmask.sum().clamp_min(1))
            zm = states[m] if msg is None else states[m] + complex_dropout(msg, self.p, self.training)
            zm = self.ln1[m](zm)
            zm = self.ln2[m](zm + complex_dropout(self.ffn[m](zm), self.p, self.training))
            new_states[m] = zm
        return new_states, otoc_log, pair_states


class ClassicalCrossModalBlock(nn.Module):
    """Matched-parameter CLASSICAL control for OTOCCrossModalBlock.

    Standard real cross-attention among modalities (no OTOC kernel, no unitary
    rotation, no Householder operators). Same directed-pair / hub topology and the
    same (new_states, otoc_log, pair_states) return contract -- otoc_log is empty
    ({}), since there is no scrambling diagnostic in the classical control. Operates
    on the [Re;Im] lift of the complex states; param_match as in ClassicalTemporalBlock.
    """

    def __init__(self, modalities, dim, n_heads, ffn_mult=2, attn_dropout=0.1, state_dropout=0.1,
                 window=0, rel_buckets=16, param_match="arch", ablate_relbias=False):
        super().__init__()
        self.modalities = modalities
        self.pairs = [(m, n) for m in modalities for n in modalities if m != n]
        dw = 2 * dim if param_match == "arch" else dim
        self.dw = dw
        self.inp = nn.ModuleDict(
            {m: (nn.Identity() if dw == 2 * dim else nn.Linear(2 * dim, dw)) for m in modalities})
        self.attn = nn.ModuleDict(
            {f"{m}<{n}": _RealMHA(dw, n_heads, attn_dropout, window, rel_buckets, ablate_relbias)
             for m, n in self.pairs})
        self.ln1 = nn.ModuleDict({m: nn.LayerNorm(dw) for m in modalities})
        self.ln2 = nn.ModuleDict({m: nn.LayerNorm(dw) for m in modalities})
        self.ffn = nn.ModuleDict(
            {m: nn.Sequential(nn.Linear(dw, ffn_mult * dw), nn.GELU(),
                              nn.Dropout(state_dropout), nn.Linear(ffn_mult * dw, dw))
             for m in modalities})
        self.proj = nn.ModuleDict({m: nn.Linear(dw, 2 * dim) for m in modalities})
        self.p = state_dropout

    def _to_complex(self, m, x):
        re, im = self.proj[m](x).chunk(2, dim=-1)
        return torch.complex(re, im)

    def forward(self, states, tpos, spk, mask, availability=None):
        toks = {m: self.inp[m](torch.cat([states[m].real, states[m].imag], dim=-1))
                for m in self.modalities}
        pair_states = {}
        msgs = {m: None for m in self.modalities}
        for m, n in self.pairs:
            source_mask = mask if availability is None else mask & availability[n]
            ctx = self.attn[f"{m}<{n}"](
                toks[m], toks[n], spk, source_mask
            )   # (B,T,dw) real
            source_present = source_mask.any(dim=1)[:, None, None]
            target_present = (
                mask if availability is None else mask & availability[m]
            )[:, :, None]
            ctx = ctx * source_present * target_present
            pair_states[f"{m}<{n}"] = self._to_complex(m, ctx)
            msgs[m] = ctx if msgs[m] is None else msgs[m] + ctx
        new_states = {}
        for m in self.modalities:
            x = toks[m]
            if msgs[m] is not None:
                x = self.ln1[m](x + F.dropout(msgs[m], self.p, self.training))
            x = self.ln2[m](x + F.dropout(self.ffn[m](x), self.p, self.training))
            new_states[m] = unit_normalize(self._to_complex(m, x))
        return new_states, {}, pair_states


class DensityMixtureFusion(nn.Module):
    """Build the per-utterance mixed state rho = sum_m alpha_m |psi_m><psi_m| (factored form)."""

    def __init__(self, modalities, dim, gate_hidden=128, dropout=0.1, ablate_gate=False, text_bias=0.0):
        super().__init__()
        self.modalities = modalities
        self.ablate_gate = ablate_gate
        # fixed additive logit bias toward the text sector. Motivation: the learned
        # gate is audio-dominant, yet audio is the noisiest modality (dead dims / IS10)
        # -> biasing the mixture toward the clean RoBERTa text sector may reduce the
        # train/test gap. 0.0 = no-op (exact baseline).
        bias = torch.zeros(len(modalities))
        if text_bias != 0.0 and "t" in modalities:
            bias[modalities.index("t")] = text_bias
        self.register_buffer("gate_bias", bias)
        if not ablate_gate:
            self.gate = nn.Sequential(
                nn.Linear(len(modalities) * (dim + 1), gate_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(gate_hidden, len(modalities)),
            )

    def forward(self, states, intensities, availability=None):
        """states: dict -> (B,T,D) complex unit vectors; intensities: dict -> (B,T,1) real.

        Returns
          psis:   (B,T,M,D) complex  - the mixture components
          alpha:  (B,T,M) real       - mixture weights (sum to 1)
          purity: (B,T) real         - Tr[rho^2] in (0, 1]
        """
        psis = torch.stack([unit_normalize(states[m]) for m in self.modalities], dim=2)
        if availability is not None:
            psis = torch.where(availability.unsqueeze(-1), psis, torch.zeros_like(psis))
        if self.ablate_gate:
            B, T, M, _ = psis.shape
            if availability is None:
                alpha = torch.full(
                    (B, T, M), 1.0 / M, device=psis.device, dtype=psis.real.dtype
                )
            else:
                available_weight = availability.to(psis.real.dtype)
                available_count = available_weight.sum(-1, keepdim=True)
                uniform = torch.full_like(available_weight, 1.0 / M)
                alpha = torch.where(
                    available_count > 0,
                    available_weight / available_count.clamp_min(1.0),
                    uniform,
                )
        else:
            state_magnitudes = [torch.abs(states[m]) for m in self.modalities]
            state_intensities = [intensities[m] for m in self.modalities]
            if availability is not None:
                state_magnitudes = [
                    value * availability[:, :, index : index + 1]
                    for index, value in enumerate(state_magnitudes)
                ]
                state_intensities = [
                    value * availability[:, :, index : index + 1]
                    for index, value in enumerate(state_intensities)
                ]
            mags = torch.cat(state_magnitudes + state_intensities, dim=-1)
            logits = self.gate(mags) + self.gate_bias
            if availability is not None:
                logits = logits.masked_fill(
                    ~availability, torch.finfo(logits.dtype).min
                )
            alpha = torch.softmax(logits, dim=-1)
            if availability is not None:
                # Multiplying and renormalising makes unavailable sectors exactly
                # zero (rather than merely relying on softmax underflow).  The
                # uniform fallback only applies to padding/all-missing positions,
                # which are excluded from every supervised objective.
                alpha = alpha * availability.to(alpha.dtype)
                normalizer = alpha.sum(-1, keepdim=True)
                uniform = torch.full_like(alpha, 1.0 / alpha.size(-1))
                alpha = torch.where(
                    normalizer > 0,
                    alpha / normalizer.clamp_min(torch.finfo(alpha.dtype).tiny),
                    uniform,
                )

        overlap = torch.einsum("btmd,btnd->btmn", psis.conj(), psis)
        ov2 = overlap.real ** 2 + overlap.imag ** 2
        purity = torch.einsum("btm,btn,btmn->bt", alpha, alpha, ov2)
        return psis, alpha, purity


class BornReadout(nn.Module):
    """POVM readout: p(k) = Tr[M_k rho], M_k = A_k^dagger A_k (rank-r factors)."""

    def __init__(self, dim: int, n_classes: int, rank: int = 32):
        super().__init__()
        self.A_re = nn.Parameter(torch.randn(n_classes, rank, dim) / dim ** 0.5)
        self.A_im = nn.Parameter(torch.randn(n_classes, rank, dim) / dim ** 0.5)
        # learnable measurement sharpness ("power Born"): log p -> log p / tau
        self.log_tau = nn.Parameter(torch.zeros(()))

    def forward(self, psis: torch.Tensor, alpha: torch.Tensor = None) -> torch.Tensor:
        """psis: (B,T,M,D) complex; alpha: (B,T,M) or None (pure state, M=1 expected).

        Returns log-probabilities (B,T,K) by the Born rule.
        """
        A = torch.complex(self.A_re, self.A_im)  # (K, R, D)
        y = torch.einsum("krd,btmd->btmkr", A, psis)
        e = (y.real ** 2 + y.imag ** 2).sum(-1)  # (B,T,M,K) = ||A_k psi_m||^2
        if alpha is None:
            p = e.mean(dim=2)
        else:
            p = torch.einsum("btm,btmk->btk", alpha, e)
        p = p.clamp_min(1e-10)
        logp = torch.log(p / p.sum(-1, keepdim=True))
        tau = torch.exp(self.log_tau).clamp(0.05, 20.0)
        return torch.log_softmax(logp / tau, dim=-1)


class EntangledBornReadout(nn.Module):
    """Born measurement combining coherent superposition and entangled-projector POVMs.

    (i) Coherent (direct-sum) term.  The modalities span the direct-sum space
        H_t (+) H_a (+) H_v; the fused state is the sector superposition
            |Psi> = sqrt(a_t)|psi_t> (+) sqrt(a_a)|psi_a> (+) sqrt(a_v)|psi_v>,
        automatically unit-norm because sum_m a_m = 1 and each |psi_m> is unit.
        p(k) = ||A_k Psi||^2 with A_k in C^{R x Md}.  The off-block parts of
        M_k = A_k^dag A_k are first-order cross-modal *interference* terms -
        strictly more expressive than the probability mixture
        sum_m a_m ||A_k psi_m||^2 (its block-diagonal special case).

    (ii) Entangled (tensor-product) term.  M_k = sum_r |phi_kr><phi_kr| with
        |phi_kr> = a_kr (x) b_kr (x) c_kr on H_t (x) H_a (x) H_v; on the product
        state the Born probability factorises:
            p(k) = sum_r | <a_kr|psi_t> <b_kr|psi_a> <c_kr|psi_v> |^2,
        capturing conjunctive trimodal evidence without ever building d^3 objects.

    Both probabilities are convexly combined (learnable weight) and sharpened
    by a learnable measurement temperature ("power Born").
    """

    def __init__(self, dim: int, n_classes: int, n_modalities: int, mix_rank: int = 32,
                 joint_rank: int = 64, coherence_init: float = None, n_cp: int = None,
                 joint_init: float = 0.0, joint_freeze: bool = False, per_class_tau: bool = False):
        super().__init__()
        self.dim = dim
        self.n_cp = n_cp or n_modalities  # CP entangled term acts on the first n_cp sectors
        D = n_modalities * dim
        self.A_re = nn.Parameter(torch.randn(n_classes, mix_rank, D) / D ** 0.5)
        self.A_im = nn.Parameter(torch.randn(n_classes, mix_rank, D) / D ** 0.5)
        self.F_re = nn.Parameter(torch.randn(self.n_cp, n_classes, joint_rank, dim) / dim ** 0.5)
        self.F_im = nn.Parameter(torch.randn(self.n_cp, n_classes, joint_rank, dim) / dim ** 0.5)
        # joint_init biases the entangled<->coherent blend at init: w=sigmoid(joint_init).
        # Ablation finding (2026-06-19): the entangled CP term is load-bearing (+2.4 IE/+1.1 ME);
        # the gate settles on a flat 0.5-1.0 plateau, so IEMOCAP never climbs to its w~1 optimum.
        # joint_init>0 starts in that basin; joint_freeze pins it for a clean pure-entangled run.
        self.joint_weight = nn.Parameter(torch.tensor(float(joint_init)), requires_grad=not joint_freeze)
        # per-class measurement temperature (PCTau): recalibrates each class's decision
        # sharpness. Under support-weighted WF1 the gradient preferentially sharpens the
        # high-support boundaries (neutral/joy/anger), where the metric actually lives.
        # zeros at init -> exact baseline (single global log_tau only).
        self.class_log_tau = nn.Parameter(torch.zeros(n_classes)) if per_class_tau else None
        self.log_tau = nn.Parameter(torch.zeros(()))
        # learnable decoherence: interpolates Tr[M_k rho] (mixture) <-> coherent ||A Psi||^2
        self.coherence = nn.Parameter(torch.tensor(float(coherence_init))) if coherence_init is not None else None

    def forward(
        self,
        psis: torch.Tensor,
        alpha: torch.Tensor,
        availability: torch.Tensor = None,
    ) -> torch.Tensor:
        """psis: (B,T,M,D) complex unit states; alpha: (B,T,M)."""
        B, T, M, D = psis.shape
        # coherent superposition over the direct-sum space
        Psi = (torch.sqrt(alpha.clamp_min(1e-8)).unsqueeze(-1) * psis).reshape(B, T, M * D)
        A = torch.complex(self.A_re, self.A_im)            # (K, R, M*D)
        y = torch.einsum("krd,btd->btkr", A, Psi)
        p_sup = (y.real ** 2 + y.imag ** 2).sum(-1).clamp_min(1e-12)
        p_sup = p_sup / p_sup.sum(-1, keepdim=True)

        if self.coherence is not None:
            # block-diagonal (decohered) part of the SAME measurement: Tr[M_k rho]
            Av = A.view(A.shape[0], A.shape[1], M, self.dim)
            ym = torch.einsum("krmd,btmd->btmkr", Av, psis)
            e = (ym.real ** 2 + ym.imag ** 2).sum(-1)      # (B,T,M,K)
            p_mix = torch.einsum("btm,btmk->btk", alpha, e).clamp_min(1e-12)
            p_mix = p_mix / p_mix.sum(-1, keepdim=True)
            c = torch.sigmoid(self.coherence)
            p_sup = (1 - c) * p_mix + c * p_sup

        # entangled joint term (over the first n_cp sectors = the modality states)
        F = torch.complex(self.F_re, self.F_im)            # (n_cp,K,R,D)
        proj = torch.einsum("mkrd,btmd->btmkr", F, psis[:, :, : self.n_cp])  # (B,T,n_cp,K,R)
        if availability is not None:
            present = availability[:, :, : self.n_cp, None, None]
            proj = torch.where(present, proj, torch.ones_like(proj))
        prod = proj[:, :, 0]
        for m in range(1, proj.shape[2]):
            prod = prod * proj[:, :, m]
        p_joint = (prod.real ** 2 + prod.imag ** 2).sum(-1).clamp_min(1e-12)  # (B,T,K)
        p_joint = p_joint / p_joint.sum(-1, keepdim=True)

        w = torch.sigmoid(self.joint_weight)
        p = (1 - w) * p_sup + w * p_joint
        if self.class_log_tau is not None:
            tau = torch.exp(self.log_tau + self.class_log_tau).clamp(0.05, 20.0)  # (K,) per-class
        else:
            tau = torch.exp(self.log_tau).clamp(0.05, 20.0)
        return torch.log_softmax(torch.log(p.clamp_min(1e-12)) / tau, dim=-1)


class MLPReadout(nn.Module):
    """Matched-parameter CLASSICAL control for EntangledBornReadout.

    A plain MLP over the SAME fused state the Born head measures: the sector
    superposition Psi = sqrt(alpha) * psis (the sqrt(alpha) weighting is KEPT so the
    DensityMixtureFusion gate still receives gradient), lifted to real [Re;Im]. No
    Born rule, no POVM, no entangled CP term. hidden=0 auto-matches the Born head's
    parameter budget (born_param_ref). Signature mirrors EntangledBornReadout so it
    is a drop-in at the readout call site.
    """

    def __init__(self, dim, n_classes, n_modalities, hidden=0, dropout=0.1, born_param_ref=0):
        super().__init__()
        D = n_modalities * dim
        if hidden <= 0:
            hidden = max(1, round(born_param_ref / (2 * D + n_classes))) if born_param_ref > 0 else 2 * D
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(2 * D, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, psis, alpha):
        B, T, M, D = psis.shape
        Psi = (alpha.clamp_min(1e-8).sqrt().unsqueeze(-1) * psis).reshape(B, T, M * D)
        x = torch.cat([Psi.real, Psi.imag], dim=-1)
        return torch.log_softmax(self.net(x), dim=-1)


class MetricReadout(nn.Module):
    """Normalised metric-learning (cosine / prototype) classifier baseline for EntangledBornReadout.

    Over the SAME fused state Psi = sqrt(alpha) * psis lifted to real [Re;Im]: L2-normalise the
    feature and K learnable L2-normalised class prototypes, then logits = s * cos(feature, prototype)
    with a learnable scale s (temperature) -- the standard normalised-softmax / NormFace / prototype
    head. No Born rule, no cross-modal interference, no entangled CP term. Directly tests whether the
    quantum measurement structure beats a plain normalised-similarity classifier on the identical
    fused representation. Signature mirrors EntangledBornReadout / MLPReadout (hidden / born_param_ref
    accepted for parity, unused)."""

    def __init__(self, dim, n_classes, n_modalities, hidden=0, dropout=0.1, born_param_ref=0):
        super().__init__()
        D = n_modalities * dim
        self.proto = nn.Parameter(torch.randn(n_classes, 2 * D) / (2 * D) ** 0.5)
        self.log_scale = nn.Parameter(torch.tensor(2.3026))   # exp(2.3026) ~ 10: cosine-softmax temperature
        self.drop = nn.Dropout(dropout)

    def forward(self, psis, alpha):
        B, T, M, D = psis.shape
        Psi = (alpha.clamp_min(1e-8).sqrt().unsqueeze(-1) * psis).reshape(B, T, M * D)
        x = self.drop(torch.cat([Psi.real, Psi.imag], dim=-1))
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.proto, dim=-1)
        logits = self.log_scale.exp() * (x @ w.t())
        return torch.log_softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Complex-but-NON-quantum control blocks (reviewer "is it just complex-valued?" defense).
#
# These keep the FULL complex-valued machinery (ComplexLinear / ComplexLayerNorm /
# ComplexFFN -> SAME number field and ~same parameter budget as the quantum blocks) but
# REMOVE the three quantum-physics kernels:
#   - the diagonal-unitary Heisenberg evolution rotation D(t)=e^{iHt} (no make_rotation/omega),
#   - the OTOC squared-commutator scrambling gate (no Householder operators / 4s(1-s)),
#   - the Born POVM / entangled-CP readout (replaced by a complex MLP + real logit head).
# So vs the REAL classical control they differ ONLY by the number field (complex vs real),
# and vs the QUANTUM model they differ ONLY by the quantum structure. The 3-way
# quantum > complex > classical (if the thesis holds) cleanly attributes the gain to the
# quantum machinery, not to complex arithmetic. param_match is accepted for signature
# parity with the classical blocks but unused (parity is restored at the model level by
# widening --enc-hidden, the same method as the §2 param-matched ablations).
# ---------------------------------------------------------------------------
class _ComplexMHA(nn.Module):
    """Plain complex multi-head attention: the complex counterpart of _RealMHA.

    Same validity/window/speaker masking and T5 relative bias as the quantum attention,
    so the ONLY difference vs UnitaryEvolution/OTOC attention is the ABSENCE of the
    unitary Heisenberg rotation kernel and the OTOC scattering gate. score_ij =
    Re<q_i|k_j>/sqrt(Dh). Supports self- (kv_in is q_in) and cross-attention.
    """

    def __init__(self, dim, n_heads, attn_dropout=0.1, window=0,
                 rel_buckets=16, ablate_relbias=False):
        super().__init__()
        assert dim % n_heads == 0
        self.h, self.dh = n_heads, dim // n_heads
        self.window = window
        self.wq = ComplexLinear(dim, dim)
        self.wk = ComplexLinear(dim, dim)
        self.wv = ComplexLinear(dim, dim)
        self.wo = ComplexLinear(dim, dim)
        self.spk_bias = nn.Parameter(torch.zeros(n_heads, 2))
        self.rel_bias = None if ablate_relbias else RelativeBias(n_heads, rel_buckets)
        self.p_attn = attn_dropout

    def forward(self, q_in, kv_in, spk, mask):
        T = q_in.shape[1]
        q = _split_heads(self.wq(q_in), self.h)
        k = _split_heads(self.wk(kv_in), self.h)
        v = _split_heads(self.wv(kv_in), self.h)
        scores = torch.einsum("bihd,bjhd->bhij", q.conj(), k).real / (self.dh ** 0.5)
        same = _same_speaker(spk)
        scores = scores + _speaker_bias(same, self.spk_bias)
        if self.rel_bias is not None:
            scores = scores + self.rel_bias(T, q_in.device)
        keep = _build_keep(mask, T, q_in.device, self.window)
        scores = scores.masked_fill(~keep, NEG_INF)
        attn = F.dropout(torch.softmax(scores, dim=-1), self.p_attn, self.training)
        out = torch.complex(
            torch.einsum("bhij,bjhd->bihd", attn, v.real),
            torch.einsum("bhij,bjhd->bihd", attn, v.imag),
        )
        return self.wo(_merge_heads(out))


class ComplexTemporalBlock(nn.Module):
    """Complex-but-non-quantum control for UnitaryEvolutionBlock.

    Identical complex machinery (ComplexLinear/LayerNorm/FFN) -- so SAME number field
    and ~same params (minus the tiny omega bank) -- but a PLAIN complex Transformer
    self-attention with the unitary Heisenberg rotation kernel REMOVED.
    """

    def __init__(self, dim, n_heads, ffn_mult=2, attn_dropout=0.1, state_dropout=0.1, window=0,
                 modrelu_bias=0.0, rel_buckets=16, param_match="arch", ablate_relbias=False):
        super().__init__()
        self.attn = _ComplexMHA(dim, n_heads, attn_dropout, window, rel_buckets, ablate_relbias)
        self.ln1 = ComplexLayerNorm(dim)
        self.ffn = ComplexFFN(dim, ffn_mult, state_dropout, modrelu_bias)
        self.ln2 = ComplexLayerNorm(dim)
        self.p = state_dropout

    def forward(self, z, tpos, spk, mask):
        ctx = self.attn(z, z, spk, mask)
        z = self.ln1(z + complex_dropout(ctx, self.p, self.training))
        z = self.ln2(z + complex_dropout(self.ffn(z), self.p, self.training))
        return z


class ComplexCrossModalBlock(nn.Module):
    """Complex-but-non-quantum control for OTOCCrossModalBlock.

    Plain complex cross-attention among modalities (NO OTOC scrambling kernel, NO
    unitary rotation), same directed-pair / hub topology and the same
    (new_states, otoc_log, pair_states) return contract -- otoc_log is empty ({}).
    The complex counterpart of ClassicalCrossModalBlock (which lifts to real [Re;Im]).
    """

    def __init__(self, modalities, dim, n_heads, ffn_mult=2, attn_dropout=0.1, state_dropout=0.1,
                 window=0, modrelu_bias=0.0, rel_buckets=16, param_match="arch", ablate_relbias=False):
        super().__init__()
        self.modalities = modalities
        self.pairs = [(m, n) for m in modalities for n in modalities if m != n]
        self.attn = nn.ModuleDict(
            {f"{m}<{n}": _ComplexMHA(dim, n_heads, attn_dropout, window, rel_buckets, ablate_relbias)
             for m, n in self.pairs})
        self.ln1 = nn.ModuleDict({m: ComplexLayerNorm(dim) for m in modalities})
        self.ffn = nn.ModuleDict({m: ComplexFFN(dim, ffn_mult, state_dropout, modrelu_bias) for m in modalities})
        self.ln2 = nn.ModuleDict({m: ComplexLayerNorm(dim) for m in modalities})
        self.p = state_dropout

    def forward(self, states, tpos, spk, mask, availability=None):
        otoc_log = {}
        new_states = {}
        pair_states = {}
        for m in self.modalities:
            msg = None
            for mt, n in self.pairs:
                if mt != m:
                    continue
                source_mask = mask if availability is None else mask & availability[n]
                out = self.attn[f"{m}<{n}"](
                    states[m], states[n], spk, source_mask
                )
                source_present = source_mask.any(dim=1)[:, None, None]
                target_present = (
                    mask if availability is None else mask & availability[m]
                )[:, :, None]
                out = out * source_present * target_present
                pair_states[f"{m}<{n}"] = out
                msg = out if msg is None else msg + out
            zm = states[m] if msg is None else states[m] + complex_dropout(msg, self.p, self.training)
            zm = self.ln1[m](zm)
            zm = self.ln2[m](zm + complex_dropout(self.ffn[m](zm), self.p, self.training))
            new_states[m] = zm
        return new_states, otoc_log, pair_states


class ComplexMLPReadout(nn.Module):
    """Complex-but-non-quantum control for EntangledBornReadout.

    A complex-valued MLP over the SAME fused sector-superposition state Psi (the
    sqrt(alpha) weighting is KEPT so the DensityMixtureFusion gate still gets gradient):
    ComplexLinear -> CGELU (split-GELU on Re/Im) -> real logit head over [Re;Im]. Keeps
    complex arithmetic in the body but uses NO Born rule, NO POVM, NO entangled CP term.
    hidden=0 auto-matches the Born head's parameter budget (born_param_ref).
    """

    def __init__(self, dim, n_classes, n_modalities, hidden=0, dropout=0.1, born_param_ref=0):
        super().__init__()
        D = n_modalities * dim
        if hidden <= 0:
            # complex hidden layer (2*D*hidden) + real head (2*hidden*K) ~= 2*hidden*(D+K)
            hidden = max(1, round(born_param_ref / (2 * (D + n_classes)))) if born_param_ref > 0 else D
        self.hidden = hidden
        self.fc1 = ComplexLinear(D, hidden)
        self.head = nn.Linear(2 * hidden, n_classes)
        self.p = dropout

    def forward(self, psis, alpha):
        B, T, M, D = psis.shape
        Psi = (alpha.clamp_min(1e-8).sqrt().unsqueeze(-1) * psis).reshape(B, T, M * D)
        h = self.fc1(Psi)
        h = torch.complex(F.gelu(h.real), F.gelu(h.imag))  # CGELU: phase-mixing, non-Born
        h = complex_dropout(h, self.p, self.training)
        x = torch.cat([h.real, h.imag], dim=-1)
        return torch.log_softmax(self.head(x), dim=-1)
