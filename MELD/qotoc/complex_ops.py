"""Complex-valued building blocks for the quantum-inspired model.

All learnable parameters are real tensors (so any optimizer works untouched);
activations are torch.complex64.  A complex linear map

    (x_re + i x_im)(W_re + i W_im)^T

is realised with two real nn.Linear layers.
"""
import math

import torch
import torch.nn as nn


def unit_normalize(z: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Project a complex vector onto the unit sphere: |psi> = z / ||z||."""
    norm = torch.sqrt((z.real ** 2 + z.imag ** 2).sum(dim=dim, keepdim=True).clamp_min(eps * eps))
    return z / norm


def complex_dropout(z: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Dropout with a shared mask for the real and imaginary parts."""
    if not training or p == 0.0:
        return z
    mask = torch.empty(z.shape, dtype=torch.float32, device=z.device).bernoulli_(1 - p) / (1 - p)
    return z * mask


class ComplexLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = True):
        super().__init__()
        self.re = nn.Linear(d_in, d_out, bias=bias)
        self.im = nn.Linear(d_in, d_out, bias=bias)
        # halve the effective gain so |Wz| matches real-valued init scale
        with torch.no_grad():
            self.re.weight.mul_(1 / math.sqrt(2))
            self.im.weight.mul_(1 / math.sqrt(2))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        zr, zi = z.real, z.imag
        return torch.complex(self.re(zr) - self.im(zi), self.re(zi) + self.im(zr))


class ModReLU(nn.Module):
    """modReLU(z) = relu(|z| + b) * z / |z| - phase-preserving nonlinearity.

    NOTE: bias_init = 0 makes this the identity map (relu(|z|)*z/|z| = z), i.e.
    no nonlinearity at all; a negative bias_init (e.g. -0.5) gates low-magnitude
    components and gives the FFN genuine nonlinear capacity.
    """

    def __init__(self, dim: int, bias_init: float = 0.0):
        super().__init__()
        self.bias = nn.Parameter(torch.full((dim,), bias_init))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(z)
        scale = torch.relu(mag + self.bias) / mag.clamp_min(1e-8)
        return z * scale


class ComplexLayerNorm(nn.Module):
    """LayerNorm on the complex plane: centre by complex mean, scale by RMS magnitude."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim, 2))
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mu = z.mean(dim=-1, keepdim=True)
        zc = z - mu
        var = (zc.real ** 2 + zc.imag ** 2).mean(dim=-1, keepdim=True)
        zn = zc / torch.sqrt(var + self.eps)
        return zn * self.gamma + torch.complex(self.beta[..., 0], self.beta[..., 1])


class ComplexFFN(nn.Module):
    """Position-wise feed-forward in the complex domain with modReLU."""

    def __init__(self, dim: int, mult: int = 2, dropout: float = 0.1, modrelu_bias: float = 0.0):
        super().__init__()
        self.fc1 = ComplexLinear(dim, dim * mult)
        self.act = ModReLU(dim * mult, modrelu_bias)
        self.fc2 = ComplexLinear(dim * mult, dim)
        self.p = dropout

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(z))
        h = complex_dropout(h, self.p, self.training)
        return self.fc2(h)


def make_rotation(t: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Diagonal unitary time evolution D(t) = diag(e^{i w_l t}).

    t:     (B, T) real positions
    omega: (H, Dh) learned Hamiltonian eigen-frequencies
    returns rot: (B, T, H, Dh) complex
    """
    phase = t[:, :, None, None] * omega[None, None, :, :]
    return torch.complex(torch.cos(phase), torch.sin(phase))


def init_frequencies(n_heads: int, d_head: int, base: float = 10000.0) -> torch.Tensor:
    """RoPE-style log-spaced frequencies, one bank per head (slightly perturbed)."""
    freqs = base ** (-torch.arange(d_head, dtype=torch.float32) / d_head)
    omega = freqs[None, :].repeat(n_heads, 1)
    omega = omega * (1.0 + 0.02 * torch.randn(n_heads, d_head))
    return omega
