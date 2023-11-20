import torch
import torch.nn.functional as F


def max_norm_(module, c=4., return_module=False):
    """Applies a max-norm constraint on the weight of the passed module.
    Clamps the norm of the weight vector to the specified value if it exceeds the limit.
    The constraint is applied in-place on the module, thereby permanently changing the weights.

    Use this in the forward pass:
        def forward(self, x):
            x = max_norm_(self.layer1, c=4., return_module=True)(x)

    Args:
        module: A nn.Module instance, e.g. nn.Conv1d or nn.Linear
        c: The maximum constraint on the weight.
        return_module: Specify whether the module should be returned for convenience.
    """
    norms = module.weight.data.norm(dim=None, keepdim=True)
    desired = torch.clamp(norms, 0., c)
    module.weight.data = module.weight.data * (desired / (norms + 1e-6))
    if return_module:
        return module


def torch_hilbert_freq(x, forward_fourier=True):
    """Computes the Hilbert transform using PyTorch,
    with the real and imaginary parts as separate dimensions.

    Input shape (forward_fourier=True): (..., seq_len)
    Input shape (forward_fourier=False): (..., seq_len / 2 + 1, 2)
    Output shape: (..., seq_len, 2)
    """
    if forward_fourier:
        x = torch.fft.rfft(x, norm=None, dim=-1)
        x = torch.view_as_real(x)
    x = x * 2.
    x[..., 0, :] = x[..., 0, :] / 2.  # Don't multiply the DC-term by 2
    x = F.pad(x, [0, 0, 0, x.shape[-2] - 2])  # Fill Fourier coefficients to retain shape
    x = torch.view_as_complex(x)
    x = torch.fft.ifft(x, norm=None, dim=-1)  # returns complex signal
    x = torch.view_as_real(x)

    return x


def plv_time(x, forward_fourier=True):
    """PLV metric in time domain.
    x (..., channels, time/(freqs, 2)) -> (..., channels, channels)"""
    x_a = torch_hilbert_freq(x, forward_fourier)
    amp = torch.sqrt(x_a[..., 0] ** 2 + x_a[..., 1] ** 2 + 1e-6)
    x_u = x_a / amp.unsqueeze(-1)
    x_u_rr = torch.matmul(x_u[..., 0], x_u[..., 0].transpose(-2, -1))
    x_u_ii = torch.matmul(x_u[..., 1], x_u[..., 1].transpose(-2, -1))
    x_u_ri = torch.matmul(x_u[..., 0], x_u[..., 1].transpose(-2, -1))
    x_u_ir = torch.matmul(x_u[..., 1], x_u[..., 0].transpose(-2, -1))
    r = x_u_rr + x_u_ii
    i = x_u_ri - x_u_ir
    time = amp.shape[-1]
    plv = 1 / time * torch.sqrt(r ** 2 + i ** 2 + 1e-6)

    return plv
