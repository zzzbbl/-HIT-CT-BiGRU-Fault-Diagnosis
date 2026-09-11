from __future__ import annotations

import random

import torch


class SafeSignalAugmenter:
    """Signal augmentation used by the original notebooks.

    It adds random Gaussian noise, applies a small circular time shift, and
    scales each channel independently. The augmenter is intended for training
    samples only.
    """

    def __init__(
        self,
        noise_range: tuple[float, float] = (-5.0, 15.0),
        max_shift: float = 0.05,
        scale_range: tuple[float, float] = (0.8, 1.2),
        seed: int = 42,
    ) -> None:
        self.noise_range = noise_range
        self.max_shift = max_shift
        self.scale_range = scale_range
        self.rng = random.Random(seed)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            x = x.unsqueeze(0)

        time_dim = x.dim() - 1
        time_length = x.size(time_dim)
        shift_amount = int(time_length * self.rng.uniform(-self.max_shift, self.max_shift))

        snr_db = self.rng.uniform(*self.noise_range)
        signal_power = torch.mean(x**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        x = x + torch.randn_like(x) * torch.sqrt(noise_power)

        if shift_amount != 0:
            x = torch.roll(x, shifts=shift_amount, dims=time_dim)

        if x.dim() > 1:
            scales = torch.tensor(
                [self.rng.uniform(*self.scale_range) for _ in range(x.size(0))],
                dtype=x.dtype,
                device=x.device,
            )
            x = x * scales.view(-1, 1)

        return x


def add_awgn(inputs: torch.Tensor, snr_db: float) -> torch.Tensor:
    signal_power = torch.mean(inputs**2, dim=tuple(range(1, inputs.dim())), keepdim=True)
    noise_power = signal_power / (10 ** (snr_db / 10))
    return inputs + torch.randn_like(inputs) * torch.sqrt(noise_power)

