import math

import torch

__all__ = [
    "build_convrot_matrix", "choose_convrot_group", "is_convrot_group", "rotate_activation", "rotate_weight",
]

_SEED = (
    (1.0, 1.0, 1.0, -1.0),
    (1.0, 1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0, 1.0),
)
_CACHE: dict[tuple, torch.Tensor] = {}


def is_convrot_group(value: object) -> bool:
    return (
        type(value) is int and value >= 4 and value & (value - 1) == 0
        and (value.bit_length() - 1) % 2 == 0
    )


def choose_convrot_group(in_features: int, requested: int = 256) -> int | None:
    if not is_convrot_group(requested):
        raise ValueError(f"convrot group size must be a power of four at least 4, got {requested}")
    group = requested
    while group >= 4 and in_features % group:
        group //= 4
    return group if group >= 4 else None


def build_convrot_matrix(order: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (order, device.type, device.index, dtype)
    matrix = _CACHE.get(key)
    if matrix is None:
        seed = torch.tensor(_SEED, dtype=torch.float32)
        matrix = seed
        for _ in range(int(round(math.log(order, 4))) - 1):
            matrix = torch.kron(matrix, seed)
        matrix = (matrix / math.sqrt(order)).to(device=device, dtype=dtype)
        _CACHE[key] = matrix
    return matrix


def rotate_activation(x: torch.Tensor, matrix: torch.Tensor, group: int) -> torch.Tensor:
    features = x.shape[-1]
    if features % group:
        raise ValueError(f"in_features {features} is not divisible by convrot group {group}")
    grouped = x.reshape(-1, features // group, group)
    return torch.matmul(grouped, matrix.to(grouped.dtype)).reshape(x.shape)


def rotate_weight(weight: torch.Tensor, matrix: torch.Tensor, group: int, *, inverse: bool = False) -> torch.Tensor:
    out_features, in_features = weight.shape
    if in_features % group:
        raise ValueError(f"in_features {in_features} is not divisible by convrot group {group}")
    grouped = weight.reshape(out_features, in_features // group, group)
    kernel = matrix if inverse else matrix.t()
    return torch.matmul(grouped, kernel.to(grouped.dtype)).reshape(out_features, in_features)
