import copy
import json
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from ..registry import get_scheme
from ..schemes import Scheme

#: Every component must start 16-byte aligned: the codecs read them with 16-byte vector loads.
_ALIGN = 16
_LENGTH_PREFIX_BYTES = 8

_REQUIRED_FIELDS = {"compress_method", "dtype", "shape", "buffer_names", "buffer_dtypes", "buffer_shapes", "header"}


def _parse_dtype(name: str) -> torch.dtype:
    if not name:
        raise TypeError("CompressedTensor serialized dtype must be a non-empty string")
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported serialized torch dtype '{name}'")
    return dtype


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _aligned(size: int) -> int:
    return -(-size // _ALIGN) * _ALIGN


def _metadata(header: dict, buffers: Mapping[str, torch.Tensor], shape, encoded_dtype: torch.dtype) -> dict:
    return {
        "compress_method": header["compress_method"],
        "dtype": _dtype_name(encoded_dtype),
        "shape": list(shape),
        "buffer_names": list(buffers),
        "buffer_dtypes": [_dtype_name(value.dtype) for value in buffers.values()],
        "buffer_shapes": [list(value.shape) for value in buffers.values()],
        "header": header,
    }


def _pack(metadata: dict, buffers: Mapping[str, torch.Tensor]) -> torch.Tensor:
    device = next(iter(buffers.values())).device
    raw = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    slots, offset = [], _aligned(_LENGTH_PREFIX_BYTES + len(raw))
    for value in buffers.values():
        slots.append((offset, value))
        offset = _aligned(offset + value.numel() * value.element_size())
    blob = torch.zeros(offset, dtype=torch.uint8, device=device)
    blob[:_LENGTH_PREFIX_BYTES].view(torch.uint64).fill_(len(raw))
    blob[_LENGTH_PREFIX_BYTES : _LENGTH_PREFIX_BYTES + len(raw)].copy_(torch.frombuffer(bytearray(raw), dtype=torch.uint8))
    for offset, value in slots:
        size = value.numel() * value.element_size()
        blob[offset : offset + size].view(value.dtype).view(value.shape).copy_(value)
    return blob


def _parse_quant_state(quant_state: torch.Tensor) -> tuple[dict, int]:
    if quant_state.dtype != torch.uint8 or quant_state.ndim != 1:
        raise TypeError(f"CompressedTensor quant_state must be a 1-D uint8 tensor; got {quant_state.dtype} with shape {tuple(quant_state.shape)}")
    if quant_state.numel() < _LENGTH_PREFIX_BYTES:
        raise ValueError(f"CompressedTensor quant_state is {quant_state.numel()} bytes, too short to hold a header length")
    header_len = int(quant_state[:_LENGTH_PREFIX_BYTES].view(torch.uint64).item())
    end = _LENGTH_PREFIX_BYTES + header_len
    if end > quant_state.numel():
        raise ValueError(
            f"CompressedTensor header length {header_len} runs past the quant_state's end ({quant_state.numel()} bytes); "
            "the checkpoint is likely truncated"
        )
    try:
        metadata = json.loads(bytes(quant_state[_LENGTH_PREFIX_BYTES:end].detach().cpu().tolist()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("CompressedTensor serialized header is invalid") from error
    if not isinstance(metadata, dict):
        raise TypeError("CompressedTensor serialized header is not a JSON object")

    missing = _REQUIRED_FIELDS - metadata.keys()
    if missing:
        raise ValueError(
            f"CompressedTensor serialized header is missing fields {sorted(missing)}; "
            "it was likely written by a different entropack version"
        )
    return metadata, _aligned(end)


def _views(quant_state: torch.Tensor, metadata: dict, offset: int) -> dict[str, torch.Tensor]:
    buffers = {}
    for name, dtype_name, shape in zip(
        metadata["buffer_names"], metadata["buffer_dtypes"], metadata["buffer_shapes"]
    ):
        dtype = _parse_dtype(dtype_name)
        size = math.prod(shape) * dtype.itemsize
        if offset + size > quant_state.numel():
            raise ValueError(f"CompressedTensor quant_state is {quant_state.numel()} bytes, too short for buffer '{name}'")
        buffers[name] = quant_state[offset : offset + size].view(dtype).view(shape)
        offset = _aligned(offset + size)
    return buffers


class CompressedTensor(torch.Tensor):
    """Compressed tensor storage exposed through a frozen PyTorch Tensor interface.

    Create containers with :func:`entropack.compress` or :meth:`unpack`. The wrapper exposes the uncompressed
    shape, while ``entropack_quant_state`` holds the packed representation and ``buffers`` are views
    into it holding the stored components and the codec metadata. Use :func:`entropack.decompress` to obtain a
    dense tensor before numerical operations.

    ``dtype`` selects the output type of ``decompress``; ``encoded_dtype`` records the codec's input type.
    With the default ``copy=False``, ``to(dtype=...)`` changes only the output type and shares the packed
    storage. Device transfers move the quant_state without changing the components' dtypes. Use ``clone()`` or
    ``to(..., copy=True)`` for an independent storage copy.

    :meth:`state_dict` exports the quant_state for saving, and it is also what pickles this object, so ``torch.save``
    stores the compressed value itself. Loading restores the encoded dtype; temporary
    output-dtype changes are not saved. The wrapper cannot require gradients. Treat it as read-only.

    Args:
        header: codec metadata containing the ``compress_method`` scheme name.
        buffers: named tensors holding stored data and codec metadata on one device.
        shape: shape of the uncompressed tensor.
        dtype: logical dtype returned by ``decompress``.
        encoded_dtype: dtype expected by the codec. Defaults to ``dtype``.
    """

    header: dict[str, Any]
    buffers: dict[str, torch.Tensor]
    entropack_quant_state: torch.Tensor

    @staticmethod
    def __new__(cls, header, buffers, shape, dtype, *, encoded_dtype=None):
        device = next(iter(buffers.values())).device
        return torch.Tensor._make_wrapper_subclass(cls, tuple(shape), dtype=dtype, device=device, requires_grad=False)

    def __init__(self, header, buffers, shape, dtype, *, encoded_dtype=None):
        self.encoded_dtype = dtype if encoded_dtype is None else encoded_dtype
        self._adopt(_pack(_metadata(header, buffers, shape, self.encoded_dtype), buffers))

    def _adopt(self, quant_state: torch.Tensor) -> None:
        metadata, offset = _parse_quant_state(quant_state)
        self.entropack_quant_state = quant_state
        self.header = metadata["header"]
        self.encoded_dtype = _parse_dtype(metadata["dtype"])
        self.buffers = _views(quant_state, metadata, offset)
        self.validate()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device}, scheme={self.compress_method!r})"

    def __tensor_flatten__(self):
        return ["entropack_quant_state"], ()

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, metadata, outer_size, outer_stride):
        return cls.unpack(inner_tensors["entropack_quant_state"])

    def __getstate__(self):
        # torch refuses to serialize several tensors viewing one storage as different dtypes, so the component
        # views stay out and only the quant_state travels.
        return {"entropack_quant_state": self.entropack_quant_state}

    def __setstate__(self, state):
        self._adopt(state["entropack_quant_state"])

    @classmethod
    def unpack(cls, quant_state: torch.Tensor) -> "CompressedTensor":
        """Restore a container from a packed quant_state, sharing its storage."""
        metadata, _ = _parse_quant_state(quant_state)
        result = torch.Tensor._make_wrapper_subclass(
            cls, tuple(metadata["shape"]), dtype=_parse_dtype(metadata["dtype"]),
            device=quant_state.device, requires_grad=False,
        )
        result._adopt(quant_state)
        return result

    def _map_quant_state(self, fn, *, dtype=None) -> "CompressedTensor":
        entropack_quant_state = fn(self.entropack_quant_state)
        result = torch.Tensor._make_wrapper_subclass(
            type(self), tuple(self.shape), dtype=self.dtype if dtype is None else dtype,
            device=entropack_quant_state.device, requires_grad=False,
        )
        result._adopt(entropack_quant_state)
        return result

    def to(self, *args, copy: bool = False, **kwargs) -> "CompressedTensor":
        """Move the packed storage or change the logical dtype, with an optional keyword-only copy."""
        device, dtype, non_blocking, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        result = super().to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy, memory_format=memory_format)
        return result.clone() if copy and result.device == self.device else result

    def __copy__(self) -> "CompressedTensor":
        result = self._map_quant_state(lambda value: value)
        if getattr(self, "_is_param", False):
            result._is_param = True
        return result

    def __deepcopy__(self, memo) -> "CompressedTensor":
        if id(self) in memo:
            return memo[id(self)]
        result = self._map_quant_state(lambda value: copy.deepcopy(value, memo))
        if getattr(self, "_is_param", False):
            result._is_param = True
        memo[id(self)] = result
        return result

    def requires_grad_(self, requires_grad: bool = False):
        """Compressed values are frozen; gradients may flow through their consumers."""
        if requires_grad:
            raise RuntimeError("CompressedTensor cannot require gradients; decompress it to train dense values")
        return torch.Tensor.requires_grad_(self, False)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        source = args[0] if args else None
        aten = torch.ops.aten

        if func in (aten.detach.default, aten.alias.default):
            result = source._map_quant_state(torch.detach)
            return return_and_correct_aliasing(func, args, kwargs, result)

        if func in (aten.clone.default, aten._to_copy.default):
            if kwargs.get("memory_format") not in (None, torch.preserve_format, torch.contiguous_format):
                raise ValueError("CompressedTensor does not support that memory format")
            if func == aten.clone.default:
                return source._map_quant_state(torch.clone)
            if kwargs.get("layout", torch.strided) not in (None, torch.strided):
                raise ValueError("CompressedTensor only supports strided storage")
            options = {name: value for name, value in kwargs.items() if name in ("device", "non_blocking")}
            return source._map_quant_state(lambda value: value.to(**options), dtype=kwargs.get("dtype"))

        result = func.decompose(*args, **kwargs)
        if result is not NotImplemented:
            return result
        raise NotImplementedError(f"CompressedTensor does not implement {func}; decompress it before numerical operations")

    @property
    def compress_method(self) -> str:
        """The scheme name the header carries."""
        return self.header["compress_method"]

    @property
    def scheme(self) -> Scheme:
        """The codec named by the header."""
        return get_scheme(self.compress_method)

    @property
    def lossless(self) -> bool:
        """Whether the encoding scheme is lossless, before any output dtype conversion."""
        return self.scheme.lossless

    @property
    def actual_bpp(self) -> float:
        """Bits per parameter of :attr:`shape`, serialized header included."""
        return self.storage_nbytes() * 8 / math.prod(self.shape)

    def validate(self) -> None:
        """Check the scheme, dtype, and buffer names without inspecting contents."""
        if any(isinstance(value, CompressedTensor) for value in self.buffers.values()):
            raise TypeError("CompressedTensor buffers cannot contain another CompressedTensor")
        scheme = self.scheme
        if not scheme.supports(self.encoded_dtype):
            raise ValueError(f"'{scheme.name}' does not support format {self.encoded_dtype}")
        if set(self.buffers) != set(scheme.buffer_names):
            raise ValueError(
                f"CompressedTensor buffers for {self.compress_method} must be "
                f"{list(scheme.buffer_names)}, got {sorted(self.buffers)}"
            )

    def storage_nbytes(self, include_header: bool = True) -> int:
        """Bytes the quant_state occupies, less the header when ``include_header`` is false."""
        if include_header:
            return self.entropack_quant_state.numel()
        header_len = int(self.entropack_quant_state[:_LENGTH_PREFIX_BYTES].view(torch.uint64).item())
        return self.entropack_quant_state.numel() - _aligned(_LENGTH_PREFIX_BYTES + header_len)

    def state_dict(self, leaf: str | None = None) -> dict[str, torch.Tensor]:
        """The container as a one-tensor dictionary, under the leaf the caller names."""
        self.validate()
        return {(leaf or "entropack_quant_state"): self.entropack_quant_state.detach()}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, torch.Tensor], leaf: str | None = None) -> "CompressedTensor":
        """Restore a container from a tensor-only checkpoint written by :meth:`state_dict`."""
        leaf = leaf or "entropack_quant_state"
        if leaf not in state:
            raise ValueError(f"CompressedTensor state is missing '{leaf}'")
        return cls.unpack(state[leaf])
