"""Implementation of the Muon optimizer.

This module adapts the public reference implementation from Keller Jordan's Muon
project (https://github.com/KellerJordan/Muon) to work within the Picotron
training stack without relying on PyTorch private optimizer utilities.
"""

from __future__ import annotations

import math
from collections.abc import MutableMapping
from typing import Optional, Sequence

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT

__all__ = ["Muon", "DEFAULT_NS_COEFFICIENTS", "DEFAULT_NS_STEPS", "EPS"]

# Constants from Keller Jordan's Muon post: https://kellerjordan.github.io/posts/muon/
EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5
DEFAULT_NS_COEFFICIENTS = (DEFAULT_A, DEFAULT_B, DEFAULT_C)


def _to_scalar(x):
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            raise ValueError("Tensor scalar inputs must be 1-element")
        return float(x.item())
    return float(x)


def _zeropower_via_newtonschulz(
    grad: Tensor,
    ns_coefficients: Sequence[float],
    ns_steps: int,
    eps: float,
) -> Tensor:
    if ns_steps >= 100:
        raise ValueError("Number of steps must be less than 100 for computational efficiency")
    if grad.ndim != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must contain exactly 3 values")

    a, b, c = ns_coefficients

    working_dtype = torch.bfloat16 if grad.dtype != torch.bfloat16 else grad.dtype
    ortho_grad = grad.to(dtype=working_dtype)
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.transpose(0, 1)

    denom = ortho_grad.norm()
    denom = denom.clamp(min=eps)
    ortho_grad = ortho_grad.div(denom)

    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.transpose(0, 1)
        gram_update = torch.addmm(gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c)
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)

    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.transpose(0, 1)

    return ortho_grad.to(dtype=grad.dtype)


def _adjust_lr(lr: float, adjust_lr_fn: Optional[str], param_shape: torch.Size) -> float:
    if len(param_shape) < 2:
        return lr

    a_dim, b_dim = param_shape[:2]

    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1, a_dim / b_dim))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(a_dim, b_dim))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


class Muon(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: Sequence[float] = DEFAULT_NS_COEFFICIENTS,
        eps: float = EPS,
        ns_steps: int = DEFAULT_NS_STEPS,
        adjust_lr_fn: Optional[str] = None,
    ) -> None:
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        lr_scalar = _to_scalar(lr)
        if lr_scalar < 0.0:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if momentum < 0.0:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if ns_steps < 0:
            raise ValueError("ns_steps must be non-negative")
        if adjust_lr_fn is not None and adjust_lr_fn not in {"original", "match_rms_adamw"}:
            raise ValueError(f"Adjust learning rate function {adjust_lr_fn} is not supported")

        defaults = {
            "lr": lr_scalar,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": tuple(ns_coefficients),
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        "Muon only supports 2D parameters whereas we found a parameter with size:"
                        f" {tuple(p.shape)}"
                    )

    def _init_group(
        self,
        group: MutableMapping,
        params_with_grad: list[Tensor],
        grads: list[Tensor],
        muon_momentum_bufs: list[Tensor],
    ) -> bool:
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            if torch.is_complex(p):
                has_complex = True
                continue
            if p.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")

            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
            muon_momentum_bufs.append(state["momentum_buffer"])
        return has_complex

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            muon_momentum_bufs: list[Tensor] = []

            has_complex = self._init_group(group, params_with_grad, grads, muon_momentum_bufs)

            _single_tensor_muon(
                params_with_grad,
                grads,
                muon_momentum_bufs,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=group["nesterov"],
                ns_coefficients=group["ns_coefficients"],
                ns_steps=group["ns_steps"],
                eps=group["eps"],
                adjust_lr_fn=group["adjust_lr_fn"],
                has_complex=has_complex,
            )
        return loss


def _single_tensor_muon(
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: Sequence[float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: Optional[str],
    has_complex: bool,
) -> None:
    lr_scalar = _to_scalar(lr)
    if has_complex:
        raise ValueError("Complex parameters are not supported")

    for idx, param in enumerate(params):
        grad = grads[idx]
        if grad.ndim != 2:
            raise ValueError("Param gradient must be a 2D matrix")

        buf = muon_momentum_bufs[idx]
        buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(buf, momentum) if nesterov else buf

        update = _zeropower_via_newtonschulz(update, ns_coefficients, ns_steps, eps)

        adjusted_lr = _adjust_lr(lr_scalar, adjust_lr_fn, param.shape)

        param.mul_(1 - lr_scalar * weight_decay)
        param.add_(update, alpha=-adjusted_lr)


