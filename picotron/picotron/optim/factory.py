from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from torch import nn
from torch.optim import AdamW

from .muon import DEFAULT_NS_COEFFICIENTS, DEFAULT_NS_STEPS, Muon


@dataclass
class OptimizerConfig:
    name: str
    learning_rate: float
    weight_decay: float = 0.0
    betas: Optional[Tuple[float, float]] = None
    eps: Optional[float] = None
    momentum: float = 0.95
    nesterov: bool = True
    ns_coefficients: tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS
    ns_steps: int = DEFAULT_NS_STEPS
    adjust_lr_fn: Optional[str] = None
    muon_weight_decay: Optional[float] = None
    muon_momentum: Optional[float] = None
    muon_adjust_lr_fn: Optional[str] = None
    muon_eps: Optional[float] = None


class HybridOptimizer:
    """Applies Muon to matrix weights and AdamW to the remaining parameters."""

    def __init__(
        self,
        muon_params: Iterable[nn.Parameter],
        adam_params: Iterable[nn.Parameter],
        *,
        muon_kwargs: dict,
        adam_kwargs: dict,
    ) -> None:
        self._muon_params = list(muon_params)
        self._adam_params = list(adam_params)

        self._muon = Muon(self._muon_params, **muon_kwargs) if self._muon_params else None
        self._adam = AdamW(self._adam_params, **adam_kwargs) if self._adam_params else None

    @property
    def param_groups(self):
        groups: list[dict] = []
        if self._muon is not None:
            groups.extend(self._muon.param_groups)
        if self._adam is not None:
            groups.extend(self._adam.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self._muon is not None:
            self._muon.zero_grad(set_to_none=set_to_none)
        if self._adam is not None:
            self._adam.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        local_closure = closure
        if self._muon is not None:
            loss = self._muon.step(local_closure)
            local_closure = None
        if self._adam is not None:
            adam_loss = self._adam.step(local_closure)
            if loss is None:
                loss = adam_loss
        return loss

    def state_dict(self) -> dict:
        return {
            "muon": self._muon.state_dict() if self._muon is not None else None,
            "adam": self._adam.state_dict() if self._adam is not None else None,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        muon_state = state_dict.get("muon") if state_dict is not None else None
        adam_state = state_dict.get("adam") if state_dict is not None else None
        if self._muon is not None and muon_state is not None:
            self._muon.load_state_dict(muon_state)
        if self._adam is not None and adam_state is not None:
            self._adam.load_state_dict(adam_state)


def create_optimizer(module: nn.Module, config: OptimizerConfig, *, adam_extra_kwargs: Optional[dict] = None):
    muon_params: list[nn.Parameter] = []
    adam_params: list[nn.Parameter] = []

    for param_name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2 and param_name.endswith("weight"):
            muon_params.append(param)
        else:
            adam_params.append(param)

    muon_kwargs = {
        "lr": config.learning_rate,
        "weight_decay": config.muon_weight_decay if config.muon_weight_decay is not None else config.weight_decay,
        "momentum": config.muon_momentum if config.muon_momentum is not None else config.momentum,
        "nesterov": config.nesterov,
        "ns_coefficients": config.ns_coefficients,
        "ns_steps": config.ns_steps,
        "adjust_lr_fn": config.muon_adjust_lr_fn if config.muon_adjust_lr_fn is not None else config.adjust_lr_fn,
    }
    if config.muon_eps is not None:
        muon_kwargs["eps"] = config.muon_eps

    adam_kwargs = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    if config.betas is not None:
        adam_kwargs["betas"] = config.betas
    if config.eps is not None:
        adam_kwargs["eps"] = config.eps
    if adam_extra_kwargs:
        adam_kwargs.update(adam_extra_kwargs)

    if config.name.lower() == "muon":
        if not muon_params:
            raise ValueError("Muon optimizer selected but no 2D weight parameters available")
        return HybridOptimizer(muon_params, adam_params, muon_kwargs=muon_kwargs, adam_kwargs=adam_kwargs)

    if config.name.lower() == "adamw":
        return AdamW(module.parameters(), **adam_kwargs)

    if config.name.lower() == "hybrid":
        return HybridOptimizer(muon_params, adam_params, muon_kwargs=muon_kwargs, adam_kwargs=adam_kwargs)

    raise ValueError(f"Unsupported optimizer type: {config.name}")


