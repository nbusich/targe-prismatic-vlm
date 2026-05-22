"""
materialize.py

Factory class defining functions for instantiating various Training Strategies, supporting different VLMs, backbones,
and strategy configurations.
"""

from typing import Callable, Optional

import torch

from prismatic.models.vlms import PrismaticVLM
from prismatic.training.strategies import DDPStrategy, FSDPStrategy, TrainingStrategy

TRAIN_STRATEGIES = {
    "fsdp-shard-grad-op": {"cls": FSDPStrategy, "kwargs": {"sharding_strategy": "shard-grad-op"}},
    "fsdp-full-shard": {"cls": FSDPStrategy, "kwargs": {"sharding_strategy": "full-shard"}},
    "ddp": {"cls": DDPStrategy, "kwargs": {}},
}


def get_train_strategy(
    train_strategy: str,
    vlm: PrismaticVLM,
    device_id: int,
    epochs: int,
    max_steps: Optional[int],
    global_batch_size: int,
    per_device_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    lr_scheduler_type: str,
    warmup_ratio: float,
    enable_gradient_checkpointing: bool = True,
    enable_mixed_precision_training: bool = True,
    reduce_in_full_precision: bool = False,
    mixed_precision_dtype: torch.dtype = torch.bfloat16,
    worker_init_fn: Optional[Callable[[int], None]] = None,
    selector_tau_start: float = 1.0,
    selector_tau_end: float = 0.5,
    selector_tau_hold_ratio: float = 0.2,
    selector_lambda_target: float = 0.05,
    selector_lambda_warmup_ratio: float = 0.1,
    selector_target_keep_ratio: float = 0.5,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    aux_attn_enabled: bool = True,
    aux_attn_weight: float = 1.0,
    aux_attn_layers: tuple = (0,),
) -> TrainingStrategy:
    if train_strategy in TRAIN_STRATEGIES:
        strategy_cfg = TRAIN_STRATEGIES[train_strategy]
        strategy = strategy_cfg["cls"](
            vlm=vlm,
            device_id=device_id,
            epochs=epochs,
            max_steps=max_steps,
            global_batch_size=global_batch_size,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            enable_mixed_precision_training=enable_mixed_precision_training,
            reduce_in_full_precision=reduce_in_full_precision,
            mixed_precision_dtype=mixed_precision_dtype,
            worker_init_fn=worker_init_fn,
            selector_tau_start=selector_tau_start,
            selector_tau_end=selector_tau_end,
            selector_tau_hold_ratio=selector_tau_hold_ratio,
            selector_lambda_target=selector_lambda_target,
            selector_lambda_warmup_ratio=selector_lambda_warmup_ratio,
            selector_target_keep_ratio=selector_target_keep_ratio,
            num_workers=num_workers,
            pin_memory=pin_memory,
            aux_attn_enabled=aux_attn_enabled,
            aux_attn_weight=aux_attn_weight,
            aux_attn_layers=aux_attn_layers,
            **strategy_cfg["kwargs"],
        )
        return strategy
    else:
        raise ValueError(f"Train Strategy `{train_strategy}` is not supported!")
