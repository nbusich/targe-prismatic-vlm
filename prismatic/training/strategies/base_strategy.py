"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForLanguageModeling

# NOT BEST PRACTICE: using model dataclass to store training parameters

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
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
        selector_tau_start: float = 1.0,
        selector_tau_end: float = 0.5,
        selector_tau_hold_ratio: float = 0.2,
        selector_lambda_target: float = 0.05,
        selector_lambda_warmup_ratio: float = 0.1,
        selector_target_keep_ratio: float = 0.5,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        num_workers: Optional[int] = None,
        pin_memory: bool = True,
        **_: str,
    ) -> None:
        self.vlm, self.device_id = vlm, device_id

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn
        # Auto-scale num_workers to min(8, cpu_count) — Colab usually has 2-12 vCPUs.
        # Cap at 8 so we don't trigger oversubscription on big boxes.
        import os as _os
        self.num_workers = int(num_workers) if num_workers is not None else min(8, _os.cpu_count() or 2)
        self.pin_memory = pin_memory

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # First-backward gradient audit (one-shot; verifies the selector/Q-Former graph isn't severed)
        self._grad_audit_done = False

        # Selector schedule parameters
        self.selector_tau_start = selector_tau_start
        self.selector_tau_end = selector_tau_end
        self.selector_tau_hold_ratio = selector_tau_hold_ratio
        self.selector_lambda_target = selector_lambda_target
        self.selector_lambda_warmup_ratio = selector_lambda_warmup_ratio
        self.selector_target_keep_ratio = selector_target_keep_ratio

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        overwatch.info(
            f"DataLoader: num_workers={self.num_workers}  pin_memory={self.pin_memory}  "
            f"persistent_workers={self.num_workers > 0}"
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=self.num_workers,
            worker_init_fn=self.worker_init_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # Unwrap through DDP/FSDP wrappers so attribute writes hit the underlying selector
                    # module — FSDP/DDP forward attribute reads via __getattr__ but DO NOT intercept
                    # __setattr__, so `wrapped.tau = X` would set on the wrapper, not the inner module.
                    vlm_inner = getattr(self.vlm, "module", self.vlm)
                    projector = vlm_inner.projector
                    projector_inner = getattr(projector, "module", projector)
                    if hasattr(projector_inner, "tau") and getattr(projector_inner, "selector", None) is not None:
                        total = self.max_steps if self.max_steps is not None else len(dataloader) * self.epochs
                        # Hold tau at `tau_start` for the first `hold_ratio` of training, then linearly
                        # anneal to `tau_end` over the remainder — gives the router time to settle before
                        # logits get sharpened.
                        hold_steps = int(self.selector_tau_hold_ratio * total)
                        if metrics.global_step < hold_steps:
                            projector_inner.tau = self.selector_tau_start
                        else:
                            anneal_progress = (metrics.global_step - hold_steps) / max(1, total - hold_steps)
                            projector_inner.tau = max(
                                self.selector_tau_end,
                                self.selector_tau_start
                                - (self.selector_tau_start - self.selector_tau_end) * anneal_progress,
                            )

                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss

                    # --- NaN/Inf guard: diagnose blow-ups before they corrupt optimizer state ---
                    if not torch.isfinite(loss):
                        diag = {
                            "step": metrics.global_step,
                            "tau": getattr(projector_inner, "tau", None),
                            "ce_loss": float(loss),
                        }
                        for name, p in projector_inner.named_parameters():
                            if not torch.isfinite(p).all():
                                diag[f"param_{name}_nonfinite"] = True
                        overwatch.error(f"[NaN GUARD] non-finite loss detected: {diag}")
                        # Skip backward/step so we don't poison the optimizer; log and continue to next batch.
                        self.optimizer.zero_grad(set_to_none=True)
                        continue

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # One-shot gradient audit on the first backward. Surfaces top-k / detach
                    # bugs that silently sever the selector graph: if `selector.router.weight`
                    # gets a None/zero grad, Gumbel straight-through plumbing is broken.
                    if not self._grad_audit_done and overwatch.is_rank_zero():
                        self._grad_audit_done = True
                        projector = (
                            self.vlm.module.projector if hasattr(self.vlm, "module") else self.vlm.projector
                        )
                        print("=== first-backward gradient audit (projector) ===", flush=True)
                        any_zero = False
                        n_inspected = 0
                        for _name, _p in projector.named_parameters():
                            if not _p.requires_grad:
                                continue
                            n_inspected += 1
                            if _p.grad is None:
                                print(f"  [WARN] {_name}: grad is None", flush=True)
                                any_zero = True
                                continue
                            try:
                                _n = _p.grad.detach().float().norm().item()
                            except Exception as _e:
                                print(f"  [WARN] {_name}: could not compute norm ({type(_e).__name__}: {_e})", flush=True)
                                any_zero = True
                                continue
                            _tag = "  [ZERO]" if _n == 0.0 else ""
                            print(f"  {_name:60s}  ||grad|| = {_n:.4e}{_tag}", flush=True)
                            if _n == 0.0:
                                any_zero = True
                        if n_inspected == 0:
                            print(
                                "[WARN] projector has zero trainable params — `freeze_backbones` "
                                "may have frozen everything by mistake.",
                                flush=True,
                            )
                        elif any_zero:
                            print(
                                "[WARN] one or more projector params received zero/None grad. "
                                "Top-k / detach may be severing the graph.",
                                flush=True,
                            )

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                            return

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()
