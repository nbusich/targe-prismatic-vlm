"""
run_kq_sweep.py

Sweep the visual-token budget (K, Q) over a trained TARGE checkpoint to characterize
how cos_vs_A / POPE accuracy / latency / FLOPS scale with the number of tokens that
reach the LLM.

  - K = `vlm.projector.inference_k` — top-k tokens kept by the selector / oracle path.
  - Q = number of Q-Former queries — slice of the learned `compress_queries` tensor.
        Q=0 disables the Q-Former entirely for that cell (use_qformer toggled off).

Defaults sweep:
    groups = (C, D)                 # C = oracle+QF, D = selector+QF
    k_grid = (32, 128, 256)         # coarse, mid, near-full (clamped to ViT patch count)
    q_grid = (0, 4, 8, 16, 32)      # 0 → no Q-Former; 32 = trained Q

For each (group, K, Q) cell, runs the same per-example pattern as `run_ablation.py`:
captures Group-A reference latent, applies the cell's routing, records cos / IoU /
generation. Then measures connector latency + FLOPS and (optionally) POPE accuracy.

Run:
  python scripts/eval/run_kq_sweep.py \
      --model_path /content/drive/MyDrive/targe-prismatic-vlms/runs/targe-smollm2-75 \
      --heldout_json /content/pope/chat_heldout.json \
      --image_root /content/pope \
      --oracle_pt /content/oracle_indices.pt \
      --pope_json /content/pope/pope_popular.json \
      --pope_image_root /content/pope \
      --out_json /content/drive/MyDrive/targe-prismatic-vlms/runs/targe-smollm2-75/kq_sweep_results.json
"""

import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import draccus
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from prismatic import load
from prismatic.overwatch import initialize_overwatch

# `scripts/` isn't a Python package, so import the sibling module via path injection.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ablation import (  # noqa: E402
    _capture_connector_output,
    _connector_flops,
    _connector_from_features,
    _emit_status,
    _flatten_conv,
    _generate_text,
    _get_patch_features,
    _iou,
    _prep_inputs,
    _set_route,
    _time_connector,
    _vision_features,
)
import time  # for status timing

overwatch = initialize_overwatch(__name__)


GROUP_PRESETS: Dict[str, dict] = {
    # `use_qformer` here is the *base* preference — actual value per cell is
    # `base_use_qformer and (Q > 0)` (Q=0 always disables the Q-Former).
    "C": {"route_mode": "oracle",   "use_qformer": True},
    "D": {"route_mode": "selector", "use_qformer": True},
    # Easy extensions (left in but off by default):
    "B": {"route_mode": "random_topk", "use_qformer": True},
    "E": {"route_mode": "selector",    "use_qformer": False},
}


@dataclass
class KQSweepConfig:
    # fmt: off
    model_path: Union[str, Path] = Path("runs/targe-smollm2-75")
    heldout_json: Path = Path("/content/pope/chat_heldout.json")
    image_root: Path = Path("/content/pope")
    oracle_pt: Optional[Path] = None
    pope_json: Optional[Path] = None
    pope_image_root: Optional[Path] = None
    out_json: Path = Path("kq_sweep_results.json")

    k_grid: Tuple[int, ...] = (32, 128, 256)
    q_grid: Tuple[int, ...] = (0, 4, 8, 16, 32)
    groups: Tuple[str, ...] = ("C", "D")

    max_examples: Optional[int] = None         # cap heldout examples per cell (for smoke tests)
    pope_max_examples: Optional[int] = 300     # cap POPE per cell — full ~3000 makes the sweep too slow
    max_new_tokens: int = 16                   # tight cap — anti-repetition + early stop emit EOS usually sooner
    timing_warmup: int = 5
    timing_iters: int = 20
    # Logging cadence — print a flushed status line every N items in any inner loop.
    log_every: int = 50

    hf_token: Union[str, Path] = Path(".hf_token")
    # fmt: on


def _apply_kq(projector, K: int, Q: int, snapshot_qs: torch.Tensor, snapshot_M: int) -> None:
    """Mutate the projector in-place to the (K, Q) cell. Reversible via `_restore_kq`."""
    projector.inference_k = int(K)
    if Q <= 0:
        # Keep `compress_queries` at its full snapshot (it's just not exercised).
        projector.compress_queries.data = snapshot_qs.clone()
        projector.num_compressed_tokens = int(snapshot_M)
    else:
        projector.compress_queries.data = snapshot_qs[:, :Q, :].contiguous().clone()
        projector.num_compressed_tokens = int(Q)


def _restore_kq(projector, snapshot_qs: torch.Tensor, snapshot_M: int, snapshot_K: int) -> None:
    projector.compress_queries.data = snapshot_qs.clone()
    projector.num_compressed_tokens = int(snapshot_M)
    projector.inference_k = int(snapshot_K)


def _cell_id(group: str, K: int, Q: int) -> str:
    return f"{group}_k{K}_q{Q}"


def _dump_partial(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(out_path)


@draccus.wrap()
def main(cfg: KQSweepConfig) -> None:
    torch.manual_seed(7)
    overwatch.info(f"Loading VLM from `{cfg.model_path}`")
    hf_token = (
        cfg.hf_token.read_text().strip()
        if isinstance(cfg.hf_token, Path) and cfg.hf_token.exists()
        else os.environ.get(str(cfg.hf_token), "") or None
    )
    vlm = load(cfg.model_path, hf_token=hf_token)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    vlm.to(device, dtype=torch.bfloat16)
    vlm.eval()
    assert hasattr(vlm.projector, "route_mode"), "Projector lacks ablation routing; rebuild from latest code."

    projector = vlm.projector
    snapshot_qs = projector.compress_queries.data.detach().clone()
    snapshot_M = int(projector.num_compressed_tokens)
    snapshot_K = int(projector.inference_k)
    overwatch.info(
        f"[snapshot] inference_k={snapshot_K}  num_compressed_tokens={snapshot_M}  "
        f"compress_queries shape={tuple(snapshot_qs.shape)}"
    )

    # Validate Q grid against the trained Q.
    bad_q = [q for q in cfg.q_grid if q > snapshot_M]
    if bad_q:
        raise ValueError(
            f"q_grid contains values > trained Q ({snapshot_M}): {bad_q}. "
            "Q>Q_train would require re-initializing query rows — retrain instead."
        )

    # Load heldout examples.
    with open(cfg.heldout_json) as f:
        heldout = json.load(f)
    if cfg.max_examples:
        heldout = heldout[: cfg.max_examples]
    overwatch.info(f"Heldout examples: {len(heldout):,}")

    # Load oracle indices if any group needs them.
    oracle_table: Dict[str, torch.LongTensor] = {}
    needs_oracle = any(GROUP_PRESETS[g]["route_mode"] == "oracle" for g in cfg.groups)
    if needs_oracle and cfg.oracle_pt and cfg.oracle_pt.exists():
        oracle_table = torch.load(cfg.oracle_pt, map_location="cpu")
        overwatch.info(f"Loaded {len(oracle_table)} oracle entries from `{cfg.oracle_pt}`")
    elif needs_oracle:
        overwatch.info("[warn] oracle group requested but no oracle_pt provided — C cells will skip")

    pope_items: List[dict] = []
    if cfg.pope_json and cfg.pope_json.exists():
        with open(cfg.pope_json) as f:
            pope_items = json.load(f)
        if cfg.pope_max_examples:
            pope_items = pope_items[: cfg.pope_max_examples]
        overwatch.info(f"POPE items per cell: {len(pope_items):,}")

    # ─── Precompute vision features + Group A reference latents (once) ─────────
    # These are invariant across (K, Q, group) cells, so caching once eliminates
    # ~30 redundant vision-backbone forwards per heldout example. CPU storage
    # keeps GPU memory free for cell-specific projector forwards.
    overwatch.info("Precomputing vision features + Group A reference latents for heldout...")
    _pre_t0 = time.time()
    model_dtype = next(vlm.vision_backbone.parameters()).dtype
    image_transform = vlm.vision_backbone.image_transform

    heldout_cache: List[Optional[dict]] = []
    for _pre_idx, ex in enumerate(heldout):
        try:
            img_rel = ex.get("image")
            ex_id = str(ex.get("id") or img_rel)
            if not img_rel:
                heldout_cache.append(None); continue
            img_path = cfg.image_root / img_rel
            if not img_path.is_file():
                heldout_cache.append(None); continue
            human, _ = _flatten_conv(ex.get("conversations", []))
            if not human:
                heldout_cache.append(None); continue

            img = Image.open(img_path).convert("RGB")
            pv = image_transform(img)
            if isinstance(pv, dict):
                pv_gpu = {k: v[None, ...].to(device=device, dtype=model_dtype) for k, v in pv.items()}
                pv_cpu = {k: v.detach().cpu() for k, v in pv_gpu.items()}
            else:
                pv_gpu = pv[None, ...].to(device=device, dtype=model_dtype)
                pv_cpu = pv_gpu.detach().cpu()

            pf_gpu = _vision_features(vlm, pv_gpu)
            pf_cpu = pf_gpu.detach().cpu() if isinstance(pf_gpu, torch.Tensor) else {
                k: v.detach().cpu() for k, v in pf_gpu.items()
            }

            # Group A reference latent — projector with full + no Q-Former, independent of K/Q.
            _set_route(projector, "full", False, None)
            ref_latent_cpu = _connector_from_features(vlm, pf_gpu).float().flatten(1).detach().cpu()

            heldout_cache.append({
                "ex_id": ex_id,
                "prompt": human,
                "patch_features_cpu": pf_cpu,
                "pixel_values_cpu": pv_cpu,
                "ref_latent_cpu": ref_latent_cpu,
            })
        except Exception as _e:
            heldout_cache.append(None)

        if (_pre_idx + 1) % cfg.log_every == 0 or (_pre_idx + 1) == len(heldout):
            _emit_status("precompute/heldout", _pre_idx + 1, len(heldout), _pre_t0)

    n_cached = sum(1 for h in heldout_cache if h is not None)
    overwatch.info(
        f"[precompute] {n_cached}/{len(heldout)} heldout cached in "
        f"{time.time() - _pre_t0:.1f}s"
    )

    # Result container.
    cells: Dict[str, dict] = {}
    payload = {
        "config": {
            "model_path": str(cfg.model_path),
            "heldout_json": str(cfg.heldout_json),
            "oracle_pt": str(cfg.oracle_pt) if cfg.oracle_pt else None,
            "pope_json": str(cfg.pope_json) if cfg.pope_json else None,
            "k_grid": list(cfg.k_grid),
            "q_grid": list(cfg.q_grid),
            "groups": list(cfg.groups),
            "max_examples": cfg.max_examples,
            "pope_max_examples": cfg.pope_max_examples,
            "trained_Q": snapshot_M,
        },
        "cells": cells,
    }

    first_traceback_printed = {"flag": False}

    def _maybe_traceback(stage: str, cell_id: str, ex_id: str = "") -> None:
        if not first_traceback_printed["flag"]:
            print(
                f"\n[kq-sweep] FIRST FAILURE during `{stage}` on cell={cell_id} ex={ex_id} — "
                f"full traceback follows (subsequent failures will be one-liners):\n",
                flush=True,
            )
            traceback.print_exc()
            first_traceback_printed["flag"] = True

    # =====================================================================
    # Main sweep
    # =====================================================================
    for group in cfg.groups:
        if group not in GROUP_PRESETS:
            overwatch.info(f"[skip] unknown group `{group}` (known: {list(GROUP_PRESETS)})")
            continue
        opt = GROUP_PRESETS[group]
        for K in cfg.k_grid:
            for Q in cfg.q_grid:
                cell_id = _cell_id(group, K, Q)
                use_qf = bool(opt["use_qformer"]) and (Q > 0)
                _apply_kq(projector, K, Q, snapshot_qs, snapshot_M)
                overwatch.info(
                    f"\n=== cell {cell_id}  route={opt['route_mode']}  use_qformer={use_qf}  "
                    f"K={K}  Q={Q} ==="
                )

                cell_result = {
                    "group": group,
                    "k": int(K),
                    "q": int(Q),
                    "route_mode": opt["route_mode"],
                    "use_qformer": use_qf,
                    "n_generations": 0,
                    "n_errors": 0,
                    "cos_vs_A": [],
                    "iou_vs_oracle": [],
                    "generations": [],
                    "pope": None,
                    "hardware": None,
                }

                # ---- per-example: cos vs A + (optional) IoU vs oracle + gen ----
                # Uses precomputed heldout_cache so vision_backbone runs 0 times in the cos pass.
                _cell_t0 = time.time()
                tokenizer = vlm.llm_backbone.tokenizer
                for _ex_idx, cache_entry in enumerate(tqdm(heldout_cache, desc=cell_id, leave=False)):
                    if cache_entry is None:
                        continue
                    ex_id = cache_entry["ex_id"]
                    try:
                        # Move cached tensors to GPU (cheap — they live on CPU).
                        pf_cpu = cache_entry["patch_features_cpu"]
                        if isinstance(pf_cpu, torch.Tensor):
                            patch_features_gpu = pf_cpu.to(device, non_blocking=True)
                        else:
                            patch_features_gpu = {k: v.to(device, non_blocking=True) for k, v in pf_cpu.items()}
                        ref_latent = cache_entry["ref_latent_cpu"].to(device, non_blocking=True)

                        # Oracle indices for this cell: truncated to K.
                        oracle_idx = oracle_table.get(ex_id) if oracle_table else None
                        oracle_idx_dev = None
                        if opt["route_mode"] == "oracle":
                            if oracle_idx is None:
                                continue
                            k_eff = min(K, oracle_idx.shape[0])
                            oracle_idx_dev = oracle_idx[:k_eff].to(device).unsqueeze(0)

                        # Cell's routing — uses cached patch features (no vision_backbone forward).
                        _apply_kq(projector, K, Q, snapshot_qs, snapshot_M)
                        _set_route(projector, opt["route_mode"], use_qf, oracle_idx_dev)
                        latent = _connector_from_features(vlm, patch_features_gpu).float().flatten(1)

                        min_dim = min(ref_latent.shape[1], latent.shape[1])
                        cos = F.cosine_similarity(ref_latent[:, :min_dim], latent[:, :min_dim], dim=1).item()
                        cell_result["cos_vs_A"].append(cos)

                        if opt["route_mode"] == "selector" and oracle_idx is not None:
                            sel = projector.latest_selected_indices
                            if sel is not None:
                                k_eff = min(K, oracle_idx.shape[0])
                                cell_result["iou_vs_oracle"].append(
                                    _iou(sel[0].cpu(), oracle_idx[:k_eff])
                                )

                        # Generation still re-runs vision_backbone inside vlm.generate (acceptable cost
                        # at 16 tokens; can't be skipped without refactoring PrismaticVLM.forward).
                        pv_cpu = cache_entry["pixel_values_cpu"]
                        if isinstance(pv_cpu, torch.Tensor):
                            pixel_values_gpu = pv_cpu.to(device, non_blocking=True)
                        else:
                            pixel_values_gpu = {k: v.to(device, non_blocking=True) for k, v in pv_cpu.items()}
                        prompt_builder = vlm.get_prompt_builder()
                        prompt_builder.add_turn(role="human", message=cache_entry["prompt"])
                        prompt_text = prompt_builder.get_prompt()
                        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(device)

                        _set_route(projector, opt["route_mode"], use_qf, oracle_idx_dev)
                        gen = _generate_text(vlm, input_ids, pixel_values_gpu, cfg.max_new_tokens)
                        cell_result["generations"].append({"id": ex_id, "prompt": cache_entry["prompt"], "gen": gen})
                        cell_result["n_generations"] += 1

                    except Exception as e:
                        _maybe_traceback("per-example", cell_id, ex_id)
                        cell_result["n_errors"] += 1
                        cell_result["generations"].append(
                            {"id": ex_id, "prompt": "", "gen": f"<error: {type(e).__name__}: {e}>"}
                        )

                    if (_ex_idx + 1) % cfg.log_every == 0 or (_ex_idx + 1) == len(heldout_cache):
                        cos_list = cell_result["cos_vs_A"]
                        cos_mean = sum(cos_list) / max(1, len(cos_list)) if cos_list else None
                        _emit_status(
                            cell_id, _ex_idx + 1, len(heldout_cache), _cell_t0,
                            cos=(f"{cos_mean:.3f}" if cos_mean is not None else None),
                            ngens=cell_result["n_generations"],
                        )

                # ---- POPE accuracy for this cell ----
                if pope_items:
                    correct = n = yes_pred = 0
                    _pope_t0 = time.time()
                    for _pope_idx, item in enumerate(tqdm(pope_items, desc=f"POPE/{cell_id}", leave=False)):
                        try:
                            img = Path(cfg.pope_image_root or cfg.image_root) / item["image"]
                            if not img.is_file():
                                continue
                            image = Image.open(img).convert("RGB")
                            input_ids, pixel_values = _prep_inputs(vlm, image, item["question"])

                            oracle_idx_dev = None
                            if opt["route_mode"] == "oracle":
                                oi = oracle_table.get(str(item.get("id") or item["image"]))
                                if oi is None:
                                    continue
                                k_eff = min(K, oi.shape[0])
                                oracle_idx_dev = oi[:k_eff].to(device).unsqueeze(0)

                            _apply_kq(projector, K, Q, snapshot_qs, snapshot_M)
                            _set_route(projector, opt["route_mode"], use_qf, oracle_idx_dev)

                            autocast_dtype = vlm.llm_backbone.half_precision_dtype
                            with torch.autocast(
                                "cuda", dtype=autocast_dtype, enabled=vlm.enable_mixed_precision_training
                            ):
                                out = super(type(vlm), vlm).generate(
                                    input_ids=input_ids,
                                    pixel_values=pixel_values,
                                    do_sample=False,
                                    max_new_tokens=4,
                                    min_length=1,
                                    output_scores=True,
                                    return_dict_in_generate=True,
                                    early_stopping=True,
                                )
                            first_logits = out.scores[0][0]
                            yes_idx = vlm.string2idx["Yes"]
                            no_idx = vlm.string2idx["No"]
                            pred_yes = first_logits[yes_idx] > first_logits[no_idx]
                            gold_yes = str(item["label"]).strip().lower().startswith("y")
                            correct += int(pred_yes.item() == gold_yes)
                            yes_pred += int(pred_yes.item())
                            n += 1
                        except Exception:
                            _maybe_traceback("pope", cell_id)
                            continue

                        if (_pope_idx + 1) % cfg.log_every == 0 or (_pope_idx + 1) == len(pope_items):
                            _emit_status(
                                f"POPE/{cell_id}", _pope_idx + 1, len(pope_items), _pope_t0,
                                acc=f"{(correct / max(1, n)):.3f}",
                                yes_rate=f"{(yes_pred / max(1, n)):.3f}",
                                n_valid=n,
                            )
                    cell_result["pope"] = {
                        "accuracy": correct / n if n else None,
                        "yes_rate": yes_pred / n if n else None,
                        "n": n,
                    }

                # ---- Hardware: connector FLOPS + latency for this cell ----
                try:
                    _apply_kq(projector, K, Q, snapshot_qs, snapshot_M)
                    # Get a dummy patch tensor from the first loadable heldout image.
                    dummy = None
                    for ex in heldout:
                        try:
                            p = cfg.image_root / ex["image"]
                            if not p.is_file():
                                continue
                            image = Image.open(p).convert("RGB")
                            _, pixel_values = _prep_inputs(vlm, image, "")
                            dummy = _get_patch_features(vlm, pixel_values).detach()
                            break
                        except Exception:
                            continue
                    if dummy is not None:
                        B = dummy.shape[0]
                        dummy_oracle = (
                            torch.arange(K, device=device).unsqueeze(0).expand(B, -1)
                            if opt["route_mode"] == "oracle"
                            else None
                        )
                        _set_route(projector, opt["route_mode"], use_qf, dummy_oracle)
                        flops = _connector_flops(projector, dummy)
                        latency_ms = _time_connector(projector, dummy, cfg.timing_warmup, cfg.timing_iters)
                        cell_result["hardware"] = {"flops": flops, "latency_ms_median": latency_ms}
                except Exception as e:
                    _maybe_traceback("hardware", cell_id)
                    cell_result["hardware"] = {"error": f"{type(e).__name__}: {e}"}

                # ---- summarize + flush ----
                cos_list = cell_result["cos_vs_A"]
                iou_list = cell_result["iou_vs_oracle"]
                cell_result["cos_vs_A_mean"] = (sum(cos_list) / len(cos_list)) if cos_list else None
                cell_result["iou_vs_oracle_mean"] = (sum(iou_list) / len(iou_list)) if iou_list else None

                cells[cell_id] = cell_result
                _dump_partial(cfg.out_json, payload)
                overwatch.info(
                    f"[done] {cell_id}  "
                    f"cos_vs_A={cell_result['cos_vs_A_mean']}  "
                    f"iou={cell_result['iou_vs_oracle_mean']}  "
                    f"pope_acc={(cell_result['pope'] or {}).get('accuracy')}  "
                    f"latency={((cell_result['hardware'] or {}).get('latency_ms_median'))}"
                )

    _restore_kq(projector, snapshot_qs, snapshot_M, snapshot_K)
    _dump_partial(cfg.out_json, payload)
    overwatch.info(f"Wrote K×Q sweep results to `{cfg.out_json}`")


if __name__ == "__main__":
    main()
