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
    _flatten_conv,
    _generate_text,
    _get_patch_features,
    _iou,
    _prep_inputs,
    _set_route,
    _time_connector,
)

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
                for ex in tqdm(heldout, desc=cell_id, leave=False):
                    ex_id = str(ex.get("id") or ex.get("image"))
                    try:
                        img_rel = ex.get("image")
                        if not img_rel:
                            continue
                        img_path = cfg.image_root / img_rel
                        if not img_path.is_file():
                            continue
                        image = Image.open(img_path).convert("RGB")
                        human, _ = _flatten_conv(ex.get("conversations", []))
                        if not human:
                            continue

                        input_ids, pixel_values = _prep_inputs(vlm, image, human)

                        # Oracle indices for this cell: truncated to K.
                        oracle_idx = oracle_table.get(ex_id)
                        oracle_idx_dev = None
                        if opt["route_mode"] == "oracle":
                            if oracle_idx is None:
                                continue
                            k_eff = min(K, oracle_idx.shape[0])
                            oracle_idx_dev = oracle_idx[:k_eff].to(device).unsqueeze(0)

                        # Group-A reference latent for cosine sim.
                        _set_route(projector, "full", False, None)
                        ref_latent = _capture_connector_output(vlm, pixel_values).float().flatten(1)

                        # Cell's routing.
                        _apply_kq(projector, K, Q, snapshot_qs, snapshot_M)
                        _set_route(projector, opt["route_mode"], use_qf, oracle_idx_dev)

                        latent = _capture_connector_output(vlm, pixel_values).float().flatten(1)
                        min_dim = min(ref_latent.shape[1], latent.shape[1])
                        cos = F.cosine_similarity(
                            ref_latent[:, :min_dim], latent[:, :min_dim], dim=1
                        ).item()
                        cell_result["cos_vs_A"].append(cos)

                        if opt["route_mode"] == "selector" and oracle_idx is not None:
                            sel = projector.latest_selected_indices
                            if sel is not None:
                                k_eff = min(K, oracle_idx.shape[0])
                                cell_result["iou_vs_oracle"].append(
                                    _iou(sel[0].cpu(), oracle_idx[:k_eff])
                                )

                        _set_route(projector, opt["route_mode"], use_qf, oracle_idx_dev)
                        gen = _generate_text(vlm, input_ids, pixel_values, cfg.max_new_tokens)
                        cell_result["generations"].append({"id": ex_id, "prompt": human, "gen": gen})
                        cell_result["n_generations"] += 1

                    except Exception as e:
                        _maybe_traceback("per-example", cell_id, ex_id)
                        cell_result["n_errors"] += 1
                        cell_result["generations"].append(
                            {"id": ex_id, "prompt": "", "gen": f"<error: {type(e).__name__}: {e}>"}
                        )

                # ---- POPE accuracy for this cell ----
                if pope_items:
                    correct = n = yes_pred = 0
                    for item in tqdm(pope_items, desc=f"POPE/{cell_id}", leave=False):
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
