"""
run_ablation.py

Unified six-group ablation sweep over a single trained TARGE checkpoint.

Groups:
  A: route_mode=full,        use_qformer=False   # upper bound (all tokens, no compression)
  B: route_mode=random_topk, use_qformer=False   # lower bound
  C: route_mode=oracle,      use_qformer=False   # oracle indices
  D: route_mode=selector,    use_qformer=True    # trained behavior
  E: route_mode=selector,    use_qformer=False   # selector only
  F: route_mode=full,        use_qformer=True    # Q-Former only

Per group, this records:
  - generated text on the held-out prompts
  - connector-output tensor (for cosine similarity vs. Group A)
  - selector vs. oracle IoU (groups D, E)
  - POPE-style yes/no accuracy if `pope_json` is provided
  - connector latency (ms, CUDA events, 5 warmup + 20 timed)
  - connector FLOPS (if `fvcore` is importable)

Output: `ablation_results.json` next to the checkpoint.

Run:
  python scripts/eval/run_ablation.py \
      --model_path /content/drive/MyDrive/targe-prismatic-vlms/runs/targe-smollm2-5k \
      --heldout_json /content/data/download/llava-laion-cc-sbu-558k/chat_heldout.json \
      --image_root /content/data/download/llava-laion-cc-sbu-558k \
      --oracle_pt /content/drive/MyDrive/targe-prismatic-vlms/runs/targe-smollm2-5k/oracle_indices.pt \
      --out_json /content/drive/MyDrive/targe-prismatic-vlms/runs/targe-smollm2-5k/ablation_results.json
"""

import json
import os
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import draccus
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from prismatic import load
from prismatic.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)

GROUPS: List[Tuple[str, dict]] = [
    ("A", {"route_mode": "full",        "use_qformer": False}),
    ("B", {"route_mode": "random_topk", "use_qformer": False}),
    ("C", {"route_mode": "oracle",      "use_qformer": False}),
    ("D", {"route_mode": "selector",    "use_qformer": True }),
    ("E", {"route_mode": "selector",    "use_qformer": False}),
    # Group F (route_mode="full", use_qformer=True) was removed: in `_forward_inference`
    # the `dropped` tensor is empty when route_mode="full", so the Q-Former operates on
    # zero tokens and returns zeros, making F's connector output identical to A's padded
    # with zeros (cos_vs_A ≈ 0.9996). It is not a meaningful ablation as currently coded.
]


@dataclass
class AblationConfig:
    # fmt: off
    model_path: Union[str, Path] = Path("runs/targe-smollm2-5k")
    heldout_json: Path = Path("/content/data/download/llava-laion-cc-sbu-558k/chat_heldout.json")
    image_root: Path = Path("/content/data/download/llava-laion-cc-sbu-558k")
    oracle_pt: Optional[Path] = None
    pope_json: Optional[Path] = None  # if provided, expects [{image, question, label("yes"/"no")}, ...]
    pope_image_root: Optional[Path] = None
    out_json: Path = Path("ablation_results.json")

    max_examples: Optional[int] = None
    max_new_tokens: int = 16      # tight cap — repetition penalty + early stop usually emit EOS sooner
    timing_warmup: int = 5
    timing_iters: int = 20
    # Batching + logging knobs (set high on Blackwell/H100 to actually use the GPU).
    eval_batch_size: int = 64     # batch size for POPE forward (cos pass is per-example)
    log_every: int = 50           # print a flushed status line every N examples

    hf_token: Union[str, Path] = Path(".hf_token")
    # fmt: on


def _flatten_conv(conv) -> Tuple[str, str]:
    # Some chat.json variants store `conversations` as a JSON-encoded string instead of a list.
    if isinstance(conv, str):
        conv = json.loads(conv)
    human = next((t["value"] for t in conv if t.get("from") == "human"), "")
    gpt = next((t["value"] for t in conv if t.get("from") == "gpt"), "")
    return human.replace("<image>", "").strip(), gpt.strip()


def _emit_status(tag: str, idx: int, total: int, t0: float, **stats) -> None:
    """Flushed one-line status update so Colab actually shows progress during long evals."""
    elapsed = time.time() - t0
    rate = idx / max(elapsed, 1e-6)
    eta = (total - idx) / max(rate, 1e-6)
    extras = "  ".join(f"{k}={v}" for k, v in stats.items() if v is not None)
    print(
        f"[{tag}] {idx}/{total}  rate={rate:.1f}/s  elapsed={elapsed:.0f}s  ETA={eta:.0f}s  {extras}",
        flush=True,
    )


@torch.inference_mode()
def _vision_features(vlm, pixel_values):
    """Run the vision backbone once and return patch features (no projector, no LLM)."""
    return vlm.vision_backbone(pixel_values)


@torch.inference_mode()
def _connector_from_features(vlm, patch_features) -> torch.Tensor:
    """Run only the projector on already-computed vision patch features."""
    return vlm.projector(patch_features)


def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / max(union, 1)


def _connector_flops(projector, dummy_feat) -> Optional[float]:
    """Return FLOPS for a single forward of the projector. Skips silently if fvcore is unavailable."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception:
        return None
    projector.eval()
    flops = FlopCountAnalysis(projector, dummy_feat)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    return float(flops.total())


def _time_connector(projector, dummy_feat, warmup: int, iters: int) -> float:
    """Median connector-forward latency (ms) over `iters` runs."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        _ = projector(dummy_feat)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        _ = projector(dummy_feat)
        ends[i].record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return ms[len(ms) // 2]


def _prep_inputs(vlm, image: Image.Image, human: str):
    """Mirror `vlm.generate` input prep but return tensors for both generation and forward-capture."""
    tokenizer = vlm.llm_backbone.tokenizer
    image_transform = vlm.vision_backbone.image_transform
    device = vlm.device

    prompt_builder = vlm.get_prompt_builder()
    prompt_builder.add_turn(role="human", message=human)
    prompt_text = prompt_builder.get_prompt()
    input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(device)

    # The VLM is fully cast to bf16 upstream; `_capture_connector_output` runs without
    # autocast, so pixel_values must already match the model dtype.
    model_dtype = next(vlm.vision_backbone.parameters()).dtype
    pixel_values = image_transform(image)
    if isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values[None, ...].to(device=device, dtype=model_dtype)
    elif isinstance(pixel_values, dict):
        pixel_values = {k: v[None, ...].to(device=device, dtype=model_dtype) for k, v in pixel_values.items()}
    else:
        raise ValueError(f"Unsupported pixel_values type: {type(pixel_values)}")

    return input_ids, pixel_values


@torch.inference_mode()
def _get_patch_features(vlm, pixel_values):
    if isinstance(pixel_values, dict):
        return vlm.vision_backbone(pixel_values)
    return vlm.vision_backbone(pixel_values)


@torch.inference_mode()
def _generate_text(vlm, input_ids, pixel_values, max_new_tokens) -> str:
    tokenizer = vlm.llm_backbone.tokenizer
    autocast_dtype = vlm.llm_backbone.half_precision_dtype
    with torch.autocast("cuda", dtype=autocast_dtype, enabled=vlm.enable_mixed_precision_training):
        # Anti-repetition: align-only checkpoints love to loop the same trigram until
        # max_new_tokens is exhausted, which dominates eval runtime. The repetition_penalty
        # + no_repeat_ngram_size combo plus EOS-based early stopping keeps generations short.
        generated_ids = super(type(vlm), vlm).generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            min_length=1,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            early_stopping=True,
        )
    return tokenizer.decode(generated_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()


def _set_route(projector, route_mode: str, use_qformer: bool, oracle_idx: Optional[torch.LongTensor]):
    projector.route_mode = route_mode
    projector.use_qformer = use_qformer
    projector._oracle_indices = oracle_idx


def _capture_connector_output(vlm, pixel_values) -> torch.Tensor:
    """Run vision_backbone + projector once with current routing flags; return the connector output."""
    patch = _get_patch_features(vlm, pixel_values)
    return vlm.projector(patch)  # (B, S, D)


def _aggregate(results, pope_metrics, hw_metrics):
    summary = {}
    for name, _ in GROUPS:
        r = results[name]
        cos = r["cos_vs_A"]
        iou = r["iou_vs_oracle"]
        summary[name] = {
            "n_generations": len(r["generations"]),
            "n_errors": r.get("n_errors", 0),
            "cos_vs_A_mean": (sum(cos) / len(cos)) if cos else None,
            "iou_vs_oracle_mean": (sum(iou) / len(iou)) if iou else None,
            "pope": pope_metrics.get(name),
            "hardware": hw_metrics.get(name),
        }
    return summary


def _dump_partial(out_path: Path, results, pope_metrics, hw_metrics, cfg):
    """Atomic write so a kill mid-flush never corrupts the file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": _aggregate(results, pope_metrics, hw_metrics),
        "generations": {name: results[name]["generations"] for name, _ in GROUPS},
        "config": {
            "model_path": str(cfg.model_path),
            "heldout_json": str(cfg.heldout_json),
            "oracle_pt": str(cfg.oracle_pt) if cfg.oracle_pt else None,
            "pope_json": str(cfg.pope_json) if cfg.pope_json else None,
            "max_examples": cfg.max_examples,
        },
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(out_path)


@draccus.wrap()
def main(cfg: AblationConfig) -> None:
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

    with open(cfg.heldout_json) as f:
        heldout = json.load(f)
    if cfg.max_examples:
        heldout = heldout[: cfg.max_examples]

    oracle_table: Dict[str, torch.LongTensor] = {}
    if cfg.oracle_pt and cfg.oracle_pt.exists():
        oracle_table = torch.load(cfg.oracle_pt, map_location="cpu")
        overwatch.info(f"Loaded {len(oracle_table)} oracle index entries from `{cfg.oracle_pt}`")

    pope_items: List[dict] = []
    if cfg.pope_json and cfg.pope_json.exists():
        with open(cfg.pope_json) as f:
            pope_items = json.load(f)
        if cfg.max_examples:
            pope_items = pope_items[: cfg.max_examples]
        overwatch.info(f"Loaded {len(pope_items)} POPE items from `{cfg.pope_json}`")

    # ---- Per-example sweep: generations + connector latent capture ----
    # Each group's results are independent: a failure in one group never blocks the others,
    # and the JSON is flushed after every example so a crash leaves usable partial output.
    results = {
        name: {"generations": [], "iou_vs_oracle": [], "cos_vs_A": [], "n_errors": 0}
        for name, _ in GROUPS
    }
    pope_metrics: Dict[str, dict] = {}
    hw_metrics: Dict[str, dict] = {}

    # Print the first failure (across all swallowers) in full, then go quiet on tracebacks.
    first_traceback_printed = {"flag": False}
    def _maybe_print_first_traceback(stage: str, ex_id: str) -> None:
        if not first_traceback_printed["flag"]:
            print(
                f"\n[ablation] FIRST FAILURE during `{stage}` on ex={ex_id} — full traceback follows "
                "(subsequent failures will be one-liners):\n",
                flush=True,
            )
            traceback.print_exc()
            first_traceback_printed["flag"] = True

    _loop_t0 = time.time()
    overwatch.info(f"Starting per-example loop: {len(heldout):,} examples, log_every={cfg.log_every}")
    for _idx, ex in enumerate(tqdm(heldout, desc="ablation")):
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
            oracle_idx = oracle_table.get(ex_id)
            oracle_idx_dev = oracle_idx.to(device).unsqueeze(0) if oracle_idx is not None else None

            # Cache the vision-backbone output once per example — it's invariant across the
            # 5 groups, so we skip 4 redundant ViT forwards per example.
            patch_features_cached = _vision_features(vlm, pixel_values)

            # Group A reference latent for cosine similarity (uses cached features).
            try:
                _set_route(vlm.projector, "full", False, None)
                ref_latent = _connector_from_features(vlm, patch_features_cached).float().flatten(1)
            except Exception as e:
                _maybe_print_first_traceback("ref-latent", ex_id)
                overwatch.info(f"[ex={ex_id}] ref-latent failed, skipping example: {type(e).__name__}: {e}")
                continue
        except Exception as e:
            _maybe_print_first_traceback("setup", ex_id)
            overwatch.info(f"[ex={ex_id}] setup failed: {type(e).__name__}: {e}")
            continue

        for name, opt in GROUPS:
            if name == "C" and oracle_idx_dev is None:
                continue
            try:
                _set_route(vlm.projector, opt["route_mode"], opt["use_qformer"], oracle_idx_dev)

                # Connector output for cosine sim (reuses cached vision features).
                latent = _connector_from_features(vlm, patch_features_cached).float().flatten(1)
                min_dim = min(ref_latent.shape[1], latent.shape[1])
                cos = F.cosine_similarity(ref_latent[:, :min_dim], latent[:, :min_dim], dim=1).item()
                results[name]["cos_vs_A"].append(cos)

                # IoU vs oracle (selector groups).
                if name in {"D", "E"} and oracle_idx is not None:
                    sel = vlm.projector.latest_selected_indices
                    if sel is not None:
                        results[name]["iou_vs_oracle"].append(_iou(sel[0].cpu(), oracle_idx))

                # Generation.
                _set_route(vlm.projector, opt["route_mode"], opt["use_qformer"], oracle_idx_dev)
                gen = _generate_text(vlm, input_ids, pixel_values, cfg.max_new_tokens)
                results[name]["generations"].append({"id": ex_id, "prompt": human, "gen": gen})
            except Exception as e:
                _maybe_print_first_traceback(f"group-{name}", ex_id)
                results[name]["n_errors"] += 1
                results[name]["generations"].append(
                    {"id": ex_id, "prompt": human, "gen": f"<error: {type(e).__name__}: {e}>"}
                )

        # Flush after every example so an OOM / disconnect leaves usable partial output.
        try:
            _dump_partial(cfg.out_json, results, pope_metrics, hw_metrics, cfg)
        except Exception as e:
            overwatch.info(f"[ex={ex_id}] partial dump failed: {type(e).__name__}: {e}")

        # Periodic flushed status line — Colab buffers tqdm output so this is what you actually see.
        if (_idx + 1) % cfg.log_every == 0 or (_idx + 1) == len(heldout):
            cos_means = {
                name: (sum(results[name]["cos_vs_A"]) / max(1, len(results[name]["cos_vs_A"])))
                for name, _ in GROUPS if results[name]["cos_vs_A"]
            }
            cos_summary = " ".join(f"{n}={v:.3f}" for n, v in cos_means.items())
            _emit_status("ablation", _idx + 1, len(heldout), _loop_t0, cos=cos_summary)

    # ---- POPE accuracy (per-group try/except so a broken group doesn't kill the rest) ----
    if pope_items:
        for name, opt in GROUPS:
            try:
                correct = 0
                n = 0
                yes_pred = 0
                _pope_t0 = time.time()
                overwatch.info(f"POPE/{name}: starting on {len(pope_items):,} items")
                for _pope_idx, item in enumerate(tqdm(pope_items, desc=f"POPE/{name}")):
                    try:
                        img = Path(cfg.pope_image_root or cfg.image_root) / item["image"]
                        if not img.is_file():
                            continue
                        image = Image.open(img).convert("RGB")
                        input_ids, pixel_values = _prep_inputs(vlm, image, item["question"])

                        oracle_idx_dev = None
                        if name == "C":
                            oi = oracle_table.get(str(item.get("id") or item["image"]))
                            if oi is None:
                                continue
                            oracle_idx_dev = oi.to(device).unsqueeze(0)
                        _set_route(vlm.projector, opt["route_mode"], opt["use_qformer"], oracle_idx_dev)

                        autocast_dtype = vlm.llm_backbone.half_precision_dtype
                        with torch.autocast("cuda", dtype=autocast_dtype, enabled=vlm.enable_mixed_precision_training):
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
                        continue

                    # Periodic status during POPE — useful when running 3000 items per group.
                    if (_pope_idx + 1) % cfg.log_every == 0 or (_pope_idx + 1) == len(pope_items):
                        running_acc = correct / max(1, n)
                        running_yes = yes_pred / max(1, n)
                        _emit_status(
                            f"POPE/{name}", _pope_idx + 1, len(pope_items), _pope_t0,
                            acc=f"{running_acc:.3f}", yes_rate=f"{running_yes:.3f}", n_valid=n,
                        )
                pope_metrics[name] = {
                    "accuracy": correct / n if n else None,
                    "yes_rate": yes_pred / n if n else None,
                    "n": n,
                }
                _dump_partial(cfg.out_json, results, pope_metrics, hw_metrics, cfg)
                overwatch.info(
                    f"[POPE/{name}] done. acc={pope_metrics[name]['accuracy']}  "
                    f"yes_rate={pope_metrics[name]['yes_rate']}  n={n}"
                )
            except Exception as e:
                pope_metrics[name] = {"error": f"{type(e).__name__}: {e}"}

    # ---- Hardware metrics (per-group try/except) ----
    if heldout:
        # Use the first successfully loadable image as the patch-feature template.
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
            for name, opt in GROUPS:
                try:
                    k = int(vlm.projector.inference_k)
                    dummy_oracle = (
                        torch.arange(k, device=device).unsqueeze(0).expand(B, -1)
                        if opt["route_mode"] == "oracle"
                        else None
                    )
                    _set_route(vlm.projector, opt["route_mode"], opt["use_qformer"], dummy_oracle)
                    flops = _connector_flops(vlm.projector, dummy)
                    latency_ms = _time_connector(vlm.projector, dummy, cfg.timing_warmup, cfg.timing_iters)
                    hw_metrics[name] = {"flops": flops, "latency_ms_median": latency_ms}
                except Exception as e:
                    hw_metrics[name] = {"error": f"{type(e).__name__}: {e}"}
                _dump_partial(cfg.out_json, results, pope_metrics, hw_metrics, cfg)

    _dump_partial(cfg.out_json, results, pope_metrics, hw_metrics, cfg)
    overwatch.info(f"Wrote ablation results to `{cfg.out_json}`")
    overwatch.info(json.dumps(_aggregate(results, pope_metrics, hw_metrics), indent=2))


if __name__ == "__main__":
    main()
