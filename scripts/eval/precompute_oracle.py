"""
precompute_oracle.py

Precompute "oracle" visual-token indices for the TARGE ablation. Given a held-out
chat.json (LLaVA-style), this script:

  1. Loads the trained PrismaticVLM (with `route_mode="full"`, `use_qformer=False`)
     so the LLM sees every projected visual token.
  2. Runs a single forward pass per example with `output_attentions=True` and the
     gold response included in the input sequence.
  3. Averages attention from response-token positions to visual-token positions
     across the early layers (default 0-3) and all heads.
  4. Saves the top-k indices per example to a single torch tensor file keyed by id.

Output: `oracle_indices.pt` -> dict[example_id -> LongTensor(k)]

Run:
  python scripts/eval/precompute_oracle.py \
      --model_path /content/drive/MyDrive/targe-prismatic-vlms/runs/targe-smollm2-5k \
      --heldout_json /content/data/download/llava-laion-cc-sbu-558k/chat_heldout.json \
      --image_root /content/data/download/llava-laion-cc-sbu-558k \
      --out_pt /content/drive/MyDrive/targe-prismatic-vlms/runs/targe-smollm2-5k/oracle_indices.pt
"""

import json
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import draccus
import torch
from PIL import Image
from tqdm import tqdm

from prismatic import load
from prismatic.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


@dataclass
class OracleConfig:
    # fmt: off
    model_path: Union[str, Path] = Path("runs/targe-smollm2-5k")
    heldout_json: Path = Path("/content/data/download/llava-laion-cc-sbu-558k/chat_heldout.json")
    image_root: Path = Path("/content/data/download/llava-laion-cc-sbu-558k")
    out_pt: Path = Path("oracle_indices.pt")

    early_layers: Tuple[int, ...] = (0, 1, 2, 3)
    top_k: Optional[int] = None    # default = vlm.projector.inference_k

    hf_token: Union[str, Path] = Path(".hf_token")
    max_examples: Optional[int] = None
    # fmt: on


def _flatten_conv(conv) -> Tuple[str, str]:
    """LLaVA-style conversation -> (human_message, gpt_response). Strips `<image>` token."""
    if isinstance(conv, str):
        conv = json.loads(conv)
    human = next((t["value"] for t in conv if t.get("from") == "human"), "")
    gpt = next((t["value"] for t in conv if t.get("from") == "gpt"), "")
    return human.replace("<image>", "").strip(), gpt.strip()


@torch.inference_mode()
def _extract_oracle_indices(
    vlm,
    image: Image.Image,
    human: str,
    gpt: str,
    early_layers: Tuple[int, ...],
    top_k: int,
) -> torch.LongTensor:
    """Run a single forward pass over (image, human + gpt) and return top-k visual indices."""
    tokenizer = vlm.llm_backbone.tokenizer
    image_transform = vlm.vision_backbone.image_transform
    device = vlm.device

    prompt_builder = vlm.get_prompt_builder()
    prompt_builder.add_turn(role="human", message=human)
    prompt_builder.add_turn(role="gpt", message=gpt)
    full_text = prompt_builder.get_prompt()

    input_ids = tokenizer(full_text, truncation=True, return_tensors="pt").input_ids.to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    labels = input_ids.clone()
    labels[:, 0] = -100  # mark BOS as ignore, consistent with the train collator contract

    # The VLM is fully cast to bf16 upstream, so pixel_values must match — PIL transforms
    # return float32, which would otherwise collide with the bf16 vision-backbone weights.
    model_dtype = next(vlm.vision_backbone.parameters()).dtype
    pixel_values = image_transform(image)
    if isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values[None, ...].to(device=device, dtype=model_dtype)
    elif isinstance(pixel_values, dict):
        pixel_values = {k: v[None, ...].to(device=device, dtype=model_dtype) for k, v in pixel_values.items()}
    else:
        raise ValueError(f"Unexpected pixel_values type: {type(pixel_values)}")

    # NOTE: Do NOT wrap in `torch.autocast` here. The VLM is already fully cast to bf16
    # via `vlm.to(device, dtype=torch.bfloat16)` upstream; layering autocast on top of an
    # eager-attention forward (needed for `output_attentions=True`) causes a dtype clash
    # inside the attention mask broadcast ("expected Float but found BFloat16").
    output = vlm(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels,
        output_attentions=True,
        return_dict=True,
    )

    # output.attentions: tuple of length num_layers, each (B=1, num_heads, seq, seq)
    # Visual tokens occupy positions [1, 1 + V), where V = projector output length.
    # On the `full` route, V == N (the raw projected ViT token count).
    # Response tokens are at positions [1 + V, seq).
    attns = output.attentions
    if attns is None:
        raise RuntimeError("LLM returned no attentions. Check `output_attentions` plumbing.")

    # Infer V by replaying the projector forward (cheap; identical to what `vlm.forward` did).
    with torch.no_grad():
        if isinstance(pixel_values, dict):
            patch_feats = vlm.vision_backbone(pixel_values)
        else:
            patch_feats = vlm.vision_backbone(pixel_values)
        projected = vlm.projector(patch_feats)
    V = projected.shape[1]

    seq = attns[0].shape[-1]
    visual_slice = slice(1, 1 + V)
    response_slice = slice(1 + V, seq)
    if response_slice.stop <= response_slice.start:
        # No response tokens (degenerate); fall back to all post-visual tokens.
        response_slice = slice(1 + V, seq)
    if response_slice.stop - response_slice.start == 0:
        # Truly nothing past the image — fall back to attention from the last text token.
        response_slice = slice(seq - 1, seq)

    # Average over selected early layers + all heads + all response positions.
    layer_attns = [attns[i] for i in early_layers if i < len(attns)]
    stacked = torch.stack(layer_attns, dim=0)  # (L, 1, H, S, S)
    avg = stacked.float().mean(dim=(0, 2))  # (1, S, S)
    attn_to_visual = avg[0, response_slice, visual_slice]  # (R, V)
    scores = attn_to_visual.mean(dim=0)  # (V,)

    actual_k = min(top_k, V)
    _, top_idx = torch.topk(scores, k=actual_k, largest=True)
    return top_idx.detach().cpu().to(torch.long)


@draccus.wrap()
def main(cfg: OracleConfig) -> None:
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

    # The default LLM backbone is built with SDPA attention, whose kernel does not return
    # attention weights, so `output_attentions=True` silently yields `attentions=None`.
    # HF Llama-family models dispatch attention via `self.config._attn_implementation` at
    # each forward, so flipping it on the loaded config propagates to every layer.
    inner_llm = vlm.llm_backbone.llm
    inner_llm.config._attn_implementation = "eager"
    if hasattr(inner_llm.config, "attn_implementation"):
        inner_llm.config.attn_implementation = "eager"
    overwatch.info("Forced LLM attention implementation -> `eager` (required for output_attentions).")

    # Force "full" route + no Q-Former so the LLM sees every projected ViT token.
    assert hasattr(vlm.projector, "route_mode"), "Projector lacks ablation routing; rebuild from latest code."
    vlm.projector.route_mode = "full"
    vlm.projector.use_qformer = False
    top_k = cfg.top_k or int(vlm.projector.inference_k)
    overwatch.info(f"Oracle top_k = {top_k} (early_layers = {cfg.early_layers})")

    with open(cfg.heldout_json) as f:
        heldout = json.load(f)
    if cfg.max_examples:
        heldout = heldout[: cfg.max_examples]

    oracle: dict = {}
    skipped = 0
    first_traceback_printed = False
    for ex in tqdm(heldout, desc="oracle"):
        ex_id = ex.get("id") or ex.get("image")
        img_rel = ex.get("image")
        if not img_rel:
            skipped += 1
            continue
        img_path = cfg.image_root / img_rel
        if not img_path.is_file():
            skipped += 1
            continue
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            overwatch.info(f"[skip] {ex_id}: {type(e).__name__}")
            skipped += 1
            continue

        human, gpt = _flatten_conv(ex.get("conversations", []))
        if not human:
            skipped += 1
            continue

        try:
            idx = _extract_oracle_indices(vlm, image, human, gpt, cfg.early_layers, top_k)
        except Exception as e:
            if not first_traceback_printed:
                print(
                    f"\n[oracle] FIRST FAILURE on ex={ex_id} — full traceback follows "
                    "(subsequent failures will be one-liners):\n",
                    flush=True,
                )
                traceback.print_exc()
                first_traceback_printed = True
            overwatch.info(f"[skip] {ex_id}: {type(e).__name__}: {e}")
            skipped += 1
            continue

        oracle[str(ex_id)] = idx

    cfg.out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(oracle, cfg.out_pt)
    overwatch.info(f"Saved {len(oracle):,} oracle entries to `{cfg.out_pt}` (skipped {skipped})")


if __name__ == "__main__":
    main()
