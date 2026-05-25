"""
run_basic_eval.py

Architecture-agnostic single-pass eval over a held-out caption set. Works for any
PrismaticVLM checkpoint (MLP, selector, Q-Former — no `route_mode` required).

For each held-out example:
  - Build the (human, gpt) prompt as `FinetuneDataset` would, masking the human
    turn with IGNORE_INDEX and supervising the gpt turn.
  - One teacher-forced forward pass through the VLM (vision + projector + LLM)
    to capture caption-CE loss.
  - One greedy generation pass for qualitative inspection.

Reports mean loss + ppl and saves all (prompt, gold, gen, loss) tuples to JSON.
This is the right tool when the projector lacks ablation hooks; for selector
checkpoints with `route_mode`, use `run_ablation.py` instead.

Run:
  python scripts/eval/run_basic_eval.py \
      --model_path /path/to/run_dir \
      --heldout_json /path/to/chat_heldout.json \
      --image_root  /path/to/images/ \
      --out_json    /path/to/basic_eval.json
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import draccus
import torch
from PIL import Image
from tqdm import tqdm

from prismatic import load
from prismatic.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)

IGNORE_INDEX = -100


@dataclass
class BasicEvalConfig:
    # fmt: off
    model_path: Union[str, Path] = Path("runs/<run-id>")
    heldout_json: Path = Path("/content/data/download/llava-laion-cc-sbu-558k/chat_heldout.json")
    image_root:  Path = Path("/content/data/download/llava-laion-cc-sbu-558k")
    out_json:    Path = Path("basic_eval.json")

    max_examples:   Optional[int] = None
    max_new_tokens: int = 32
    log_every:      int = 50

    hf_token: Union[str, Path] = Path(".hf_token")
    # fmt: on


def _flatten_conv(conv) -> Tuple[str, str]:
    if isinstance(conv, str):
        conv = json.loads(conv)
    human = next((t["value"] for t in conv if t.get("from") == "human"), "")
    gpt = next((t["value"] for t in conv if t.get("from") == "gpt"), "")
    # Strip the `<image>` placeholder — the image is spliced in by `forward()`.
    return human.replace("<image>", "").strip(), gpt.strip()


def _build_tf_inputs(vlm, human: str, gpt: str, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror `FinetuneDataset.__getitem__`: tokenize turn-by-turn, mask the human turn."""
    tokenizer = vlm.llm_backbone.tokenizer
    prompt_builder = vlm.get_prompt_builder()

    input_ids: List[int] = []
    labels: List[int] = []
    for turn_idx, (role, value) in enumerate([("human", human), ("gpt", gpt)]):
        msg = prompt_builder.add_turn(role, value)
        turn_ids = tokenizer(msg, add_special_tokens=(turn_idx == 0)).input_ids
        turn_labels = (
            [IGNORE_INDEX] * len(turn_ids) if turn_idx % 2 == 0 else list(turn_ids)
        )
        input_ids.extend(turn_ids)
        labels.extend(turn_labels)

    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    labels = torch.tensor(labels, device=device).unsqueeze(0)
    # Align with the dataset / VLM contract: position-0 is always ignored.
    labels[0, 0] = IGNORE_INDEX
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    return input_ids, attention_mask, labels


def _pixel_values(vlm, image: Image.Image, device):
    pixel_values = vlm.vision_backbone.image_transform(image)
    model_dtype = next(vlm.vision_backbone.parameters()).dtype
    if isinstance(pixel_values, torch.Tensor):
        return pixel_values[None, ...].to(device=device, dtype=model_dtype)
    if isinstance(pixel_values, dict):
        return {k: v[None, ...].to(device=device, dtype=model_dtype) for k, v in pixel_values.items()}
    raise ValueError(f"Unsupported pixel_values type: {type(pixel_values)}")


@torch.inference_mode()
def _teacher_forced_loss(vlm, pixel_values, input_ids, attention_mask, labels) -> float:
    autocast_dtype = vlm.llm_backbone.half_precision_dtype
    with torch.autocast("cuda", dtype=autocast_dtype, enabled=vlm.enable_mixed_precision_training):
        out = vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            multimodal_indices=None,
        )
    return float(out.loss.detach().item())


@torch.inference_mode()
def _generate(vlm, pixel_values, human: str, max_new_tokens: int) -> str:
    tokenizer = vlm.llm_backbone.tokenizer
    prompt_builder = vlm.get_prompt_builder()
    prompt_builder.add_turn("human", human)
    prompt_text = prompt_builder.get_prompt()
    input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(vlm.device)
    autocast_dtype = vlm.llm_backbone.half_precision_dtype
    with torch.autocast("cuda", dtype=autocast_dtype, enabled=vlm.enable_mixed_precision_training):
        # Bypass PrismaticVLM.generate (which assumes batch=1 pixel handling) and call
        # GenerationMixin.generate directly through the parent class.
        gen_ids = super(type(vlm), vlm).generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            min_length=1,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            early_stopping=True,
        )
    return tokenizer.decode(gen_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()


def _emit(idx: int, total: int, t0: float, running_loss: float, n: int) -> None:
    elapsed = time.time() - t0
    rate = idx / max(elapsed, 1e-6)
    eta = (total - idx) / max(rate, 1e-6)
    print(
        f"[basic-eval] {idx}/{total}  rate={rate:.1f}/s  elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
        f"loss={running_loss / max(1, n):.4f}  n={n}",
        flush=True,
    )


def _atomic_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


@draccus.wrap()
def main(cfg: BasicEvalConfig) -> None:
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

    with open(cfg.heldout_json) as f:
        heldout = json.load(f)
    if cfg.max_examples:
        heldout = heldout[: cfg.max_examples]
    overwatch.info(f"Loaded {len(heldout):,} held-out examples")

    records: List[dict] = []
    running_loss = 0.0
    n_loss = 0
    n_errors = 0
    t0 = time.time()

    for idx, ex in enumerate(tqdm(heldout, desc="basic-eval")):
        ex_id = str(ex.get("id") or ex.get("image"))
        try:
            img_rel = ex.get("image")
            if not img_rel:
                continue
            img_path = cfg.image_root / img_rel
            if not img_path.is_file():
                continue
            image = Image.open(img_path).convert("RGB")
            human, gold = _flatten_conv(ex.get("conversations", []))
            if not human or not gold:
                continue

            pixel_values = _pixel_values(vlm, image, device)
            input_ids, attn_mask, labels = _build_tf_inputs(vlm, human, gold, device)

            loss = _teacher_forced_loss(vlm, pixel_values, input_ids, attn_mask, labels)
            gen = _generate(vlm, pixel_values, human, cfg.max_new_tokens)

            running_loss += loss
            n_loss += 1
            records.append({"id": ex_id, "prompt": human, "gold": gold, "gen": gen, "loss": loss})
        except Exception as e:
            n_errors += 1
            records.append({"id": ex_id, "error": f"{type(e).__name__}: {e}"})

        if (idx + 1) % cfg.log_every == 0 or (idx + 1) == len(heldout):
            mean_loss = running_loss / max(1, n_loss)
            _atomic_dump(
                cfg.out_json,
                {
                    "summary": {
                        "n": n_loss,
                        "n_errors": n_errors,
                        "mean_loss": mean_loss,
                        "perplexity": float(torch.tensor(mean_loss).exp().item()) if n_loss else None,
                    },
                    "config": {
                        "model_path": str(cfg.model_path),
                        "heldout_json": str(cfg.heldout_json),
                        "max_examples": cfg.max_examples,
                    },
                    "records": records,
                },
            )
            _emit(idx + 1, len(heldout), t0, running_loss, n_loss)

    mean_loss = running_loss / max(1, n_loss)
    overwatch.info(
        f"[basic-eval] done. n={n_loss}  n_errors={n_errors}  "
        f"mean_loss={mean_loss:.4f}  ppl={float(torch.tensor(mean_loss).exp().item()):.2f}"
    )


if __name__ == "__main__":
    main()
