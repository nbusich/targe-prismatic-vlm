"""
prepare_pope.py

Download a POPE subset (random | popular | adversarial) via the HuggingFace
`lmms-lab/POPE` dataset, materialize its COCO val2014 images to local disk, and
write two JSON files that the existing ablation eval consumes:

  out_dir/
    images/<id>.jpg            # one file per POPE row, written once
    pope_<subset>.json         # [{id, image, question, label}]  (--pope_json input)
    chat_heldout.json          # LLaVA chat schema, n_heldout rows (--heldout_json input)

POPE uses COCO val2014, which is disjoint from the LLaVA-CC-SBU pretrain shard
the model trains on — so this gives a guaranteed-clean held-out set without
retraining.

Run:
  python scripts/eval/prepare_pope.py \
      --out_dir /content/data/pope \
      --subset popular \
      --n_heldout 500
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
from PIL import Image
from tqdm import tqdm

from prismatic.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


@dataclass
class PreparePopeConfig:
    # fmt: off
    out_dir: Path = Path("/content/data/pope")
    subset: str = "popular"               # "random" | "popular" | "adversarial"
    n_heldout: int = 500
    seed: int = 7
    hf_token: Union[str, Path] = Path(".hf_token")
    max_examples: Optional[int] = None    # for smoke tests
    # fmt: on


def _resolve_hf_token(token_or_path: Union[str, Path]) -> Optional[str]:
    if isinstance(token_or_path, Path) and token_or_path.exists():
        return token_or_path.read_text().strip() or None
    return os.environ.get(str(token_or_path), "").strip() or None


def _pick_category_field(sample: dict) -> str:
    """POPE on HF has been distributed under various column names — pick the one present."""
    for key in ("category", "subset", "split", "type"):
        if key in sample:
            return key
    raise KeyError(
        f"Couldn't find a category field on a POPE row. Available keys: {list(sample.keys())}"
    )


def _pick_label_field(sample: dict) -> str:
    for key in ("label", "answer", "gt_answer"):
        if key in sample:
            return key
    raise KeyError(f"Couldn't find a label field. Available keys: {list(sample.keys())}")


def _pick_image_field(sample: dict) -> str:
    for key in ("image", "image_source", "img"):
        if key in sample:
            return key
    raise KeyError(f"Couldn't find an image field. Available keys: {list(sample.keys())}")


def _pick_id_field(sample: dict, idx: int) -> str:
    for key in ("question_id", "id", "qid"):
        if key in sample:
            return str(sample[key])
    return f"row_{idx}"


def _normalize_label(raw) -> str:
    s = str(raw).strip().lower()
    if s.startswith("y"):
        return "yes"
    if s.startswith("n"):
        return "no"
    return s


@draccus.wrap()
def main(cfg: PreparePopeConfig) -> None:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "`datasets` is required. Install with: pip install datasets"
        ) from e

    hf_token = _resolve_hf_token(cfg.hf_token)
    overwatch.info(f"Loading `lmms-lab/POPE` (subset filter: `{cfg.subset}`)")

    # POPE on HF is a single split; try the common names and fall through.
    ds = None
    last_err = None
    for split_name in ("test", "train", "validation"):
        try:
            ds = load_dataset("lmms-lab/POPE", split=split_name, token=hf_token)
            overwatch.info(f"Loaded split=`{split_name}` ({len(ds):,} rows)")
            break
        except Exception as e:
            last_err = e
            continue
    if ds is None:
        raise RuntimeError(f"Failed to load `lmms-lab/POPE` on any standard split: {last_err}")

    if len(ds) == 0:
        raise RuntimeError("POPE dataset loaded but is empty.")

    sample = ds[0]
    cat_field   = _pick_category_field(sample)
    label_field = _pick_label_field(sample)
    image_field = _pick_image_field(sample)

    # Filter by subset.
    filtered_indices = [i for i, row in enumerate(ds) if str(row[cat_field]).lower() == cfg.subset.lower()]
    if not filtered_indices:
        all_cats = sorted({str(r[cat_field]) for r in ds})
        raise ValueError(
            f"No rows match subset=`{cfg.subset}`. Available categories: {all_cats}"
        )
    if cfg.max_examples:
        filtered_indices = filtered_indices[: cfg.max_examples]
    overwatch.info(f"After filter to `{cfg.subset}`: {len(filtered_indices):,} rows")

    images_dir = cfg.out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    pope_entries = []
    written = 0
    skipped = 0
    for idx in tqdm(filtered_indices, desc="materialize"):
        row = ds[idx]
        row_id = _pick_id_field(row, idx)
        # IDs from POPE can contain "/" and other path-unfriendly chars — slug it.
        safe_id = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in row_id)
        img_rel = f"images/{safe_id}.jpg"
        img_abs = cfg.out_dir / img_rel

        if not img_abs.exists():
            try:
                pil = row[image_field]
                if not isinstance(pil, Image.Image):
                    raise TypeError(f"Expected PIL.Image for field `{image_field}`, got {type(pil)}")
                pil.convert("RGB").save(img_abs, format="JPEG", quality=90)
            except Exception as e:
                overwatch.info(f"[skip] row {row_id}: {type(e).__name__}: {e}")
                skipped += 1
                continue

        pope_entries.append({
            "id": safe_id,
            "image": img_rel,
            "question": row["question"],
            "label": _normalize_label(row[label_field]),
        })
        written += 1

    pope_json = cfg.out_dir / f"pope_{cfg.subset}.json"
    with open(pope_json, "w") as f:
        json.dump(pope_entries, f, indent=2)
    overwatch.info(f"Wrote {len(pope_entries):,} POPE entries -> {pope_json}  (skipped {skipped})")

    # Seeded shuffle, take first n_heldout, write chat schema for the cos/IoU/gen pass.
    rng = random.Random(cfg.seed)
    shuffled = list(pope_entries)
    rng.shuffle(shuffled)
    held = shuffled[: cfg.n_heldout]
    chat_entries = [
        {
            "id": e["id"],
            "image": e["image"],
            "conversations": [
                {"from": "human", "value": e["question"]},
                {"from": "gpt",   "value": e["label"]},
            ],
        }
        for e in held
    ]
    chat_json = cfg.out_dir / "chat_heldout.json"
    with open(chat_json, "w") as f:
        json.dump(chat_entries, f, indent=2)
    overwatch.info(f"Wrote {len(chat_entries):,} chat-format heldout entries -> {chat_json}")

    overwatch.info("Done. Pass these to the eval scripts:")
    overwatch.info(f"  --image_root  {cfg.out_dir}")
    overwatch.info(f"  --heldout_json {chat_json}")
    overwatch.info(f"  --pope_json   {pope_json}")


if __name__ == "__main__":
    main()
