"""
make_smoke_dataset.py

Build a tiny subset of the LLaVa-Pretrain align-stage dataset (`llava-laion-cc-sbu-558k`)
for fast wiring smoke tests on memory-constrained machines (e.g., Colab T4).

Mechanics:
  1. Download `blip_laion_cc_sbu_558k.json` (~200 MB) once, subsample to N entries.
  2. Use `python-remotezip` to partial-extract ONLY the N referenced JPGs from the
     remote `images.zip` (~24 GB) via HTTP range requests — no full download.

Result lands under {root_dir}/download/llava-laion-cc-sbu-558k/ with the same path
layout the stock `LLaVa_V15_Config` expects, so training launches with
`--dataset.type llava-v15` work unchanged.

Run with:
    python scripts/make_smoke_dataset.py --num_samples 2000 --root_dir data

Reruns are idempotent — existing images are skipped, JSON is rewritten with the
current `num_samples` (so you can ratchet N up across runs without re-downloading
images already on disk).

NOTE: This script is for SMOKE TESTS ONLY. Loss curves from training on N=2000 random
LLaVa samples are not reportable. For real runs use the full dataset:
    python scripts/preprocess.py --dataset_id llava-laion-cc-sbu-558k
"""

import json
import random
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus

from prismatic.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


JSON_URL = (
    "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/"
    "resolve/main/blip_laion_cc_sbu_558k.json"
)
ZIP_URL = (
    "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/"
    "resolve/main/images.zip"
)
DATASET_SUBDIR = "llava-laion-cc-sbu-558k"


@dataclass
class SmokeDatasetConfig:
    # fmt: off
    num_samples: int = 2000                         # How many samples to keep. Suggested: 100 (pure wiring smoke), 2000 (sane loss trend), 10000 (slow but real-ish).
    train_pct: float = 1.0                          # Fraction of `num_samples` that goes to train; remainder goes to test. 1.0 = no test split.
    root_dir: Path = Path("data")                   # Mirrors scripts/preprocess.py — files land at {root_dir}/download/{DATASET_SUBDIR}/
    seed: Optional[int] = 7                         # Seed for random subsampling + train/test split. None = take the first N (deterministic but biased), no shuffle for split.
    keep_full_json: bool = False                    # If True, also stash the un-subsampled JSON next to chat.json as chat.json.full (for resampling later without re-downloading)
    # fmt: on


def _download_json(dest: Path) -> None:
    """Download the full 200 MB align JSON into `dest`. Skips if file already exists."""
    if dest.exists():
        overwatch.info(f"[smoke] full JSON already at {dest}; skipping download")
        return
    overwatch.info(f"[smoke] downloading full chat JSON ({JSON_URL}) -> {dest} (~200 MB)")
    urllib.request.urlretrieve(JSON_URL, dest)


def _subsample(entries: list, n: int, seed: Optional[int]) -> list:
    if n >= len(entries):
        overwatch.warning(
            f"[smoke] num_samples={n} >= dataset size ({len(entries)}); using everything"
        )
        return entries
    if seed is None:
        return entries[:n]
    rng = random.Random(seed)
    return rng.sample(entries, n)


def _ensure_remotezip() -> None:
    try:
        import remotezip  # noqa: F401
    except ImportError:
        overwatch.error(
            "[smoke] python-remotezip is not installed. Install with:\n"
            "    pip install python-remotezip\n"
            "Then re-run this script."
        )
        sys.exit(1)


def _extract_images(needed: set[str], dest_root: Path) -> None:
    """Partial-extract the image files listed in `needed` from the remote zip.

    Skips entries already on disk. Uses HTTP range requests under the hood so
    we never download the full 24 GB."""
    from remotezip import RemoteZip

    to_fetch = [name for name in needed if not (dest_root / name).exists()]
    already = len(needed) - len(to_fetch)
    if already:
        overwatch.info(f"[smoke] {already}/{len(needed)} images already on disk; skipping")
    if not to_fetch:
        overwatch.info("[smoke] nothing to fetch; image set is complete")
        return

    overwatch.info(
        f"[smoke] partial-extracting {len(to_fetch)} images from {ZIP_URL} "
        "(this opens a single ranged HTTP session)..."
    )
    failures: list[tuple[str, str]] = []
    with RemoteZip(ZIP_URL) as zf:
        for i, name in enumerate(to_fetch, start=1):
            try:
                zf.extract(name, path=dest_root)
            except Exception as exc:  # noqa: BLE001 — keep going on per-file failures
                failures.append((name, str(exc)))
            if i % 200 == 0:
                overwatch.info(f"[smoke]   {i}/{len(to_fetch)} extracted")

    if failures:
        overwatch.warning(
            f"[smoke] {len(failures)} images failed to extract; first 5: {failures[:5]}"
        )
    overwatch.info(f"[smoke] done: extracted {len(to_fetch) - len(failures)} new images")


def _split_train_test(entries: list, train_pct: float, seed: Optional[int]) -> tuple[list, list]:
    """Shuffle (if `seed` is set) and split into train/test. `train_pct` is in [0.0, 1.0]."""
    if not 0.0 <= train_pct <= 1.0:
        raise ValueError(f"train_pct must be in [0.0, 1.0], got {train_pct}")
    if seed is None:
        # Deterministic but biased — preserves source order.
        ordered = list(entries)
    else:
        ordered = list(entries)
        random.Random(seed + 1).shuffle(ordered)
    n_train = int(round(train_pct * len(ordered)))
    return ordered[:n_train], ordered[n_train:]


@draccus.wrap()
def make_smoke_dataset(cfg: SmokeDatasetConfig) -> None:
    overwatch.info(
        f"[smoke] building a {cfg.num_samples}-sample subset of {DATASET_SUBDIR} "
        f"(train_pct={cfg.train_pct}) under {cfg.root_dir}/download/{DATASET_SUBDIR}/"
    )

    dataset_root = cfg.root_dir / "download" / DATASET_SUBDIR
    dataset_root.mkdir(parents=True, exist_ok=True)

    chat_full_path = dataset_root / "chat.json.full"
    chat_path = dataset_root / "chat.json"
    chat_test_path = dataset_root / "chat_test.json"

    _download_json(chat_full_path)
    with chat_full_path.open() as f:
        full_entries = json.load(f)
    overwatch.info(f"[smoke] full JSON has {len(full_entries)} entries")

    subset = _subsample(full_entries, cfg.num_samples, cfg.seed)
    train_entries, test_entries = _split_train_test(subset, cfg.train_pct, cfg.seed)

    with chat_path.open("w") as f:
        json.dump(train_entries, f)
    overwatch.info(f"[smoke] wrote {chat_path} with {len(train_entries)} train entries")

    if test_entries:
        with chat_test_path.open("w") as f:
            json.dump(test_entries, f)
        overwatch.info(f"[smoke] wrote {chat_test_path} with {len(test_entries)} test entries")
    else:
        # Drop a stale test JSON from a previous run with a smaller train_pct, so we don't lie about the split.
        chat_test_path.unlink(missing_ok=True)
        overwatch.info(f"[smoke] train_pct={cfg.train_pct} → no test split written")

    needed = {entry["image"] for entry in train_entries} | {entry["image"] for entry in test_entries}
    _ensure_remotezip()
    _extract_images(needed, dataset_root)

    if not cfg.keep_full_json:
        chat_full_path.unlink(missing_ok=True)
        overwatch.info(f"[smoke] removed {chat_full_path} (set keep_full_json=true to retain)")

    overwatch.info(
        f"[smoke] ready. Train manifest: {chat_path} ({len(train_entries)} entries); "
        f"test manifest: {chat_test_path if test_entries else '<none>'} ({len(test_entries)} entries). "
        f"Launch training with `--dataset.type llava-v15 --dataset.dataset_root_dir {cfg.root_dir}`"
    )


if __name__ == "__main__":
    make_smoke_dataset()
