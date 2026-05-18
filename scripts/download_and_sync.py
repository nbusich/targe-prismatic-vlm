"""
download_and_sync.py

End-to-end dataset prep for a fast Colab/Drive workflow:
  1. Download + extract the full Prismatic dataset to local disk (NOT Drive — Drive's per-file
     overhead makes 200k individual writes take hours).
  2. Split the chat JSON into train/test manifests by `train_pct`.
  3. Tar the dataset dir into ONE archive (single big file uploads to Drive at network speed).
  4. Copy the tar + JSON manifests to Google Drive.

Run with:
    python scripts/download_and_sync.py \
        --dataset_id llava-laion-cc-sbu-558k \
        --local_root /content/data \
        --drive_dest /content/drive/MyDrive/targe-prismatic-vlm/data \
        --train_pct 0.9

To rehydrate on a fresh Colab session:
    cp /content/drive/MyDrive/targe-prismatic-vlm/data/llava-laion-cc-sbu-558k.tar /content/
    mkdir -p /content/data/download
    tar -xf /content/llava-laion-cc-sbu-558k.tar -C /content/data/download/

The rehydrated layout matches what scripts/pretrain.py expects, so training launches with
`--dataset.dataset_root_dir /content/data` work unchanged.
"""

import json
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus

from prismatic.overwatch import initialize_overwatch
from prismatic.preprocessing import convert_to_jpg, download_extract

overwatch = initialize_overwatch(__name__)


# Map dataset_id -> name of the chat JSON inside the dataset dir. Matches DATASET_REGISTRY in
# prismatic/preprocessing/download.py.
CHAT_JSON_BY_DATASET = {
    "llava-laion-cc-sbu-558k": "chat.json",
    "llava-v1.5-instruct": "llava_v1_5_mix665k.json",
}


@dataclass
class DownloadAndSyncConfig:
    # fmt: off
    dataset_id: str = "llava-laion-cc-sbu-558k"     # Which Prismatic dataset to fetch (see CHAT_JSON_BY_DATASET).
    local_root: Path = Path("/content/data")        # Fast local disk; dataset lands at {local_root}/download/{dataset_id}/.
    drive_dest: Path = Path("/content/drive/MyDrive/targe-prismatic-vlm/data")  # Where the tar + JSONs end up.
    train_pct: float = 0.9                          # Fraction of entries for train; remainder for test. 1.0 = no test split.
    seed: int = 7                                   # Seed for train/test shuffle.
    skip_download: bool = False                     # Reuse existing local dataset (debug).
    skip_tar: bool = False                          # Skip tar step (debug).
    skip_sync: bool = False                         # Skip Drive copy step (debug).
    # fmt: on


def _split_train_test(entries: list, train_pct: float, seed: int) -> tuple[list, list]:
    if not 0.0 <= train_pct <= 1.0:
        raise ValueError(f"train_pct must be in [0.0, 1.0], got {train_pct}")
    ordered = list(entries)
    random.Random(seed).shuffle(ordered)
    n_train = int(round(train_pct * len(ordered)))
    return ordered[:n_train], ordered[n_train:]


def _tar_dataset(dataset_dir: Path, tar_path: Path) -> None:
    """Shell out to native tar — much faster than the tarfile module for many small files."""
    if tar_path.exists():
        overwatch.info(f"[sync] tar already exists at {tar_path}; skipping")
        return
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    overwatch.info(f"[sync] creating {tar_path} from {dataset_dir} (may take several minutes)...")
    # -C parent ensures the archive contains the dataset_id dir as its top-level entry,
    # so extracting into `data/download/` rebuilds the expected layout.
    subprocess.run(
        ["tar", "-cf", str(tar_path), "-C", str(dataset_dir.parent), dataset_dir.name],
        check=True,
    )
    overwatch.info(f"[sync] tar size: {tar_path.stat().st_size / 1e9:.2f} GB")


def _copy_to_drive(local_path: Path, drive_dir: Path) -> None:
    drive_dir.mkdir(parents=True, exist_ok=True)
    dest = drive_dir / local_path.name
    if dest.exists() and dest.stat().st_size == local_path.stat().st_size:
        overwatch.info(f"[sync] {dest} already present and same size; skipping")
        return
    overwatch.info(f"[sync] copying {local_path} -> {dest} ({local_path.stat().st_size / 1e9:.2f} GB)")
    shutil.copy2(local_path, dest)


@draccus.wrap()
def download_and_sync(cfg: DownloadAndSyncConfig) -> None:
    if cfg.dataset_id not in CHAT_JSON_BY_DATASET:
        raise ValueError(
            f"Unsupported dataset_id `{cfg.dataset_id}`. "
            f"Known: {sorted(CHAT_JSON_BY_DATASET)}"
        )

    dataset_dir = cfg.local_root / "download" / cfg.dataset_id
    chat_path = dataset_dir / CHAT_JSON_BY_DATASET[cfg.dataset_id]
    chat_test_path = dataset_dir / "chat_test.json"

    # 1. Download + extract via the existing Prismatic preprocessing pipeline.
    if cfg.skip_download:
        overwatch.info(f"[sync] skip_download=True; assuming dataset already at {dataset_dir}")
    else:
        overwatch.info(f"[sync] downloading + extracting `{cfg.dataset_id}` to {dataset_dir}")
        download_extract(cfg.dataset_id, root_dir=cfg.local_root)
        # Mirror scripts/preprocess.py special handling for OCR VQA GIFs/PNGs.
        if cfg.dataset_id == "llava-v1.5-instruct":
            convert_to_jpg(dataset_dir / "ocr_vqa" / "images")

    # 2. Train/test split — overwrite chat.json with the train slice, write chat_test.json alongside.
    if not chat_path.exists():
        raise FileNotFoundError(f"Expected chat JSON at {chat_path} after download; not found.")

    with chat_path.open() as f:
        entries = json.load(f)
    overwatch.info(f"[sync] chat JSON has {len(entries)} entries")

    train_entries, test_entries = _split_train_test(entries, cfg.train_pct, cfg.seed)
    with chat_path.open("w") as f:
        json.dump(train_entries, f)
    overwatch.info(f"[sync] wrote {chat_path} with {len(train_entries)} train entries")

    if test_entries:
        with chat_test_path.open("w") as f:
            json.dump(test_entries, f)
        overwatch.info(f"[sync] wrote {chat_test_path} with {len(test_entries)} test entries")
    else:
        chat_test_path.unlink(missing_ok=True)
        overwatch.info(f"[sync] train_pct={cfg.train_pct} → no test split written")

    # 3. Tar the dataset dir into ONE archive (Drive hates many small files).
    tar_path = cfg.local_root / f"{cfg.dataset_id}.tar"
    if cfg.skip_tar:
        overwatch.info("[sync] skip_tar=True; skipping tar step")
    else:
        _tar_dataset(dataset_dir, tar_path)

    # 4. Copy the tar + the JSON manifests to Drive. JSONs are tiny but handy to have unpacked
    #    for inspection / quick re-splits without untarring.
    if cfg.skip_sync:
        overwatch.info("[sync] skip_sync=True; skipping Drive copy")
        return

    if not cfg.skip_tar and tar_path.exists():
        _copy_to_drive(tar_path, cfg.drive_dest)
    _copy_to_drive(chat_path, cfg.drive_dest / cfg.dataset_id)
    if test_entries:
        _copy_to_drive(chat_test_path, cfg.drive_dest / cfg.dataset_id)

    overwatch.info(
        f"[sync] done. To rehydrate on a fresh runtime:\n"
        f"    cp {cfg.drive_dest / tar_path.name} /content/\n"
        f"    mkdir -p {cfg.local_root}/download\n"
        f"    tar -xf /content/{tar_path.name} -C {cfg.local_root}/download/\n"
        f"Then train with `--dataset.dataset_root_dir {cfg.local_root}`."
    )


if __name__ == "__main__":
    download_and_sync()
