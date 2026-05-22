"""
datasets.py

PyTorch Dataset Definitions for Prismatic models; supports processing for both the `align` and `finetune` stages, with
utilities for formatting conversations during the `finetune` stage subject to the given LLM backbone's expected
formatting (e.g., SYS_PROMPT + USER: ... ASSISTANT: ... for Vicuña v1.5 Chat models).

We currently only support Map-style Datasets; assumes that all files (annotations, images) are on local disk, and that
random access image reading is relatively cheap/fast.
"""

import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple, Type

import torch
from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import Dataset
from transformers import CodeGenTokenizerFast, LlamaTokenizerFast, PreTrainedTokenizerBase

from prismatic.overwatch import initialize_overwatch
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform

# Allow PIL to load partial JPEGs (truncated downloads) instead of raising OSError mid-decode.
ImageFile.LOAD_TRUNCATED_IMAGES = True

overwatch = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

# Errors raised by PIL on unreadable / corrupt / missing image files. We skip these rather than crash training.
_BAD_IMAGE_ERRORS = (UnidentifiedImageError, OSError, FileNotFoundError, SyntaxError, ValueError)


class AlignDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        chat_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__()
        self.chat_json, self.image_dir = chat_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.dataset_type = "align"

        # Create Prompt Template
        self.prompt_template = "{caption}" + self.tokenizer.eos_token

        # Load Chat JSON
        with open(self.chat_json, "r") as f:
            self.examples = json.load(f)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Following the *actual* code executed from the LLaVa codebase, during the "align" phase, we actually discard
        the "prompt" from the human, and instead directly predict the caption from the image.

        As a concrete example given the "raw data" for the first example:
            example = self.examples[0]["conversations"]` = {
                [
                    {"from": "human", "value": "Render a clear and concise summary of the photo.\n<image>"},
                    {"from": "gpt", "value": "select luxury furniture 3 - inch gel memory foam mattress topper"}
                ]
            }

        Return =>> self.tokenizer("<image> select luxury furniture 3 - inch gel memory foam mattress topper\n")

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        # Bounded skip-and-retry loop — prevents worker stack overflow if many consecutive
        # rows are malformed (recursion through __getitem__ would otherwise hit Python's
        # recursion limit inside the dataloader worker).
        MAX_TRIES = 32
        n = len(self.examples)
        for _ in range(MAX_TRIES):
            try:
                example = self.examples[idx]
                image_path = Path(example["image"])
                conversation = example["conversations"]

                # Some chat.json variants store `conversations` as a JSON-encoded *string*
                # instead of an actual list. Decode here so downstream consumers see the
                # canonical shape (`list[dict]`).
                if isinstance(conversation, str):
                    conversation = json.loads(conversation)

                # Structural validation: align-stage expects exactly (human, gpt) with each
                # turn a dict containing a string `value` and `<image>` only in the human turn.
                # Anything else (string elements, missing keys, wrong turn count, image in gpt)
                # is treated as malformed and skipped.
                if (
                    not isinstance(conversation, list)
                    or len(conversation) != 2
                    or not all(isinstance(t, dict) and isinstance(t.get("value"), str) for t in conversation)
                    or "<image>" in conversation[-1]["value"]
                ):
                    overwatch.warning(f"[AlignDataset] Skipping malformed conversation idx={idx}")
                    idx = (idx + 1) % n
                    continue

                # Format Caption --> {caption}{eos_token}
                caption = self.prompt_template.format(caption=conversation[-1]["value"].strip())

                # input_ids = "<s> p1 p2 p3 ... <caption_text> \n"; labels copy with <BOS> -> IGNORE.
                # Shifting happens INSIDE HF LLM.forward() when labels= is passed.
                input_ids = self.tokenizer(caption, truncation=True, return_tensors="pt").input_ids[0]
                labels = copy.deepcopy(input_ids)
                labels[0] = IGNORE_INDEX

                # Process Image — fall back to the next example if corrupt/missing.
                try:
                    pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert("RGB"))
                except _BAD_IMAGE_ERRORS as ex:
                    overwatch.warning(
                        f"[AlignDataset] Skipping bad image idx={idx} path={image_path} "
                        f"({type(ex).__name__}: {ex})"
                    )
                    idx = (idx + 1) % n
                    continue

                return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

            except (KeyError, TypeError, AttributeError, IndexError, ValueError) as ex:
                # Catch any other structural defect — e.g. missing keys, wrong types.
                overwatch.warning(
                    f"[AlignDataset] Skipping idx={idx} due to {type(ex).__name__}: {ex}"
                )
                idx = (idx + 1) % n
                continue

        raise RuntimeError(
            f"[AlignDataset] no usable example after {MAX_TRIES} tries starting at idx={idx}. "
            "Dataset is likely badly corrupted or paths are wrong."
        )

    def get_modality_lengths(self, n_image_patches: int) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example
            conv = example.get("conversations", [])
            if isinstance(conv, str):
                try:
                    conv = json.loads(conv)
                except (ValueError, TypeError):
                    conv = []
            n_words = 0
            for turn in conv if isinstance(conv, list) else []:
                if isinstance(turn, dict) and isinstance(turn.get("value"), str):
                    n_words += len(turn["value"].replace("<image>", "").split())
            modality_lengths.append((is_multimodal, (n_image_patches + n_words) if is_multimodal else n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)


class FinetuneDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        instruct_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        super().__init__()
        self.instruct_json, self.image_dir = instruct_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset_type = "finetune"

        # Load Instruct JSON
        with open(self.instruct_json, "r") as f:
            self.examples = json.load(f)

    # === Unimodal + Multimodal Handling ===
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Unlike the *align* stage handling, for the *finetune* stage, we actually need to handle multiple "turns" of
        dialog grounded in a single image.

        To do this, we leverage the `prompt_builder_fn` which instantiates a PromptBuilder object. By calling the
        methods for adding turns and getting a prompt, we ensure proper formatting and consistency for each example.

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        MAX_TRIES = 32
        n = len(self.examples)
        for _ in range(MAX_TRIES):
            try:
                example = self.examples[idx]
                conversation = example["conversations"]

                # Decode JSON-string `conversations` if present (some variants store the
                # turn list as an encoded string instead of a list).
                if isinstance(conversation, str):
                    conversation = json.loads(conversation)

                # Structural validation: each turn must be a dict with string `from`+`value`.
                if not isinstance(conversation, list) or len(conversation) < 2 or not all(
                    isinstance(t, dict) and isinstance(t.get("from"), str) and isinstance(t.get("value"), str)
                    for t in conversation
                ):
                    overwatch.warning(f"[FinetuneDataset] Skipping malformed conversation idx={idx}")
                    idx = (idx + 1) % n
                    continue

                # Create Prompt Builder --> add each message sequentially.
                prompt_builder, input_ids, labels = self.prompt_builder_fn(model_family="prismatic"), [], []
                for turn_idx, turn in enumerate(conversation):
                    msg = prompt_builder.add_turn(turn["from"], turn["value"])
                    if isinstance(self.tokenizer, LlamaTokenizerFast):
                        msg = msg.rstrip()
                    elif isinstance(self.tokenizer, CodeGenTokenizerFast):
                        pass
                    else:
                        raise ValueError(f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!")

                    turn_input_ids = self.tokenizer(msg, add_special_tokens=turn_idx == 0).input_ids
                    turn_labels = (
                        [IGNORE_INDEX for _ in range(len(turn_input_ids))]
                        if (turn_idx % 2) == 0
                        else list(turn_input_ids)
                    )
                    input_ids.extend(turn_input_ids)
                    labels.extend(turn_labels)

                # Tensorize =>> shifting happens INSIDE HF LLM.forward() when labels= is passed.
                input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
                input_ids = input_ids[: self.tokenizer.model_max_length]
                labels = labels[: self.tokenizer.model_max_length]

                if "image" in example:
                    image_path = Path(example["image"])
                    labels[0] = IGNORE_INDEX

                    try:
                        pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert("RGB"))
                    except _BAD_IMAGE_ERRORS as ex:
                        overwatch.warning(
                            f"[FinetuneDataset] Skipping bad image idx={idx} path={image_path} "
                            f"({type(ex).__name__}: {ex})"
                        )
                        idx = (idx + 1) % n
                        continue

                    return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

                # Unimodal (language-only) — Collator handles the batch.
                return dict(pixel_values=None, input_ids=input_ids, labels=labels)

            except (KeyError, TypeError, AttributeError, IndexError) as ex:
                overwatch.warning(
                    f"[FinetuneDataset] Skipping idx={idx} due to {type(ex).__name__}: {ex}"
                )
                idx = (idx + 1) % n
                continue

        raise RuntimeError(
            f"[FinetuneDataset] no usable example after {MAX_TRIES} tries starting at idx={idx}. "
            "Dataset is likely badly corrupted or paths are wrong."
        )

    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example
            conv = example.get("conversations", [])
            if isinstance(conv, str):
                try:
                    conv = json.loads(conv)
                except (ValueError, TypeError):
                    conv = []
            n_words = 0
            for turn in conv if isinstance(conv, list) else []:
                if isinstance(turn, dict) and isinstance(turn.get("value"), str):
                    n_words += len(turn["value"].split())
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)
