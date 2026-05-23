"""
check_smollm2_labels.py

Quick sanity check for SmolLM2 + Prismatic:

  1. Builds a fake multimodal conversation via SmolLM2ChatPromptBuilder.
  2. Tokenizes it the same way FinetuneDataset does.
  3. Prints input_ids/labels alignment so we can confirm:
       - Position 0 (which Prismatic forces to IGNORE_INDEX for multimodal)
         is a SAFE token to ignore (e.g. <|im_start|> control token, NOT
         an assistant content token we actually want to train on).
       - The user-turn tokens are all IGNORE_INDEX (-100).
       - The assistant-turn tokens are NOT IGNORE_INDEX (loss is computed).
       - The trailing <|im_end|> on the assistant turn is supervised.

Run:
    python scripts/check_smollm2_labels.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from prismatic.models.backbones.llm.prompting import SmolLM2ChatPromptBuilder

IGNORE_INDEX = -100
HF_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


def build_example(prompt_builder_cls):
    """Mimic FinetuneDataset.__getitem__ token/label construction."""
    tokenizer = AutoTokenizer.from_pretrained(HF_ID)
    conversation = [
        {"from": "human", "value": "<image>\nIs there a person in the image?"},
        {"from": "gpt",   "value": "Yes, there is a person standing by the car."},
    ]
    pb = prompt_builder_cls(model_family="prismatic")

    input_ids, labels = [], []
    for turn_idx, turn in enumerate(conversation):
        msg = pb.add_turn(turn["from"], turn["value"])
        ids = tokenizer(msg, add_special_tokens=(turn_idx == 0)).input_ids
        if (turn_idx % 2) == 0:
            lbl = [IGNORE_INDEX] * len(ids)
        else:
            lbl = list(ids)
        input_ids.extend(ids)
        labels.extend(lbl)

    # FinetuneDataset forces position-0 label to IGNORE for multimodal examples.
    if labels:
        labels[0] = IGNORE_INDEX

    return tokenizer, input_ids, labels, pb.get_prompt()


def main():
    tokenizer, input_ids, labels, prompt_str = build_example(SmolLM2ChatPromptBuilder)

    print("=" * 72)
    print("RAW PROMPT STRING (what SmolLM2ChatPromptBuilder.get_prompt() returns):")
    print("=" * 72)
    print(prompt_str)
    print()

    print("=" * 72)
    print(f"Tokenizer class: {type(tokenizer).__name__}")
    print(f"bos={tokenizer.bos_token!r}  eos={tokenizer.eos_token!r}  pad={tokenizer.pad_token!r}")
    print(f"add_bos_token attr: {getattr(tokenizer, 'add_bos_token', '<missing>')}")
    print(f"Sequence length: {len(input_ids)}")
    print("=" * 72)

    # Per-token alignment
    print(f"{'idx':>4} | {'tok_id':>7} | {'label':>6} | token")
    print("-" * 72)
    for i, (tid, lid) in enumerate(zip(input_ids, labels)):
        tok = tokenizer.convert_ids_to_tokens(tid)
        marker = "  <-- supervised" if lid != IGNORE_INDEX else ""
        print(f"{i:>4} | {tid:>7} | {lid:>6} | {tok!r}{marker}")

    # Specific checks
    print()
    print("=" * 72)
    print("CHECKS")
    print("=" * 72)

    pos0_tok = tokenizer.convert_ids_to_tokens(input_ids[0])
    print(f"[1] Position 0 token = {pos0_tok!r}")
    print(f"    Position 0 label = {labels[0]} (forced to IGNORE by FinetuneDataset)")
    if pos0_tok.startswith("<|im_start|>") or pos0_tok in ("<|endoftext|>", "<s>", "<|im_start|>"):
        print("    -> OK: ignoring a control/BOS-style token. No real supervision lost.")
    else:
        print("    -> WARN: position 0 is a content token; forcing it to IGNORE drops a label.")

    n_supervised = sum(1 for l in labels if l != IGNORE_INDEX)
    n_total = len(labels)
    print(f"[2] Supervised tokens: {n_supervised}/{n_total}")
    if n_supervised == 0:
        print("    -> FAIL: nothing is supervised.")
    else:
        first_sup = next(i for i, l in enumerate(labels) if l != IGNORE_INDEX)
        last_sup  = max(i for i, l in enumerate(labels) if l != IGNORE_INDEX)
        print(f"    First supervised idx={first_sup} tok={tokenizer.convert_ids_to_tokens(input_ids[first_sup])!r}")
        print(f"    Last  supervised idx={last_sup} tok={tokenizer.convert_ids_to_tokens(input_ids[last_sup])!r}")
        # `<|im_end|>` is the assistant turn terminator. It may not be the LAST
        # supervised token (a trailing "\n" usually follows it) — what matters is
        # that it appears *somewhere* in the supervised span.
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        im_end_supervised = any(
            tid == im_end_id and lid != IGNORE_INDEX
            for tid, lid in zip(input_ids, labels)
        )
        if im_end_supervised:
            print("    -> OK: <|im_end|> is supervised on the assistant turn.")
        else:
            print("    -> WARN: <|im_end|> never appears as a supervised label.")

    # Roundtrip: does the tokenized text reproduce ChatML?
    decoded = tokenizer.decode(input_ids)
    print(f"[3] Decoded round-trip:")
    print(decoded)


if __name__ == "__main__":
    main()
