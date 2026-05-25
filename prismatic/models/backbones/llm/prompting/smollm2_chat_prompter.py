"""
smollm2_chat_prompter.py

PromptBuilder for SmolLM2-Instruct (HuggingFaceTB/SmolLM2-*-Instruct).

SmolLM2-Instruct was chat-tuned with a ChatML-style template:

    <|im_start|>system
    {system_prompt}<|im_end|>
    <|im_start|>user
    {user_message}<|im_end|>
    <|im_start|>assistant
    {assistant_message}<|im_end|>

The tokenizer is a GPT-2 style BPE that does NOT auto-prepend BOS (see
`SPECIAL_CASES` in base_llm.py), so `<|im_start|>` — which IS the bos_token —
must appear explicitly at the start of the prompt. Each turn begins with its
own `<|im_start|>` marker, so the first one in the prompt naturally serves as
BOS; no extra prefix is needed.
"""

from typing import Optional

from prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder


SMOLLM2_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant named SmolLM, trained by Hugging Face."
)


class SmolLM2ChatPromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)

        self.bos, self.eos = "<|im_start|>", "<|im_end|>"
        self.system_prompt = (system_prompt or SMOLLM2_DEFAULT_SYSTEM_PROMPT).strip()

        self.wrap_human = lambda msg: f"<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}<|im_end|>\n"

        # Built up over turns. First human turn embeds the system prompt; subsequent turns don't.
        self.prompt, self.turn_count = "", 0

    def _wrap_first_human(self, message: str) -> str:
        return (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        if self.turn_count == 0:
            wrapped_message = self._wrap_first_human(message)
        elif (self.turn_count % 2) == 0:
            wrapped_message = self.wrap_human(message)
        else:
            wrapped_message = self.wrap_gpt(message)

        self.prompt += wrapped_message
        self.turn_count += 1
        return wrapped_message

    def get_potential_prompt(self, message: str) -> str:
        prompt_copy = str(self.prompt)
        if self.turn_count == 0:
            prompt_copy += self._wrap_first_human(message.strip())
        else:
            prompt_copy += self.wrap_human(message.strip())
        return prompt_copy.rstrip()

    def get_prompt(self) -> str:
        return self.prompt.rstrip()
