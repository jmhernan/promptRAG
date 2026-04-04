import torch
torch.backends.mps.enable_fallback_for_missing_ops = True

from transformers import AutoTokenizer, pipeline as hf_pipeline


class HuggingFaceLLM:
    """HuggingFace transformers pipeline for text generation."""

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        device_map: str = "mps",
        max_new_tokens: int = 256,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipe = hf_pipeline(
            "text-generation",
            model=model_name,
            tokenizer=self.tokenizer,
            device_map=self._resolve_device_map(device_map),
            torch_dtype=torch.float16,
        )

    @staticmethod
    def _resolve_device_map(requested: str) -> str:
        if requested == "mps" and torch.backends.mps.is_available():
            return "mps"
        if requested == "cuda" and torch.cuda.is_available():
            return "auto"
        return "cpu"

    def generate(self, prompt: str) -> str:
        """Generate text from a formatted prompt string."""
        result = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        return result[0]["generated_text"]

    def generate_from_messages(self, messages: list[dict]) -> str:
        """Generate from a chat messages list. Applies the model's chat template."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.generate(prompt)
