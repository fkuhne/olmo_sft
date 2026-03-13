import os
import json
from typing import List

from pydantic import BaseModel

# ==============================================================================
# Pydantic Schemas for Strict API Formatting
# ==============================================================================
class SFTPair(BaseModel):
    prompt: str
    chosen: str

class SFTResponse(BaseModel):
    qa_pairs: List[SFTPair]

class DPOResponse(BaseModel):
    rejected: str

# ==============================================================================
# Provider Detection Helpers
# ==============================================================================
ANTHROPIC_MODELS = {"claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus", "claude-3-haiku", "claude-3-sonnet"}

def detect_provider(model: str) -> str:
    """Auto-detect provider from model name."""
    lower = model.lower()
    if "claude" in lower:
        return "anthropic"
    return "openai"

def _build_openai_client(api_key: str | None = None):
    from openai import OpenAI
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key=.")
    return OpenAI(api_key=key)

def _build_anthropic_client(api_key: str | None = None):
    try:
        import anthropic
    except ImportError:
        raise ImportError("The 'anthropic' package is required for Anthropic models. Run: pip install anthropic")
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key=.")
    return anthropic.Anthropic(api_key=key)

# ==============================================================================
# Synthetic Data Generator Class
# ==============================================================================
class TeacherModelSynthesizer:
    def __init__(
        self,
        domain: str = "technical documentation",
        api_key: str | None = None,
        model: str = "gpt-4o",
        provider: str | None = None,
    ):
        """
        Initializes the Teacher Model synthesizer with support for OpenAI and Anthropic.

        Args:
            domain: The subject-matter domain for prompt context (e.g., "medical devices", "automotive repair").
            api_key: API key for the chosen provider. Falls back to OPENAI_API_KEY or ANTHROPIC_API_KEY env vars.
            model: The teacher model identifier (e.g., "gpt-4o", "claude-3-5-sonnet-20241022").
            provider: "openai" or "anthropic". Auto-detected from model name if not specified.
        """
        self.model = model
        self.provider = provider or detect_provider(model)
        self.domain = domain

        print(f"Initializing Teacher Model Synthesizer ({self.provider}:{self.model}) for domain: '{domain}'...")

        if self.provider == "openai":
            self.client = _build_openai_client(api_key)
        elif self.provider == "anthropic":
            self.client = _build_anthropic_client(api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Use 'openai' or 'anthropic'.")

    # --------------------------------------------------------------------------
    # OpenAI implementation (structured outputs via Pydantic)
    # --------------------------------------------------------------------------
    def _openai_generate_sft(self, system_prompt: str, user_prompt: str) -> List[dict]:
        response = self.client.responses.parse(
            model=self.model,
            instructions=system_prompt,
            input=user_prompt,
            text_format=SFTResponse,
            temperature=0.3,
        )
        return [pair.model_dump() for pair in response.output_parsed.qa_pairs]

    def _openai_generate_dpo(self, system_prompt: str, user_prompt: str) -> str | None:
        response = self.client.responses.parse(
            model=self.model,
            instructions=system_prompt,
            input=user_prompt,
            text_format=DPOResponse,
            temperature=0.5,
        )
        return response.output_parsed.rejected

    # --------------------------------------------------------------------------
    # Anthropic implementation (JSON mode with manual Pydantic parsing)
    # --------------------------------------------------------------------------
    def _anthropic_call(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
        """Makes an Anthropic API call and returns the text content."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )
        return response.content[0].text

    def _anthropic_generate_sft(self, system_prompt: str, user_prompt: str) -> List[dict]:
        # Append JSON formatting instruction for Anthropic
        json_instruction = (
            "\n\nYou MUST respond with valid JSON only, matching this exact schema:\n"
            '{"qa_pairs": [{"prompt": "...", "chosen": "..."}]}\n'
            "Do not include any text outside the JSON object."
        )
        raw = self._anthropic_call(system_prompt + json_instruction, user_prompt, temperature=0.3)
        parsed = SFTResponse.model_validate_json(raw)
        return [pair.model_dump() for pair in parsed.qa_pairs]

    def _anthropic_generate_dpo(self, system_prompt: str, user_prompt: str) -> str | None:
        json_instruction = (
            "\n\nYou MUST respond with valid JSON only, matching this exact schema:\n"
            '{"rejected": "..."}\n'
            "Do not include any text outside the JSON object."
        )
        raw = self._anthropic_call(system_prompt + json_instruction, user_prompt, temperature=0.5)
        parsed = DPOResponse.model_validate_json(raw)
        return parsed.rejected

    # --------------------------------------------------------------------------
    # Public API (provider-agnostic)
    # --------------------------------------------------------------------------
    def generate_sft_pairs(self, markdown_chunk: str) -> List[dict]:
        """
        Takes a Docling markdown chunk and generates 2 to 3 diverse SFT QA pairs.
        """
        system_prompt = (
            f"You are an expert technical writer and data synthesizer for {self.domain}. "
            "Your task is to read documentation and generate highly accurate, realistic user questions "
            "and their corresponding step-by-step solutions.\n\n"
            "Strict Rules:\n"
            "1. Do NOT hallucinate or use external knowledge. The answer must be derived strictly from the text.\n"
            "2. If the text lacks actionable or 'how-to' information, output an empty array.\n"
            "3. Generate questions from multiple angles (e.g., direct action, symptom-based, clarification).\n"
            "4. Always reference the specific source context in the chosen response."
        )
        user_prompt = f"Text Chunk:\n\"\"\"{markdown_chunk}\"\"\"\n\nGenerate 2 to 3 Question-Answer pairs."

        try:
            if self.provider == "openai":
                return self._openai_generate_sft(system_prompt, user_prompt)
            else:
                return self._anthropic_generate_sft(system_prompt, user_prompt)
        except Exception as e:
            print(f"SFT Generation Error: {e}")
            return []

    def generate_dpo_rejection(self, prompt: str, chosen: str) -> str | None:
        """
        Takes a generated SFT pair and generates a subtly flawed 'rejected' answer for DPO alignment.
        """
        system_prompt = (
            "You are an AI safety and alignment expert. Your task is to generate a 'rejected' response "
            "for a technical support question.\n\n"
            "Strict Rules:\n"
            "1. It must look highly plausible and confidently written.\n"
            "2. It must contain a critical factual error (e.g., wrong sequence, wrong component, or subtle hallucination).\n"
            "3. It must not be cartoonishly evil or obvious; it must force the fine-tuned model to pay close attention.\n"
        )
        user_prompt = f"User Question: \"{prompt}\"\nTrue Answer: \"{chosen}\"\n\nGenerate the plausible but factually incorrect 'rejected' response."

        try:
            if self.provider == "openai":
                return self._openai_generate_dpo(system_prompt, user_prompt)
            else:
                return self._anthropic_generate_dpo(system_prompt, user_prompt)
        except Exception as e:
            print(f"DPO Generation Error: {e}")
            return None

    def process_chunk(self, markdown_chunk: str) -> List[dict]:
        """
        Orchestrates the full pipeline for a single chunk: SFT generation -> DPO generation.
        """
        final_tuples = []

        # 1. Generate the valid QA pairs (Chosen)
        sft_pairs = self.generate_sft_pairs(markdown_chunk)

        if not sft_pairs:
            return final_tuples  # Skip if no actionable data was found

        # 2. Generate the negative examples (Rejected)
        for pair in sft_pairs:
            rejected_text = self.generate_dpo_rejection(pair['prompt'], pair['chosen'])

            if rejected_text:
                final_tuples.append({
                    "prompt": pair['prompt'],
                    "chosen": pair['chosen'],
                    "rejected": rejected_text
                })

        return final_tuples

# ==============================================================================
# Execution Example for the AI Agent
# ==============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Teacher Model Synthetic Data Generator")
    parser.add_argument("--model", default="gpt-4o", help="Teacher model ID (e.g., gpt-4o, claude-3-5-sonnet-20241022)")
    parser.add_argument("--provider", default=None, help="API provider: 'openai' or 'anthropic' (auto-detected from model name)")
    parser.add_argument("--domain", default="technical documentation", help="Subject-matter domain")
    args = parser.parse_args()

    synthesizer = TeacherModelSynthesizer(domain=args.domain, model=args.model, provider=args.provider)

    # Example chunk
    example_markdown_chunk = (
        "### [Source Context: Product User Guide]\n\n"
        "**Clearing a Paper Jam in the ADF**\n"
        "1. Lift the document feeder cover.\n"
        "2. Gently pull the jammed paper out of the rollers.\n"
        "3. Close the document feeder cover until it snaps into place."
    )

    print("Synthesizing SFT and DPO data...")
    generated_data = synthesizer.process_chunk(example_markdown_chunk)

    for i, data in enumerate(generated_data):
        print(f"\n--- Synthetic Tuple {i+1} ---")
        print(json.dumps(data, indent=2))
