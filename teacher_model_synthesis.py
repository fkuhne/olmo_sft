import os
import json
from openai import OpenAI
from pydantic import BaseModel
from typing import List

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
# Synthetic Data Generator Class
# ==============================================================================
class TeacherModelSynthesizer:
    def __init__(self, api_key=None, model="gpt-4o"):
        """
        Initializes the OpenAI client. 
        GPT-4o is recommended for its high instruction-following adherence and speed.
        """
        print(f"Initializing Teacher Model Synthesizer ({model})...")
        # Agent Directive: Ensure OPENAI_API_KEY is set in the environment variables
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def generate_sft_pairs(self, markdown_chunk: str) -> List[dict]:
        """
        Takes a Docling markdown chunk and generates 2 to 3 diverse SFT QA pairs.
        """
        system_prompt = (
            "You are an expert technical writer and data synthesizer for HP printer troubleshooting. "
            "Your task is to read technical documentation and generate highly accurate, realistic user questions "
            "and their corresponding step-by-step solutions.\n\n"
            "Strict Rules:\n"
            "1. Do NOT hallucinate or use external knowledge. The answer must be derived strictly from the text.\n"
            "2. If the text lacks actionable troubleshooting or 'how-to' info, output an empty array.\n"
            "3. Generate questions from multiple angles (e.g., direct action, symptom-based).\n"
            "4. Always mention the specific printer model in the chosen response."
        )

        user_prompt = f"Text Chunk:\n\"\"\"{markdown_chunk}\"\"\"\n\nGenerate 2 to 3 Question-Answer pairs."

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=SFTResponse,
                temperature=0.3 # Low temperature for factual grounding
            )
            # Convert Pydantic objects back to a list of dicts
            return [pair.model_dump() for pair in response.choices[0].message.parsed.qa_pairs]
        
        except Exception as e:
            print(f"SFT Generation Error: {e}")
            return []

    def generate_dpo_rejection(self, prompt: str, chosen: str) -> str:
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
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=DPOResponse,
                temperature=0.5 # Slightly higher temp to encourage creative flaw generation
            )
            return response.choices[0].message.parsed.rejected
            
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
            return final_tuples # Skip if no actionable data was found
            
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
    # Ensure the agent has the API key exported in the environment
    synthesizer = TeacherModelSynthesizer()
    
    # Example chunk coming from the Docling script
    example_markdown_chunk = (
        "### [Device Context: HP OfficeJet Pro 9015]\n\n"
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
