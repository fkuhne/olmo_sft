# OLMo 2 1B Domain Adaptation: HP Printer QA

**Repository Overview:** This repository contains the complete end-to-end pipeline for extracting knowledge from raw HP printer PDF manuals, synthesizing a high-quality QA dataset using a Teacher Model, and aligning the OLMo 2 1B foundation model using Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO).

---

## I. Documentation & Master Plans

These files contain the theoretical frameworks and step-by-step instructions for the project. They act as the "brain" for any AI agent executing the repository.

* **`sft_plan.md`**
  * **Purpose:** The Master Execution Blueprint. It contains the exact Python scripts, hyperparameter configurations, and terminal commands required to run Phases 1 through 6 (Compute Provisioning, SFT Training, DPO Alignment, Evaluation, and vLLM Deployment).
  * **How to Use:** Read this file top-to-bottom to understand the chronological order of the entire model-training lifecycle.

* **`data_engineering_spec.md`**
  * **Purpose:** The Theoretical Framework for Phase 2. It defines the strict rules for PDF chunking, metadata injection, multi-angle question synthesis, and the "hard negative" protocol for DPO. 
  * **How to Use:** Reference this document to understand the logic behind the Python data scripts and to verify the strict JSON schema required for the dataset.

---

## II. Infrastructure & Environment

Scripts dedicated to preparing the hardware and software dependencies.

* **`runpod_setup.sh`**
  * **Purpose:** The environment initialization script. It updates the RunPod Ubuntu environment, installs the Hugging Face ML stack (`transformers`, `peft`, `trl`), compiles Flash Attention 2, and installs the `docling` PDF parser. 
  * **How to Use:** Execute this Bash script the moment you connect to the RunPod GPU instance via SSH. It ensures the `/workspace` directory is properly formatted so data is not lost during a pod restart.

---

## III. Phase 2: The Data Generation Pipeline

These modular Python scripts handle the autonomous conversion of raw PDFs into a strictly formatted, deduplicated ML dataset. 

* **`build_dataset.py`**
  * **Purpose:** The Phase 2 Orchestrator. This is the master loop that ties the extraction, synthesis, and deduplication scripts together. It scans a local `./manuals` folder, processes every PDF, handles API timeouts gracefully, deduplicates the data in real-time, and saves the final output.
  * **How to Use:** Once the PDFs are loaded into the `./manuals` folder, run `python build_dataset.py`. The output will be the pristine `hp_alignment_dataset.jsonl` file, completely ready for Phase 3 training.

* **`pdf_extractor.py`**
  * **Purpose:** The Document Ingestion Module. It uses IBM's `docling` library to visually analyze PDF layouts, perfectly preserve troubleshooting tables, and chunk the document semantically. It injects the device name into every chunk to prevent model amnesia.
  * **How to Use:** This is a dependency module imported and utilized by `build_dataset.py`.

* **`teacher_model_synthesis.py`**
  * **Purpose:** The Synthetic Generation Module. It takes the text chunks from the extractor and passes them to OpenAI's API using Pydantic Structured Outputs. It generates the correct "chosen" answers and creates the subtly flawed "rejected" answers for DPO alignment.
  * **How to Use:** This is a dependency module imported by `build_dataset.py`. It requires the `OPENAI_API_KEY` to be set in the environment variables.

* **`deduplicate_dataset.py`**
  * **Purpose:** The Quality Control Module. It calculates the semantic vector embeddings of every generated question using `sentence-transformers`. If a new question is >85% similar to an existing one, it drops it to prevent the 1B model from overfitting on repetitive phrasing.
  * **How to Use:** This is a dependency module imported by `build_dataset.py`.

---

## IV. Suggested Execution Order

1. Connect to the GPU Pod and run `bash runpod_setup.sh`.
2. Export the OpenAI API key: `export OPENAI_API_KEY="your_key_here"`.
3. Upload your target PDFs to the `/workspace/hp-qa-model/manuals` directory.
4. Execute `python build_dataset.py` to generate the JSONL dataset.
5. Follow the exact Python scripts detailed inside `sft_plan.md` to run Phase 3 (SFT) and Phase 4 (DPO).
