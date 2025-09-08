# NyayaNet Tokenizer & Preprocessing

This repository documents the preprocessing and tokenizer pipeline for LegalTech, a Mixture of Experts (MoE) model designed to analyze and interpret Indian legal judgments, with a focus on Indian courts and it judgements.

## Overview

LegalTech aims to advance legal NLP by developing a custom tokenizer and preprocessing framework tailored to Indian judicial text. This repository captures the end-to-end process, from raw dataset handling to generating a LORA format to custom tokenizer building. The work is intended as a novel contribution to legal AI, suitable for academic publication.

### Key Components
- **Preprocessing**: Cleaning and structuring raw legal datasets from HuggingFace.
- **LoRA Conversion**: Transforming the raw data from the datasets into a low-rank adaptation (LoRA) format for downstream LLM training.
- **Tokenizer Training**: Fine-tuning a BertWordPiece tokenizer on processed text to capture legal-specific vocabulary.
- **Foundation**: Serves as the initial pipeline for NyayaNet’s large language model development.

## Repository Structure

- **`datasets/`**: Contains processed or cleaned CSV files.
- **`custom_tokenizer/`**: Base tokenizer files (e.g., `vocab.txt`) pre-trained on Predex + High Court data + District court + Tribunal + Dailyorder
- **`custom_tokenizer_stage2/`**: Output of the fine-tuned tokenizer, including `training_texts_stage2.txt` and updated model files.
- **`notebooks/`**: Jupyter notebooks (e.g., `custom_final.ipynb`, `lora_conversion.ipynb`) detailing the workflow.
- **`.gitignore`**: Excludes large raw datasets, temporary files, and environment data.
- **`README.md`**: This file.

## Usage

### Prerequisites
- Python 3.9+
- Libraries: `pandas`, `tokenizers`, `tqdm`
- Install dependencies:
  ```bash
  pip install pandas tokenizers tqdm
