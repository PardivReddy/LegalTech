# LegalTech – NyayaNet Preprocessing & Tokenization

LegalTech is a machine learning–driven legal document intelligence project.  
This repository focuses on the preprocessing and tokenizer pipeline for NyayaNet, a Mixture of Experts (MoE) system designed to analyze Indian legal judgments across courts.

## Overview

The goal of LegalTech is to advance Indian legal NLP by developing a custom preprocessing and tokenizer framework tailored to judicial text from:
- Supreme Court
- High Courts
- District Courts
- Tribunals
- Daily Orders

This repository captures the end-to-end workflow, from raw dataset handling to tokenizer training, and forms the foundation for downstream expert models and routing logic.

## Key Components

- Preprocessing  
  Cleaning, normalizing, and structuring raw legal datasets.

- LoRA Conversion  
  Transforming datasets into LoRA-compatible formats for downstream LLM training.

- Tokenizer Training  
  Fine-tuning a BERT WordPiece tokenizer on Indian legal text to capture domain-specific vocabulary.

- Foundation Pipeline  
  Serves as the initial pipeline for NyayaNet’s MoE-based legal AI system.

## Repository Structure

- `Court-MOE/` — Core MoE-related scripts and experiments  
- `Experts/` — Expert models and training logic  
- `encoding/` — Document encoding and embedding pipelines  
- `routers/` — Expert routing and selection logic  
- `tokenization/` — Tokenizer training and inference utilities  
- `extra/` — Experimental notebooks and utilities  

## Notes

- Trained models, embeddings, and datasets are intentionally excluded.
- This repository contains source code only, ensuring a clean and reproducible setup.

## Author

Pardiv Reddy
