# workers.py
import os
import json
import torch
from tqdm import tqdm
from transformers import BertTokenizerFast

_TOKENIZERS = {}

def get_tokenizer(tokenizer_dir):
    """Load tokenizer once per process."""
    global _TOKENIZERS
    if tokenizer_dir not in _TOKENIZERS:
        _TOKENIZERS[tokenizer_dir] = BertTokenizerFast.from_pretrained(tokenizer_dir, local_files_only=True)
    return _TOKENIZERS[tokenizer_dir]

def tokenize_chunk(args):
    (
        tokenizer_dir,
        chunk_samples,
        gpu_id,
        max_length,
        stride,
        output_dir,
        run_id,
        batch_size,
    ) = args

    tokenizer = get_tokenizer(tokenizer_dir)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    use_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device(f"cuda:{gpu_id}" if use_gpu else "cpu")

    out_path = os.path.join(output_dir, f"tokenized_part_{run_id}_gpu{gpu_id}.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    written = 0
    with open(out_path, "w", encoding="utf-8") as fw:
        for sample in tqdm(chunk_samples, desc=f"GPU-{gpu_id}", position=gpu_id, leave=True):
            text = sample.get("input", "")
            label = sample.get("output", "")
            orig_index = sample.get("orig_index", None)
            court_type = sample.get("court_type", "DailyOrderCourt")

            # Tokenize text
            encoded = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            input_ids = encoded["input_ids"][0]
            chunks = []
            step = max_length - stride if (max_length - stride) > 0 else max_length

            for start in range(0, input_ids.size(0), step):
                end = min(start + max_length, input_ids.size(0))
                chunk_ids = input_ids[start:end]
                if (chunk_ids != pad_id).any():
                    chunks.append({
                        "token_ids": chunk_ids.cpu().numpy().tolist(),
                        "start_idx": int(start),
                        "end_idx": int(end),
                    })

            out_obj = {
                "orig_index": orig_index,
                "label": label,
                "court_type": court_type,
                "num_chunks": len(chunks),
                "chunks": chunks,
            }

            fw.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            written += 1

            if batch_size and (written % batch_size == 0):
                fw.flush()

        fw.flush()

    return {"gpu_id": gpu_id, "path": out_path, "count": written}
