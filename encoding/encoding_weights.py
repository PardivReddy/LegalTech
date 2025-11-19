import os
import json
import logging
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.cuda import amp

# =============================
# CONFIGURATION
# =============================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("encode_casewise_resized")

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =============================
# FUNCTION
# =============================
def encode_tokenized_output_casewise(
    tokenized_jsonl: str,
    out_dir: str,
    model_path: str,
    tokenizer_path: str,
    batch_size: int = 4,
    max_length: int = 512,
    max_samples: int | None = None,
):
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"🧠 Using device: {device}")

    # Load model + tokenizer
    logger.info("📦 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModel.from_pretrained(model_path)

    # --------------------------
    # Detect vocab mismatch
    # --------------------------
    model_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)

    logger.info(f"🔡 Tokenizer vocab size: {tokenizer_vocab_size}")
    logger.info(f"🔡 Model vocab size: {model_vocab_size}")

    if tokenizer_vocab_size != model_vocab_size:
        logger.warning("⚠️ Mismatch detected! Resizing encoder embeddings to match tokenizer vocab.")
        model.resize_token_embeddings(tokenizer_vocab_size)
        resized_path = os.path.join(model_path, "_resized_auto")
        os.makedirs(resized_path, exist_ok=True)
        model.save_pretrained(resized_path)
        tokenizer.save_pretrained(resized_path)
        logger.info(f"✅ Saved resized model+tokenizer → {resized_path}")

    model.to(device)
    model.eval()

    # --------------------------
    # Load and fix tokenized data
    # --------------------------
    logger.info("📥 Loading and fixing tokenized dataset...")
    valid_cases = []
    with open(tokenized_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="🧩 Fixing tokenization", ncols=100):
        try:
            data = json.loads(line)
            chunks = data.get("chunks") or data.get("tokenized_chunks") or []
            if not chunks:
                continue

            fixed_chunks = []
            for ch in chunks:
                input_ids = ch.get("input_ids") or ch.get("token_ids")
                if not isinstance(input_ids, list) or len(input_ids) == 0:
                    continue
                # ✅ Clamp to vocab range
                input_ids = [int(i) for i in input_ids if 0 <= int(i) < tokenizer_vocab_size]
                if len(input_ids) == 0:
                    continue
                # Pad/truncate
                input_ids = input_ids[:max_length]
                attention_mask = [1] * len(input_ids)
                if len(input_ids) < max_length:
                    pad_len = max_length - len(input_ids)
                    input_ids += [tokenizer.pad_token_id] * pad_len
                    attention_mask += [0] * pad_len
                fixed_chunks.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                })

            if not fixed_chunks:
                continue

            valid_cases.append({
                "chunks": fixed_chunks,
                "label": data.get("label", "Rejected"),
                "court_type": data.get("court_type", "DailyOrderCourt"),
                "case_id": data.get("case_id") or f"case_{len(valid_cases)}"
            })

            if max_samples and len(valid_cases) >= max_samples:
                break

        except Exception as e:
            logger.warning(f"⚠️ Skipping malformed line: {e}")
            continue

    logger.info(f"✅ Loaded {len(valid_cases):,} valid cases")

    # --------------------------
    # Prepare output
    # --------------------------
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "encoded_cases_final_resized.pth")
    encoded_records = []
    skipped = 0

    # --------------------------
    # Encoding loop
    # --------------------------
    progress = tqdm(valid_cases, total=len(valid_cases), desc="⚙️ Encoding (casewise)", ncols=100)
    for idx, case in enumerate(progress):
        try:
            cls_list = []
            for chunk in case["chunks"]:
                input_ids = torch.tensor([chunk["input_ids"]], dtype=torch.long, device=device)
                attention_mask = torch.tensor([chunk["attention_mask"]], dtype=torch.long, device=device)

                with torch.no_grad(), amp.autocast(enabled=True):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    cls_emb = outputs.last_hidden_state[:, 0, :].detach().cpu()
                    cls_list.append(cls_emb)

            if cls_list:
                case_embedding = torch.mean(torch.cat(cls_list, dim=0), dim=0)
                encoded_records.append({
                    "embeddings": case_embedding.tolist(),
                    "label": case["label"],
                    "court_type": case["court_type"],
                    "case_id": case["case_id"]
                })

            if (idx + 1) % 1000 == 0:
                progress.set_postfix({"saved": len(encoded_records)})
            if (idx + 1) % 5000 == 0:
                torch.save(encoded_records, out_path)
                logger.info(f"💾 Checkpoint updated ({len(encoded_records)} cases)")

        except Exception as e:
            skipped += 1
            logger.warning(f"⚠️ Skipping case {case['case_id']} — {e}")
            continue

    # --------------------------
    # Final save
    # --------------------------
    torch.save(encoded_records, out_path)
    logger.info(f"✅ Saved {len(encoded_records):,} encoded cases → {out_path}")
    logger.info(f"📦 File size: {os.path.getsize(out_path)/1e6:.2f} MB")
    logger.info(f"⚠️ Total skipped cases: {skipped}")

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Encode casewise using fine-tuned LegalBERT with auto-resize")
    parser.add_argument("--tokenized_jsonl", required=True, help="Path to tokenized .jsonl file")
    parser.add_argument("--model", required=True, help="Path to fine-tuned encoder model")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer")
    parser.add_argument("--out_dir", required=True, help="Output directory for encodings")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    encode_tokenized_output_casewise(
        tokenized_jsonl=args.tokenized_jsonl,
        out_dir=args.out_dir,
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_samples=args.max_samples
    )
