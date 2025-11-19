# tokenize.py
import os
import json
import time
import subprocess
from torch.multiprocessing import Pool, set_start_method
import torch

# Import worker tokenizer
from workers import tokenize_chunk

print("Script started")

# ---------------- Config ----------------
base_dir = os.path.abspath("/home/infodna/Court-MOE")  # project root
tokenizer_dir = os.path.join(base_dir, "final_tokenizer")   # pretrained tokenizer folder
lora_dataset_path = os.path.join(base_dir, "Datasets", "dataset_multi_lora_reclassified_final.jsonl")
output_dir = os.path.join(base_dir, "tokenization", "Tokenized_output_pakka")
os.makedirs(output_dir, exist_ok=True)

max_length = 512
stride = 128
batch_write = 1000   # flush frequency

# ---------------- GPU Selection ----------------
def get_available_gpus(skip_gpu0=True):
    """Return a list of GPU indices that show 0% utilization."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            encoding="utf-8",
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        selected = []
        for line in lines:
            idx, util = line.split(",")
            idx = idx.strip()
            util = int(util.strip())
            if util == 0:
                if skip_gpu0 and idx == "0":
                    continue
                selected.append(idx)
        return selected
    except Exception as e:
        print(f"⚠ GPU detection failed: {e}")
        return []

# Select GPUs (skip 0 if desired)
selected_gpus = get_available_gpus(skip_gpu0=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpus) if selected_gpus else ""
num_gpus = torch.cuda.device_count()
print(f"🔑 Selected GPUs (physical indices): {selected_gpus}")
print(f"🔑 CUDA_VISIBLE_DEVICES => {os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
print(f"🔑 Using {num_gpus} visible device(s) as indices 0..{max(0, num_gpus-1)}")

# ---------------- Load dataset ----------------
if __name__ == "__main__":
    set_start_method("spawn", force=True)
    run_id = str(int(time.time()))
    print("Starting main process")

    try:
        print(f"📥 Loading dataset from {lora_dataset_path}")
        valid = []
        with open(lora_dataset_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    if i % 1000 == 0:
                        print(f"⚠ malformed JSON at line {i}; skipping")
                    continue
                valid.append(obj)

        print(f"📊 Loaded dataset: {len(valid):,} records")

        # --- Build samples for tokenization ---
        samples = []
        for idx, obj in enumerate(valid):
            meta = obj.get("metadata", {})
            court_type = meta.get("court_type", "DailyOrderCourt")

            # Combine LoRA fields into one text string
            instruction = obj.get("instruction", "").strip()
            input_text = obj.get("input", "").strip()
            output_text = obj.get("output", "").strip()

            # Full text for tokenization
            full_text = "\n\n".join([x for x in [instruction, input_text, output_text] if x])

            samples.append({
                "input": full_text,
                "output": output_text,
                "orig_index": idx,
                "court_type": court_type
            })

        # --- Worker setup ---
        num_workers = num_gpus if num_gpus > 0 else 1
        print(f"👉 Using {num_workers} worker(s)")

        def split_into_chunks(items, n):
            total = len(items)
            size = total // n
            return [items[i*size:(i+1)*size if i < n-1 else total] for i in range(n)]

        sample_chunks = split_into_chunks(samples, num_workers)
        print(f"🚀 Starting tokenization across {num_workers} workers...")

        tokenizer_dir_abs = os.path.abspath(tokenizer_dir)
        worker_args = []
        for i in range(num_workers):
            gpu_id = i % num_gpus if num_gpus > 0 else 0
            worker_args.append(
                (
                    tokenizer_dir_abs,
                    sample_chunks[i],
                    gpu_id,
                    max_length,
                    stride,
                    output_dir,
                    run_id,
                    batch_write,
                )
            )

        with Pool(processes=num_workers) as pool:
            results = pool.map(tokenize_chunk, worker_args)

        total_written = sum(r.get("count", 0) for r in results)
        print(f"✅ Workers finished. Total samples written: {total_written}")

        # --- Merge partial outputs ---
        final_path = os.path.join(output_dir, f"tokenized_multi_fixed_{run_id}.jsonl")
        with open(final_path, "w", encoding="utf-8") as fout:
            for r in sorted(results, key=lambda x: x["gpu_id"]):
                with open(r["path"], "r", encoding="utf-8") as fr:
                    for line in fr:
                        fout.write(line)

        print(f"📄 Merged tokenized output -> {final_path}")

    except FileNotFoundError:
        print(f"⚠ File not found: {lora_dataset_path}")
    except Exception as e:
        print(f"⚠ Error: {e}")
