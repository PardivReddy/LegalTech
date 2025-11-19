import os, json, torch, numpy as np
from tqdm import tqdm

# ============================
# CONFIG
# ============================
encoded_path = "encoding/encoded_output_best/encoded_cases_final_balanced_minority.pth"
lora_path = "Datasets/dataset_multi_lora_reclassified_final.jsonl"
save_path = "encoding/metadata_augmented_district_daily.pth"

# ============================
# LOAD LORA DATASET
# ============================
print(f"🔍 Loading LoRA dataset from: {lora_path}")
lora_data = []
with open(lora_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            lora_data.append(json.loads(line))

print(f"✅ Loaded {len(lora_data):,} LoRA samples")

# ============================
# BUILD TEXT-BASED METADATA FEATURES
# ============================
def extract_meta_vector(meta: dict, text: str):
    """
    Generates a 5-dim metadata vector:
    [doc_length, has_date, has_case_num, has_bench, is_jurisdiction_present]
    """
    text = text.lower()
    meta_text = " ".join([f"{k}:{v}" for k, v in meta.items()]) if isinstance(meta, dict) else ""
    combined = f"{text} {meta_text}"

    doc_len = min(len(text) / 5000, 1.0)
    has_date = 1.0 if any(w in combined for w in ["dated", "date", "order on"]) else 0.0
    has_case = 1.0 if any(w in combined for w in ["case no", "appeal no", "petition"]) else 0.0
    has_bench = 1.0 if any(w in combined for w in ["bench", "coram", "justice", "judge"]) else 0.0
    has_juris = 1.0 if any(w in combined for w in ["jurisdiction", "court of", "tribunal"]) else 0.0

    return np.array([doc_len, has_date, has_case, has_bench, has_juris], dtype=np.float32)

print("⚙️ Generating metadata vectors for LoRA records...")
meta_vectors = [extract_meta_vector(d.get("metadata", {}), d.get("input", "")) for d in tqdm(lora_data)]

# ============================
# LOAD ENCODED EMBEDDINGS
# ============================
print(f"\n🔍 Loading encoded embeddings from: {encoded_path}")
records = torch.load(encoded_path, map_location="cpu")
print(f"✅ Loaded {len(records):,} embeddings")

# ============================
# AUGMENT DISTRICT + DAILY ONLY
# ============================
print("\n⚙️ Attaching metadata to District Court (2) and Daily Order (4)...")
augmented = []
for i, r in enumerate(tqdm(records)):
    emb = np.asarray(r["embeddings"], np.float32)
    ctype = int(r["court_type_idx"])

    if ctype in [2, 4]:  
        meta_vec = meta_vectors[i] if i < len(meta_vectors) else np.zeros(5, np.float32)
        new_emb = np.concatenate([emb, meta_vec])
    else:
        new_emb = np.concatenate([emb, np.zeros(5, np.float32)])

    r["embeddings"] = new_emb
    augmented.append(r)

# ============================
# SAVE
# ============================
torch.save(augmented, save_path)
print(f"\n💾 Saved augmented dataset → {save_path}")
print(f"📏 Original dim: 768 | Augmented dim: {len(augmented[0]['embeddings'])}")
print("✅ Done! District & Daily embeddings now include metadata features.")
