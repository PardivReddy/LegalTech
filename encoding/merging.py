#!/usr/bin/env python3
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

chunk_path = "/home/infodna/Court-MOE/encoding/encoded_output_final/encoded_chunks_all.pth"
output_path = "/home/infodna/Court-MOE/encoding/encoded_output_final/aggregated_doc_embeddings_fixed.pth"

print(f"📂 Loading chunk-level embeddings from {chunk_path} ...")
chunks = torch.load(chunk_path, map_location="cpu")
print(f"✅ Loaded {len(chunks):,} chunk embeddings")


case_embeddings = defaultdict(list)
for rec in tqdm(chunks, desc="Grouping by case_id", ncols=100):
    case_embeddings[rec["case_id"]].append(rec)

aggregated = []
print("⚙️ Aggregating embeddings ...")
for case_id, recs in tqdm(case_embeddings.items(), total=len(case_embeddings), ncols=100):
    embs = [np.array(r["embeddings"], dtype=np.float32) for r in recs if len(r["embeddings"]) == 768]
    if not embs:
        continue
    mean_emb = np.mean(embs, axis=0)
    label = recs[0]["label"]
    court = recs[0]["court_type_idx"]
    aggregated.append({
        "case_id": case_id,
        "embeddings": mean_emb.tolist(),
        "label": label,
        "court_type_idx": court
    })

torch.save(aggregated, output_path)
print(f"✅ Saved {len(aggregated):,} aggregated records → {output_path}")
