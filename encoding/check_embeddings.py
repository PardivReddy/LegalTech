import torch
from tqdm import tqdm
import os

# === Path to your aggregated embeddings ===
AGGREGATED_PATH = "/home/infodna/Court-MOE/encoding/encoded_output_final/aggregated_doc_embeddings_fixed.pth"

print(f"📂 Loading aggregated embeddings from: {AGGREGATED_PATH}")

# === Load data ===
data = torch.load(AGGREGATED_PATH, map_location="cpu")
print(f"✅ Loaded {len(data):,} aggregated records")

# === Verify embedding dimensions ===
invalid = []
for i, rec in enumerate(tqdm(data, desc="Checking dimensions", ncols=100)):
    emb = rec.get("embeddings", [])
    if len(emb) != 768:
        invalid.append((i, rec.get("case_id", f"unknown_{i}"), len(emb)))

# === Summary ===
if len(invalid) == 0:
    print("\n✅ All embeddings have correct length: 768")
else:
    print(f"\n⚠️ Found {len(invalid):,} records with incorrect embedding size:")
    for i, case_id, dim in invalid[:10]:
        print(f"   - Record {i} ({case_id}) → {dim} dims")
    report_path = os.path.join(os.path.dirname(AGGREGATED_PATH), "dimension_mismatch_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        for i, case_id, dim in invalid:
            f.write(f"{i}\t{case_id}\t{dim}\n")
    print(f"📄 Full mismatch list saved to: {report_path}")