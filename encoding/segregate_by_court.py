import torch
import os
import random
import json
from tqdm import tqdm
from collections import defaultdict, Counter

# =========================================================
# CONFIGURATION
# =========================================================
input_path = "encoded_output_custom/encoded_cases_final_resized.pth"
output_dir = "encoded_output_custom/final_balanced_by_court"
os.makedirs(output_dir, exist_ok=True)

# Redistribution plan (for unknown samples)
redistribution_plan = {
    "DistrictCourt": 10000,
    "TribunalCourt": 10000,
    "DailyOrderCourt": 15779
}

# Court label mapping (for router training)
court_label_map = {
    "SupremeCourt": 0,
    "HighCourt": 1,
    "DistrictCourt": 2,
    "TribunalCourt": 3,
    "DailyOrderCourt": 4
}

summary_path = os.path.join(output_dir, "segregation_balance_summary.json")

# =========================================================
# LOAD DATA
# =========================================================
print(f"📂 Loading encoded data from: {input_path}")
data = torch.load(input_path, map_location="cpu")
print(f"✅ Loaded {len(data):,} records")

# =========================================================
# SEGREGATE BY COURT TYPE
# =========================================================
court_groups = defaultdict(list)
unknown_records = []

for record in tqdm(data, desc="🔍 Segregating by court_type"):
    ctype = record.get("court_type", "Unknown")
    if ctype in court_label_map:
        court_groups[ctype].append(record)
    else:
        unknown_records.append(record)

# =========================================================
# BEFORE BALANCING SUMMARY
# =========================================================
print("\n📊 BEFORE BALANCING:")
for k in sorted(list(court_label_map.keys()) + ["Unknown"]):
    count = len(court_groups[k]) if k in court_groups else len(unknown_records)
    print(f"  {k:<18} → {count:,}")

# =========================================================
# REDISTRIBUTE UNKNOWN RECORDS
# =========================================================
print("\n⚖️ Redistributing Unknown samples...")
random.shuffle(unknown_records)

start = 0
for target, count in redistribution_plan.items():
    end = start + count
    subset = unknown_records[start:end]
    for r in subset:
        r["court_type"] = target
    court_groups[target].extend(subset)
    print(f"💾 Added {count:,} Unknown → {target}")
    start = end

remaining_unknowns = unknown_records[start:]
if remaining_unknowns:
    print(f"🗂️ Remaining unassigned Unknowns: {len(remaining_unknowns):,}")
    torch.save(remaining_unknowns, os.path.join(output_dir, "Unknown_unused.pth"))

# =========================================================
# ASSIGN NUMERIC LABELS (court_type_idx)
# =========================================================
print("\n🔢 Assigning numeric court_type_idx...")
for ctype, records in court_groups.items():
    for r in records:
        r["court_type_idx"] = court_label_map.get(r["court_type"], -1)

# =========================================================
# SAVE SEGREGATED + BALANCED FILES
# =========================================================
print("\n💾 Saving final balanced datasets...")
for ctype, records in court_groups.items():
    if not records:
        continue
    path = os.path.join(output_dir, f"{ctype}_embeddings_final.pth")
    torch.save(records, path)
    print(f"✅ Saved {len(records):,} → {path}")

# =========================================================
# AFTER SUMMARY
# =========================================================
after_counts = {ctype: len(records) for ctype, records in court_groups.items()}
after_counts["Unknown_unused"] = len(remaining_unknowns)

print("\n📊 AFTER BALANCING:")
for k, v in after_counts.items():
    print(f"  {k:<18} → {v:,}")

# =========================================================
# SAVE SUMMARY REPORT
# =========================================================
summary = {
    "before_counts": {k: len(v) if isinstance(v, list) else v for k, v in court_groups.items()},
    "unknown_total": len(unknown_records) + len(remaining_unknowns),
    "redistribution_plan": redistribution_plan,
    "after_counts": after_counts,
    "label_mapping": court_label_map
}

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print(f"\n📝 Summary report saved → {summary_path}")
print("\n✅ Segregation + Balancing + Label Assignment completed successfully.")
