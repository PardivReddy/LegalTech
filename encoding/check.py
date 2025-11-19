#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_court_type_idx.py
---------------------------------
Safely load .pth dataset file in PyTorch ≥2.6
and verify if 'court_type_idx' exists and counts by type.
"""

import torch
from collections import Counter
from numpy._core import multiarray

# ✅ Allow NumPy array reconstruction (required in PyTorch ≥2.6)
torch.serialization.add_safe_globals([multiarray._reconstruct, dict, list])

# ============================
# CONFIG
# ============================
PATH = "metadata_augmented_v2_district_daily.pth"  # adjust if needed

# ============================
# MAIN CHECK
# ============================
def main():
    print(f"🔍 Loading dataset from: {PATH}")
    try:
        data = torch.load(PATH, map_location="cpu", weights_only=False)
        print(f"✅ Loaded {len(data):,} records")
    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        return

    court_ids = [d.get("court_type_idx") for d in data if isinstance(d, dict) and "court_type_idx" in d]

    if not court_ids:
        print("⚠️ No 'court_type_idx' field found in any record.")
        return

    unique_ids = sorted(set(court_ids))
    counts = Counter(court_ids)

    print(f"\n🔹 Unique court_type_idx values: {unique_ids}")
    print(f"🔹 Count by type: {counts}")

    print("\n🧾 Example record keys:")
    for k in list(data[0].keys())[:10]:
        print(f"  - {k}")

    print("\n✅ Done! Check if District (2) and Daily (4) are present.")

if __name__ == "__main__":
    main()
