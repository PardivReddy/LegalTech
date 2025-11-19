import os
import json
import torch
import numpy as np
import re
from tqdm import tqdm
import textstat

# ============================
# CONFIGURATION
# ============================
ENCODED_PATH = "encoding/encoded_output_final/encoded_cases_final_balanced_minority.pth"
LORA_PATH = "Datasets/dataset_multi_lora_reclassified_final.jsonl"
SAVE_PATH = "encoding/metadata_augmented_v3_district_daily.pth"
TARGET_COURTS = {
    2: "District Court",
    4: "Daily Order"
}
LEGAL_TERMS = [
    "section", "article", "act", "tribunal", "petition", "appeal", "jurisdiction",
    "ipc", "constitution", "crpc", "evidence", "high court", "supreme court",
    "district court", "respondent", "petitioner", "appellant", "bench", "justice",
    "order", "judgment", "writ", "suo motu", "contempt", "bail", "revision",
    "review", "cognizance", "sessions", "magistrate", "civil", "criminal",
    "advocate", "counsel", "plaintiff", "defendant", "prosecution", "defense"
]

# ============================
# UTILITY FUNCTIONS
# ============================
def safe_normalize(value, max_value, default=0.0):
    """Safely normalize value to [0, 1] range"""
    if max_value <= 0:
        return default
    return min(float(value) / max_value, 1.0)

# ============================
# METADATA EXTRACTION
# ============================
def extract_meta_vector(meta: dict, text: str) -> np.ndarray:
    if not isinstance(text, str) or len(text) == 0:
        return np.zeros(9, dtype=np.float32)
    
    text_lower = text.lower()
    meta_text = " ".join([f"{k}:{v}" for k, v in meta.items()]) if isinstance(meta, dict) else ""
    combined = f"{text_lower} {meta_text}"

    # [0] Document length (normalized, cap at 5000 chars)
    doc_len = safe_normalize(len(text), 5000)

    # [1] Has date patterns
    date_pattern = r"\b(date|dated|order\s+on|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"
    has_date = 1.0 if re.search(date_pattern, combined, re.I) else 0.0

    # [2] Has case number patterns
    case_pattern = r"\b(case\s*no\.?|appeal\s*no\.?|petition\s*no\.?|wp|crl|civil\s*appeal)\b"
    has_case = 1.0 if re.search(case_pattern, combined, re.I) else 0.0

    # [3] Has bench/judge references
    bench_pattern = r"\b(bench|coram|justice|judge|hon'?ble|hon'?ble\s+justice)\b"
    has_bench = 1.0 if re.search(bench_pattern, combined, re.I) else 0.0

    # [4] Has jurisdiction mentions
    juris_pattern = r"\b(jurisdiction|court\s+of|territorial\s+jurisdiction|tribunal|authority)\b"
    has_juris = 1.0 if re.search(juris_pattern, combined, re.I) else 0.0

    # [5] Number of judges (normalized, cap at 3)
    judge_pattern = r"justice\s+[A-Z][a-z]+"
    judge_matches = re.findall(judge_pattern, combined)
    num_judges = safe_normalize(len(set(judge_matches)), 3)

    # [6] Legal terminology ratio
    words = text_lower.split()
    if len(words) > 0:
        term_count = sum(1 for term in LEGAL_TERMS if term in text_lower)
        legal_ratio = safe_normalize(term_count, 20)
    else:
        legal_ratio = 0.0

    # [7] Readability score (Flesch Reading Ease)
    try:
        flesch = textstat.flesch_reading_ease(text)
        # Map Flesch (0-100+) to [0,1] where 0=complex, 1=simple
        readability = np.clip((flesch + 30) / 130.0, 0, 1)
    except Exception:
        readability = 0.5

    # [8] Placeholder for embedding norm (filled later)
    emb_norm_placeholder = 0.0

    return np.array([
        doc_len, has_date, has_case, has_bench, has_juris,
        num_judges, legal_ratio, readability, emb_norm_placeholder
    ], dtype=np.float32)

# ============================
# MAIN PROCESSING
# ============================
def main():
    print(f"\n{'='*70}")
    print(f"🚀 District/Daily Metadata Augmentation (Optimized)")
    print(f"{'='*70}")
    print(f"Target: District Court (idx=2) and Daily Order (idx=4) ONLY")
    print(f"Output: 777-dim embeddings (768 text + 9 metadata)")
    print(f"{'='*70}\n")

    # ============================
    # STEP 1: Load LoRA Dataset
    # ============================
    print(f"📂 Step 1/4: Loading LoRA dataset")
    print(f"   Source: {LORA_PATH}")
    
    if not os.path.exists(LORA_PATH):
        raise FileNotFoundError(f"LoRA dataset not found: {LORA_PATH}")
    
    lora_data = []
    skipped_lines = 0
    
    with open(LORA_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    lora_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    skipped_lines += 1
                    if skipped_lines <= 5:
                        print(f"   ⚠  Line {line_num}: Malformed JSON")
    
    print(f"   ✅ Loaded {len(lora_data):,} samples")
    if skipped_lines > 0:
        print(f"   ⚠  Skipped {skipped_lines} malformed lines")
    print()

    # ============================
    # STEP 2: Generate Metadata Vectors
    # ============================
    print(f"⚙  Step 2/4: Extracting metadata features")
    
    meta_lookup = {}
    failed_extractions = 0
    
    for i, d in enumerate(tqdm(lora_data, desc="   Processing")):
        try:
            meta_vec = extract_meta_vector(
                d.get("metadata", {}),
                d.get("input", "")
            )
            meta_lookup[i] = meta_vec
        except Exception as e:
            failed_extractions += 1
            if failed_extractions <= 3:
                print(f"   ⚠  Sample {i}: Extraction failed - {e}")
            meta_lookup[i] = np.zeros(9, dtype=np.float32)
    
    print(f"   ✅ Generated {len(meta_lookup):,} metadata vectors")
    if failed_extractions > 0:
        print(f"   ⚠  {failed_extractions} failed (using zero vectors)")
    print()

    # ============================
    # STEP 3: Load Encoded Embeddings
    # ============================
    print(f"📂 Step 3/4: Loading encoded embeddings")
    print(f"   Source: {ENCODED_PATH}")
    
    if not os.path.exists(ENCODED_PATH):
        raise FileNotFoundError(f"Encoded embeddings not found: {ENCODED_PATH}")
    
    records = torch.load(ENCODED_PATH, map_location="cpu", weights_only=False)
    print(f"   ✅ Loaded {len(records):,} embeddings (all courts)")
    
    # Validate first embedding
    sample_emb = records[0]["embeddings"]
    if isinstance(sample_emb, torch.Tensor):
        sample_emb = sample_emb.detach().cpu().numpy()
    print(f"   📏 Base embedding dimension: {len(sample_emb)}")
    
    if len(sample_emb) != 768:
        print(f"   ⚠  Warning: Expected 768-dim, got {len(sample_emb)}-dim")
    print()

    # ============================
    # STEP 4: Augment DISTRICT + DAILY ONLY
    # ============================
    print(f"⚙  Step 4/4: Filtering and augmenting District/Daily ONLY")
    print(f"   • District Court (idx=2) → 777-dim ✨")
    print(f"   • Daily Order (idx=4)    → 777-dim ✨")
    print(f"   • All others             → SKIPPED")
    print()

    augmented = []
    stats = {
        "district": 0,
        "daily": 0,
        "skipped_other_courts": 0,
        "skipped_wrong_dim": 0,
        "skipped_no_metadata": 0
    }

    for i, record in enumerate(tqdm(records, desc="   Filtering & Augmenting")):
        # Get court type FIRST to filter early
        ctype = int(record.get("court_type_idx", -1))
        
        # 🔥 CRITICAL: Only process District (2) and Daily (4)
        if ctype not in [2, 4]:
            stats["skipped_other_courts"] += 1
            continue  # Skip Supreme/High/Tribunal entirely
        
        # Extract embedding
        emb = record["embeddings"]
        
        # Convert to NumPy float32
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy().astype(np.float32)
        else:
            emb = np.asarray(emb, dtype=np.float32)
        
        # Validate embedding dimension
        if len(emb) != 768:
            stats["skipped_wrong_dim"] += 1
            if stats["skipped_wrong_dim"] <= 3:
                print(f"   ⚠  Sample {i}: dimension {len(emb)}, expected 768")
            continue
        
        # Compute embedding L2 norm (scaled to [0,1])
        emb_norm = float(np.linalg.norm(emb))
        emb_norm_scaled = np.clip(emb_norm / 100.0, 0, 1)
        
        # Get metadata vector
        if i in meta_lookup:
            meta_vec = meta_lookup[i].copy()
        else:
            meta_vec = np.zeros(9, dtype=np.float32)
            stats["skipped_no_metadata"] += 1
        
        # Fill in the embedding norm (last feature)
        meta_vec[8] = emb_norm_scaled
        
        # Augment to 777-dim (768 text + 9 metadata)
        new_emb = np.concatenate([emb, meta_vec])
        
        # Validate output dimension
        assert len(new_emb) == 777, f"Expected 777-dim, got {len(new_emb)}"
        
        # Update record
        record["embeddings"] = new_emb
        
        # Track statistics
        if ctype == 2:
            stats["district"] += 1
        elif ctype == 4:
            stats["daily"] += 1
        
        augmented.append(record)

    # ============================
    # VALIDATION & STATISTICS
    # ============================
    print(f"\n{'='*70}")
    print(f"📊 Processing Statistics")
    print(f"{'='*70}\n")
    print(f"{'Category':<30} {'Count':<15} {'Status':<10}")
    print(f"{'-'*55}")
    print(f"{'District Court (2)':<30} {stats['district']:<15,} {'✅ Kept':<10}")
    print(f"{'Daily Order (4)':<30} {stats['daily']:<15,} {'✅ Kept':<10}")
    print(f"{'Other courts (0,1,3)':<30} {stats['skipped_other_courts']:<15,} {'⏭  Skipped':<10}")
    if stats['skipped_wrong_dim'] > 0:
        print(f"{'Wrong dimension':<30} {stats['skipped_wrong_dim']:<15,} {'⚠  Skipped':<10}")
    if stats['skipped_no_metadata'] > 0:
        print(f"{'Missing metadata':<30} {stats['skipped_no_metadata']:<15,} {'⚠  Warning':<10}")
    print(f"{'-'*55}")
    print(f"{'TOTAL OUTPUT':<30} {len(augmented):<15,} {'✅':<10}")
    print()

    # Dimension validation
    print(f"🔍 Dimension Validation:")
    all_dims = [len(r["embeddings"]) for r in augmented]
    unique_dims = set(all_dims)
    
    if unique_dims == {777}:
        print(f"   ✅ All samples: 777-dim (PERFECT!)")
    else:
        print(f"   ❌ ERROR: Found dimensions {unique_dims}, expected {777}")
        raise ValueError("Dimension mismatch detected!")
    print()
    print(f"📋 Court Type Breakdown:")
    for court_idx, court_name in TARGET_COURTS.items():
        count = sum(1 for r in augmented if r.get("court_type_idx") == court_idx)
        pct = 100 * count / len(augmented) if len(augmented) > 0 else 0
        print(f"   • {court_name:<20} {count:>6,} samples ({pct:>5.1f}%)")
    print()

    # ============================
    # SAVE AUGMENTED DATASET
    # ============================
    print(f"💾 Saving augmented dataset...")
    print(f"   Destination: {SAVE_PATH}")
    
    # Create directory if needed
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    
    # Save
    torch.save(augmented, SAVE_PATH)
    
    file_size_mb = os.path.getsize(SAVE_PATH) / (1024 * 1024)
    print(f"   ✅ Saved successfully ({file_size_mb:.1f} MB)")

    # ============================
    # FINAL SUMMARY
    # ============================
    print(f"\n{'='*70}")
    print(f"✅ AUGMENTATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📦 Dataset Summary:")
    print(f"   • Total samples: {len(augmented):,}")
    print(f"   • District Court: {stats['district']:,} samples")
    print(f"   • Daily Order: {stats['daily']:,} samples")
    print(f"   • Dimension: 777 (768 text + 9 metadata)")
    print(f"\n🎯 Purpose:")
    print(f"   The 9 metadata features help distinguish District vs Daily cases")
    print(f"   which have similar textual content but different procedural")
    print(f"   characteristics (bench size, document length, legal complexity).")
    print(f"\n📁 Output Location:")
    print(f"   {os.path.abspath(SAVE_PATH)}")
    print(f"\n💡 Note:")
    print(f"   Supreme/High/Tribunal courts are NOT in this file.")
    print(f"   They are loaded from separate files during training (768-dim).")
    print(f"\n{'='*70}\n")

    # ============================
    # METADATA STATISTICS (BONUS)
    # ============================
    print(f"📊 Metadata Feature Statistics (Sample of 100):\n")
    
    # Sample 100 random records for statistics
    sample_size = min(100, len(augmented))
    sample_indices = np.random.choice(len(augmented), sample_size, replace=False)
    
    feature_names = [
        "doc_length", "has_date", "has_case_num", "has_bench",
        "is_jurisdiction", "num_judges", "legal_ratio", "readability", "emb_norm"
    ]
    
    print(f"{'Feature':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print(f"{'-'*60}")
    
    for feat_idx, feat_name in enumerate(feature_names):
        values = [augmented[i]["embeddings"][768 + feat_idx] for i in sample_indices]
        values = np.array(values)
        
        print(f"{feat_name:<20} {values.mean():<10.3f} {values.std():<10.3f} "
              f"{values.min():<10.3f} {values.max():<10.3f}")
    
    print(f"\n{'='*70}\n")

if __name__== "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠  Process interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()