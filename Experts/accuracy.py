#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Court-MOE — Accuracy Evaluation Script
--------------------------------------
Evaluates router + expert ensembles on a CSV test set.

Features:
✓ Handles very large text fields safely
✓ Auto-detects tokenizer vocab vs encoder mismatch
✓ Resizes encoder embeddings in memory only
✓ Uses your fine-tuned LegalBERT encoder + tokenizer
✓ Evaluates router + expert predictions (accept/reject)
✓ Computes Accuracy, Precision, Recall, F1, and AUC
"""

import os, sys, csv, json, numpy as np, pandas as pd, torch, torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score

# ---------------------------------------------------------------------
# FIX: Allow very long CSV fields
# ---------------------------------------------------------------------
csv.field_size_limit(sys.maxsize)

# ---------------------------------------------------------------------
# PATH CONFIGS
# ---------------------------------------------------------------------
TOKENIZER_DIR = "tokenization/final_tokenizer"
ENCODER_DIR   = "encoding/legalbert_finetuned_courts"
ROUTER_CKPT   = "routers/router_meta_boosted_61.64/best_router.pt"
EXP_BASE      = "Experts/experts_kfold"
TEST_CSV      = "CJPE_ext_SCI_HCs_Tribunals_daily_orders_test.csv"
OUT_DIR       = "results"

# ---------------------------------------------------------------------
# MODEL DEFINITIONS
# ---------------------------------------------------------------------
class RouterMLP(nn.Module):
    def __init__(self, in_dim=773, hidden=256, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x): return self.net(x)

class ExpertMLP(nn.Module):
    def __init__(self, in_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)

# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------
def _infer_in_dim(sd):
    for k, v in sd.items():
        if "weight" in k and v.ndim == 2:
            return v.shape[1]
    return 768

def _pad_to_dim(x, want):
    have = x.shape[1]
    if have == want:
        return x
    pad = want - have
    return torch.cat([x, torch.zeros(x.size(0), pad, device=x.device)], dim=1)

def _ensemble_predict(models, x):
    probs = [torch.sigmoid(m(x)).squeeze(-1) for m in models]
    return torch.stack(probs, dim=0).mean(0)

def load_experts(court, device):
    base = os.path.join(EXP_BASE, court)
    folds = [os.path.join(base, f"{court}_fold{i}.pt") for i in range(1, 4)]
    models, in_dim = [], None
    for f in folds:
        sd = torch.load(f, map_location="cpu")
        in_dim = _infer_in_dim(sd)
        m = ExpertMLP(in_dim)
        try:
            m.load_state_dict(sd, strict=False)
        except:
            if "model" in sd: m.load_state_dict(sd["model"], strict=False)
        m.to(device).eval()
        models.append(m)
    return models, in_dim

# ---------------------------------------------------------------------
# TOKENIZER + ENCODER WITH AUTO-RESIZE
# ---------------------------------------------------------------------
def load_tokenizer_encoder():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    encoder   = AutoModel.from_pretrained(ENCODER_DIR, local_files_only=True)

    print("🔍 Scanning dataset to detect highest token ID...")
    max_id_seen = 0
    with open(TEST_CSV, "r", errors="ignore") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = (row.get("text") or "").strip()
            if not text: continue
            toks = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
            max_id_seen = max(max_id_seen, int(toks["input_ids"].max()))
            if i % 500 == 0:
                print(f"  → scanned {i} lines, max_id so far = {max_id_seen}")
            if i > 5000: break  # enough sampling

    tok_vocab = tokenizer.vocab_size
    enc_vocab = encoder.get_input_embeddings().weight.size(0)
    required_vocab = max(tok_vocab, max_id_seen + 1)
    print(f"🧩 Tokenizer vocab={tok_vocab}, Encoder emb={enc_vocab}, Max token ID={max_id_seen}")

    if required_vocab > enc_vocab:
        print(f"⚙️ Resizing encoder embeddings ({enc_vocab} → {required_vocab}) in memory...")
        old_emb = encoder.get_input_embeddings().weight.data.clone()
        encoder.resize_token_embeddings(required_vocab)
        new_emb = encoder.get_input_embeddings().weight.data
        diff = new_emb.shape[0] - old_emb.shape[0]
        new_emb[-diff:] = old_emb.mean() + 0.02 * torch.randn(diff, new_emb.shape[1])
        print(f"✅ Encoder temporarily resized: {tuple(new_emb.shape)}")
    else:
        print(f"✅ Encoder already covers all tokens (≤ {max_id_seen})")

    encoder.eval()
    return tokenizer, encoder

# ---------------------------------------------------------------------
# TEXT ENCODING
# ---------------------------------------------------------------------
@torch.inference_mode()
def encode_texts(tokenizer, encoder, texts, device):
    embeddings = []
    for i in tqdm(range(0, len(texts), 8), desc="🔤 Encoding"):
        batch = texts[i:i+8]
        tok = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        out = encoder(**tok).last_hidden_state[:, 0, :].cpu()
        embeddings.append(out)
    return torch.cat(embeddings, dim=0)

# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # === Load dataset ===
    df = pd.read_csv(TEST_CSV, engine="python", on_bad_lines="warn")
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    # === Tokenizer + Encoder ===
    tokenizer, encoder = load_tokenizer_encoder()
    X = encode_texts(tokenizer, encoder, texts, device)
    print(f"✅ Encoded {len(X)} samples, shape = {tuple(X.shape)}")

    # === Router ===
    router_sd = torch.load(ROUTER_CKPT, map_location="cpu")
    router_in_dim = _infer_in_dim(router_sd)
    router = RouterMLP(in_dim=router_in_dim).to(device)
    router.load_state_dict(router_sd, strict=False)
    router.eval()
    print(f"✅ Loaded router ({router_in_dim}-dim)")

    # === Experts ===
    experts = {}
    for court in ["supreme", "high", "tribunal", "district", "daily"]:
        experts[court] = load_experts(court, device)

    # === Router Predictions ===
    with torch.inference_mode():
        logits = router(X.to(device))
        court_ids = torch.argmax(logits, dim=1).cpu().numpy()
    id2court = {0: "supreme", 1: "high", 2: "tribunal", 3: "district", 4: "daily"}

    # === Expert Predictions ===
    preds, probs = [], []
    for i, emb in enumerate(tqdm(X, desc="⚖️ Experts")):
        cname = id2court[court_ids[i]]
        models, want_dim = experts[cname]
        x = _pad_to_dim(emb.unsqueeze(0).to(device), want_dim)
        p = float(_ensemble_predict(models, x))
        preds.append(1 if p >= 0.5 else 0)
        probs.append(p)

    # === Metrics ===
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0

    print("\n📊 Final Evaluation Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame({
        "text": texts, "true": labels, "pred": preds, "prob": probs
    }).to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}, f, indent=2)
    print(f"💾 Results saved to {OUT_DIR}/test_predictions.csv and metrics.json")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
