#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Court-MOE — Detailed Parameter Breakdown
Shows parameter count per major component + memory estimate
"""

import torch
import torch.nn as nn
from transformers import AutoModel
import pandas as pd

# ============================================================
# Define same model pieces you use
# ============================================================
class SEBlock(nn.Module):
    def __init__(self, dim, reduction=32):
        super().__init__()
        red = max(8, dim // reduction)
        self.fc1, self.fc2 = nn.Linear(dim, red), nn.Linear(red, dim)
        self.act, self.gate = nn.GELU(), nn.Sigmoid()
    def forward(self, x):
        w = self.gate(self.fc2(self.act(self.fc1(x))))
        return x * w

class SEResidualMLP(nn.Module):
    def __init__(self, dim=768, hidden=2048, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act, self.drop = nn.GELU(), nn.Dropout(dropout)
        self.se, self.norm = SEBlock(dim), nn.LayerNorm(dim)
    def forward(self, x):
        res = x
        x = self.drop(self.act(self.bn1(self.fc1(x))))
        x = self.drop(self.fc2(x))
        x = self.se(x)
        return self.norm(x + res)

class ExpertOnEmbeddings(nn.Module):
    def __init__(self, emb_dim=768, hidden=2048, dropout=0.30):
        super().__init__()
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.semlp = SEResidualMLP(emb_dim, hidden, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim,512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512,1)
        )
    def forward(self, x): return self.classifier(self.semlp(self.proj(x))).squeeze(-1)

class RouterModel(nn.Module):
    def __init__(self, input_dim=768, hidden=1024, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x): return self.net(x)

# ============================================================
# Instantiate your full architecture
# ============================================================
legalbert = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
router = RouterModel()

expert_cfgs = {
    "supreme": 1536, "high": 2304, "tribunal": 1920,
    "district": 2400, "daily": 2560
}
experts = {k: ExpertOnEmbeddings(hidden=h) for k,h in expert_cfgs.items()}

# ============================================================
# Helper functions
# ============================================================
def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

def module_table(name, module):
    rows = []
    for n, m in module.named_modules():
        if isinstance(m, (nn.Linear, nn.LayerNorm, nn.Embedding, nn.Conv1d)):
            params = sum(p.numel() for p in m.parameters())
            rows.append((f"{name}:{n}", params))
    return rows

# ============================================================
# Gather parameter stats
# ============================================================
rows = []

# --- LegalBERT ---
bert_total, _ = count_params(legalbert)
rows.append(("LegalBERT (total)", bert_total))

# Break down LegalBERT into embeddings + encoder blocks
emb_params = sum(p.numel() for p in legalbert.embeddings.parameters())
encoder_layers = [sum(p.numel() for p in blk.parameters()) for blk in legalbert.encoder.layer]
rows.append((" ├── Embeddings", emb_params))
for i, n in enumerate(encoder_layers):
    rows.append((f" ├── EncoderLayer_{i:02d}", n))

# --- Router ---
r_total, _ = count_params(router)
rows.append(("Router (total)", r_total))

# --- Experts ---
for k, exp in experts.items():
    total, _ = count_params(exp)
    rows.append((f"Expert_{k}", total))

# ============================================================
# Create nice summary table
# ============================================================
df = pd.DataFrame(rows, columns=["Module", "Parameters"])
df["Parameters (M)"] = (df["Parameters"]/1e6).round(3)
df["Memory (MB, FP32)"] = (df["Parameters"]*4/1e6).round(2)
df.loc[len(df)] = ["TOTAL", df["Parameters"].sum(), df["Parameters (M)"].sum(), df["Memory (MB, FP32)"].sum()]
print("\n📊 Parameter Breakdown — Court-MOE\n")
print(df.to_string(index=False))
