#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, torch, torch.nn as nn
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt

console = Console()
device  = torch.device("cpu")

def section(title: str): console.rule(f"[bold cyan]{title}")
def success(msg: str): console.print(f"[bold green]✔ {msg}")
def warn(msg: str): console.print(f"[bold yellow]⚠ {msg}")
def error(msg: str): console.print(f"[bold red]✘ {msg}")

# =========================================================
# === Model definitions ===================================
# =========================================================
class BiasOnly(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.bias   = nn.Parameter(torch.zeros(dim))
    def forward(self, x): return x * self.weight + self.bias

class ResidualBlockDimChange(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc0 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = BiasOnly(hidden_dim)
        self.se  = nn.Sequential(
            nn.Linear(hidden_dim, max(1, hidden_dim // 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, hidden_dim // 4), hidden_dim),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
    def forward(self, x):
        out = self.fc0(x); out = self.act(out); out = self.fc2(out)
        out = out * self.se(out)
        return self.proj(x) + out

class SERouterTrue(nn.Module):
    def __init__(self, in_dim=773, num_classes=5):
        super().__init__()
        self.in_ln  = nn.LayerNorm(in_dim)
        self.block1 = ResidualBlockDimChange(773, 1024)
        self.block2 = ResidualBlockDimChange(1024, 768)
        self.block3 = ResidualBlockDimChange(768, 512)
        self.block4 = ResidualBlockDimChange(512, 256)
        self.out    = nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.in_ln(x)
        x = self.block1(x); x = self.block2(x)
        x = self.block3(x); x = self.block4(x)
        return self.out(x)

class ExpertMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)

# =========================================================
# === Paths ===============================================
# =========================================================
ROUTER_PATH  = "routers/router_meta_boosted_61.64/best_router.pt"
EXPERTS_BASE = "Experts/experts_kfold_final"

# =========================================================
# === Loaders =============================================
# =========================================================
def _infer_in_dim_from_state_dict(sd: Dict[str, torch.Tensor]) -> int:
    for k, v in sd.items():
        if "weight" in k and v.ndim == 2:
            return v.shape[1]
    return 768

def load_router_cpu() -> nn.Module:
    if not os.path.exists(ROUTER_PATH):
        raise FileNotFoundError(f"Router checkpoint not found: {ROUTER_PATH}")
    ckpt = torch.load(ROUTER_PATH, map_location="cpu")
    model = SERouterTrue()
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    success("Router loaded on CPU")
    return model

def load_expert_ensemble_cpu(court: str):
    court = court.lower().strip()
    court_dir = os.path.join(EXPERTS_BASE, court)
    if not os.path.isdir(court_dir):
        raise FileNotFoundError(f"Experts not found at: {court_dir}")

    fold_paths = [os.path.join(court_dir, f"{court}_fold{i}.pt") for i in (1,2,3)]
    found = [p for p in fold_paths if os.path.exists(p)]
    if not found:
        raise FileNotFoundError(f"No expert checkpoints found in {court_dir}")

    models = []
    in_dim = 777 if court in ["district", "daily"] else 768
    for p in found:
        sd = torch.load(p, map_location="cpu")
        state = sd.get("model", sd)
        if court not in ["district", "daily"]:
            in_dim = _infer_in_dim_from_state_dict(state)
        m = ExpertMLP(in_dim)
        try:
            m.load_state_dict(state, strict=False)
        except Exception as e:
            warn(f"Issue loading {os.path.basename(p)}: {e}")
        m.eval().to(device)
        models.append(m)

    success(f"Loaded {court} experts ({len(models)} folds, in_dim={in_dim})")
    return models, in_dim

# =========================================================
# === Tokenizer + Encoder ================================
# =========================================================
TOKENIZER_PATH = "/home/infodna/Court-MOE/tokenization/final_tokenizer"
ENCODER_PATH   = "/home/infodna/Court-MOE/encoding/legalbert_finetuned_courts"

def load_encoder_cpu():
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        progress.add_task(description="Loading tokenizer & encoder...", total=None)
        tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
        enc = AutoModel.from_pretrained(ENCODER_PATH, local_files_only=True).eval().to(device)
    vocab_tok = len(tok)
    vocab_enc = enc.embeddings.word_embeddings.weight.size(0)
    if vocab_tok != vocab_enc:
        warn(f"Vocab mismatch — tokenizer={vocab_tok}, encoder={vocab_enc}. Resizing...")
        enc.resize_token_embeddings(vocab_tok)
        success(f"Encoder resized to {vocab_tok}")
    success(f"Tokenizer @ {TOKENIZER_PATH}")
    success(f"Encoder   @ {ENCODER_PATH}")
    return tok, enc

def encode_case_text(text: str, tok, enc) -> torch.Tensor:
    inputs = tok(text, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = enc(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
    return emb

# =========================================================
# === Inference ===========================================
# =========================================================
def _pad_to_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    cur = x.size(1)
    if cur == target_dim: return x
    if cur > target_dim:  return x[:, :target_dim]
    pad = torch.zeros((x.size(0), target_dim - cur), device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)

@torch.inference_mode()
def ensemble_predict(models: List[nn.Module], x: torch.Tensor) -> torch.Tensor:
    probs = []
    for m in models:
        out = m(x)
        prob = torch.sigmoid(out).squeeze(-1)
        probs.append(prob)
    return torch.stack(probs).mean(0)

def predict_single(case_text: str, court_type: str = None):
    valid = ["supreme", "high", "tribunal", "district", "daily"]
    tok, enc = load_encoder_cpu()
    case_emb = encode_case_text(case_text, tok, enc).to(device)

    if court_type:
        court = court_type.lower().strip()
        if court not in valid:
            raise ValueError(f"Invalid court '{court}'")
        section(f"Predicting Verdict — {court.capitalize()} Court")
        models, want_dim = load_expert_ensemble_cpu(court)
        x = _pad_to_dim(case_emb, want_dim)
        prob = ensemble_predict(models, x)[0].item()
        verdict = "✅ ACCEPTED" if prob >= 0.5 else "❌ REJECTED"
        return {"court": court, "prob_accept": round(prob,4), "verdict": verdict}

    section("Auto-Routing Mode")
    router = load_router_cpu()
    x_router = torch.cat([case_emb, torch.zeros(1, 5, device=device)], dim=1)
    logits = router(x_router)
    probs = torch.softmax(logits, dim=1)
    idx = torch.argmax(probs, dim=1).item()
    conf = probs[0, idx].item()
    courts = ["supreme","high","tribunal","district","daily"]
    court = courts[idx]
    success(f"Router → {court.upper()} (confidence={conf:.4f})")
    models, want_dim = load_expert_ensemble_cpu(court)
    x = _pad_to_dim(case_emb, want_dim)
    prob = ensemble_predict(models, x)[0].item()
    verdict = "✅ ACCEPTED" if prob >= 0.5 else "❌ REJECTED"
    return {"router_court": court, "router_confidence": round(conf,4),
            "prob_accept": round(prob,4), "verdict": verdict}

# =========================================================
# === UI ==================================================
# =========================================================
if __name__ == "__main__":
    section("Court-MOE — CPU Inference Interface")
    console.print("[bold white]Welcome to Court-MOE Single-Case Prediction[/bold white]\n")
    case_file = Prompt.ask("📂 Enter path to case file (e.g., Cases/sample.txt)")
    while not os.path.exists(case_file):
        error("Invalid path. Please enter a valid file.")
        case_file = Prompt.ask("📂 Enter path to case file")
    court_type = Prompt.ask("⚖  Enter court type [supreme/high/tribunal/district/daily] (press Enter to auto-detect)", default="")
    with open(case_file, "r", encoding="utf-8") as f:
        case_text = f.read().strip()
    result = predict_single(case_text, court_type if court_type.strip() else None)
    section("📊 Final Inference Summary")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Value", style="bold white")
    if "router_court" in result:
        table.add_row("Mode", "Auto (router-selected)")
        table.add_row("Court", result["router_court"].capitalize())
        table.add_row("Router Confidence", f"{result['router_confidence']:.4f}")
    else:
        table.add_row("Mode", "Manual (court provided)")
        table.add_row("Court", result["court"].capitalize())
    table.add_row("Acceptance Probability", f"{result['prob_accept']:.4f}")
    table.add_row("Verdict", result["verdict"])
    console.print(table)
    console.print(Panel.fit(
        f"[bold white]Verdict:[/bold white] {result['verdict']}   "
        f"[bold white]Probability:[/bold white] {result['prob_accept']:.4f}",
        title="[bold green]Court-MOE Result[/bold green]",
        border_style="bold green"
    ))
    console.print("\n[bold cyan]✨ Inference complete. CPU-only, no CUDA required.[/bold cyan]\n")
