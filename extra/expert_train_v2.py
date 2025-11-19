"""
Court-MOE — Expert Training on Precomputed Embeddings with 3-Fold Stratified CV
Final Stable Build:
  • AMP + Multi-GPU
  • Asymmetric Focal Loss (+pos_weight)
  • MixUp regularization
  • Cosine LR w/ warmup + AdamW + weight decay
  • EMA (float-only, safe) + SWA (+ BN update)
  • Early stopping (patience=6)
  • Macro-F1 threshold tuning (0.30..0.70)
  • Confusion matrices (raw + normalized) PNG per fold
  • Robust JSON (no ndarray errors)
"""
import os, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix
from transformers import get_cosine_schedule_with_warmup
from torch.optim.swa_utils import AveragedModel, update_bn
import matplotlib.pyplot as plt
plt.switch_backend("Agg")  # headless-safe

# ==============================
# CONFIG
# ==============================
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATHS = {
    "supreme":  "encoding/encoded_output_final/final_balanced_by_court/SupremeCourt_embeddings_final.pth",
    "high":     "encoding/encoded_output_final/final_balanced_by_court/HighCourt_embeddings_final.pth",
    "district": "encoding/encoded_output_final/final_balanced_by_court/DistrictCourt_embeddings_final.pth",
    "tribunal": "encoding/encoded_output_final/final_balanced_by_court/TribunalCourt_embeddings_final.pth",
    "daily":    "encoding/encoded_output_final/final_balanced_by_court/DailyOrderCourt_embeddings_final.pth",
}

SAVE_DIR = "experts_kfold"
os.makedirs(SAVE_DIR, exist_ok=True)

RUN_TAG = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_TXT = os.path.join(SAVE_DIR, f"training_{RUN_TAG}.txt")
with open(LOG_TXT, "a", encoding="utf-8") as f:
    f.write(f"\n=================== Run: {datetime.now()} ===================\n")

# Global knobs
K_FOLDS = 3
EPOCHS = 30
META_DIM = 5
EMB_DIM = 768
FP16 = True
WEIGHT_DECAY = 5e-5
WARMUP_STEPS = 500
SWA_START_FRAC = 0.7
EMA_DECAY = 0.999
EARLY_STOP_PATIENCE = 6

LABEL_MAP = {"Accepted": 0, "Rejected": 1}
CM_LABELS = ["Accepted", "Rejected"]

# Per-expert overrides
EXPERT_OVERRIDES = {
    # ✔ Supreme: stricter regularization for stability
    "supreme":  {"batch_size":144,"lr":1.6e-4,"hidden":1536,"dropout":0.40,"mixup_alpha":0.15,"mixup_prob":0.30,"gamma_pos":1.5,"gamma_neg":1.0},
    # ✔ High: slightly higher width for extra capacity
    "high":     {"batch_size":128,"lr":2.2e-4,"hidden":2304,"dropout":0.30,"mixup_alpha":0.10,"mixup_prob":0.25,"gamma_pos":2.0,"gamma_neg":1.0},
    # ✔ District: stable—gentle regularization
    "district": {"batch_size":128,"lr":2.0e-4,"hidden":1792,"dropout":0.30,"mixup_alpha":0.10,"mixup_prob":0.20,"gamma_pos":2.0,"gamma_neg":1.0},
    # ✔ Tribunal: stable—slightly wider head
    "tribunal": {"batch_size":128,"lr":2.0e-4,"hidden":1920,"dropout":0.30,"mixup_alpha":0.12,"mixup_prob":0.22,"gamma_pos":2.0,"gamma_neg":1.0},
    # ✔ Daily: weakest—more width + dropout + stronger MixUp
    "daily":    {"batch_size":160,"lr":1.8e-4,"hidden":2304,"dropout":0.38,"mixup_alpha":0.20,"mixup_prob":0.35,"gamma_pos":2.2,"gamma_neg":1.0},
}

# ==============================
# UTILS
# ==============================
def log_line(s: str):
    print(s, flush=True)
    with open(LOG_TXT, "a", encoding="utf-8") as f:
        f.write(s + "\n")

def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(42)

def to_jsonable(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32, np.int64)): return int(obj)
    if isinstance(obj, torch.Tensor): return obj.detach().cpu().numpy().tolist()
    return obj

# ==============================
# LOSS — Asymmetric Focal + pos_weight
# ==============================
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos=2.0, gamma_neg=1.0, pos_weight=None):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        with torch.no_grad():
            p = torch.sigmoid(logits)
            pt = torch.where(targets == 1.0, p, 1.0 - p)
            gamma = torch.where(
                targets == 1.0,
                torch.full_like(pt, self.gamma_pos),
                torch.full_like(pt, self.gamma_neg),
            )
            mod = (1 - pt).clamp(min=1e-6).pow(gamma)
        return (mod * bce).mean()

# ==============================
# MODEL
# ==============================
class SEBlock(nn.Module):
    def __init__(self, dim, reduction=32):
        super().__init__()
        red = max(8, dim // reduction)
        self.fc1 = nn.Linear(dim, red)
        self.fc2 = nn.Linear(red, dim)
        self.act = nn.GELU()
        self.gate = nn.Sigmoid()
    def forward(self, x):
        w = self.gate(self.fc2(self.act(self.fc1(x))))
        return x * w

class SEResidualMLP(nn.Module):
    def __init__(self, dim=EMB_DIM, hidden=2048, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.se = SEBlock(dim)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        res = x
        x = self.drop(self.act(self.bn1(self.fc1(x))))
        x = self.drop(self.fc2(x))
        x = self.se(x)
        return self.norm(x + res)

class ExpertOnEmbeddings(nn.Module):
    """
    Expert head that operates on frozen 768-d sentence embeddings (+ optional metadata)
    """
    def __init__(self, hidden=2048, dropout=0.30, use_metadata=True, meta_dim=META_DIM):
        super().__init__()
        self.use_metadata = use_metadata
        self.proj = nn.Linear(EMB_DIM, EMB_DIM)
        self.semlp = SEResidualMLP(dim=EMB_DIM, hidden=hidden, dropout=dropout)
        if self.use_metadata:
            self.meta = nn.Sequential(
                nn.Linear(meta_dim, 64), nn.GELU(),
                nn.Linear(64, 128), nn.LayerNorm(128)
            )
            in_dim = EMB_DIM + 128
        else:
            in_dim = EMB_DIM
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(512, 1),
        )
        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, emb, meta=None):
        x = self.proj(emb)
        x = self.semlp(x)
        if self.use_metadata and meta is not None:
            m = self.meta(meta)
            x = torch.cat([x, m], dim=-1)
        return self.classifier(x).squeeze(-1)

# ==============================
# DATASET
# ==============================
class CourtDataset(Dataset):
    def __init__(self, path, meta_dim=META_DIM):
        data = torch.load(path, map_location="cpu")
        embs, metas, labels = [], [], []
        for i, d in enumerate(data):
            try:
                e = torch.as_tensor(d["embeddings"], dtype=torch.float32)
                if e.ndim != 1 or e.shape[0] != EMB_DIM:
                    continue
                y = LABEL_MAP.get(str(d["label"]).strip(), None)
                if y is None:
                    continue
                m = d.get("metadata", None)
                if m is None:
                    m = torch.zeros(meta_dim, dtype=torch.float32)
                else:
                    m = torch.as_tensor(m, dtype=torch.float32)
                    if m.ndim != 1 or m.shape[0] != meta_dim:
                        m = torch.zeros(meta_dim, dtype=torch.float32)
                embs.append(e); metas.append(m); labels.append(y)
            except Exception as ex:
                print(f"⚠️  Skipping idx={i}: {ex}")
        self.emb = torch.stack(embs)                 # [N, 768]
        self.meta = torch.stack(metas)               # [N, meta_dim]
        self.y = torch.tensor(labels, dtype=torch.float32)  # [N]
        print(f"✅ Loaded {len(self.emb)} samples | emb={self.emb.shape[1]} | meta={self.meta.shape[1]}")

    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return {"emb": self.emb[idx], "meta": self.meta[idx], "y": self.y[idx]}

# ==============================
# SAMPLER & REGULARIZERS
# ==============================
def build_sampler(labels_tensor):
    # Weighted sampling by inverse freq of positive class
    counts = torch.bincount(labels_tensor.long(), minlength=2).float()
    w_pos = counts[0] / (counts[1] + 1e-6)
    weights = torch.where(labels_tensor == 1, w_pos, torch.ones_like(labels_tensor))
    return weights.double(), counts

def maybe_mixup(emb, meta, y, alpha, prob):
    if alpha <= 0 or np.random.rand() > prob:
        return emb, meta, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(emb.size(0), device=emb.device)
    emb2, meta2, y2 = emb[idx], meta[idx], y[idx]
    return lam*emb + (1-lam)*emb2, lam*meta + (1-lam)*meta2, lam*y + (1-lam)*y2

# ==============================
# EMA (float-only)
# ==============================
class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        print(f"✅ EMA initialized ({len(self.shadow)} float tensors tracked)")

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)

# ==============================
# EVALUATION & CM PLOTS
# ==============================
def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            emb = nn.functional.normalize(batch["emb"].to(DEVICE), dim=1)
            meta = batch["meta"].to(DEVICE)
            y = batch["y"].to(DEVICE)
            prob = torch.sigmoid(model(emb, meta))
            ys.append(y.cpu().numpy()); ps.append(prob.cpu().numpy())
    y_true = np.concatenate(ys); y_prob = np.concatenate(ps)

    # Threshold tuning for macro-F1
    grid = np.linspace(0.30, 0.70, 41)
    best_thr, best_f1 = 0.5, -1.0
    for t in grid:
        y_pred = (y_prob > t).astype(int)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    y_pred = (y_prob > best_thr).astype(int)

    acc  = float((y_true == y_pred).mean())
    f1   = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec  = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    mcc  = float(matthews_corrcoef(y_true, y_pred))
    cm   = confusion_matrix(y_true, y_pred, labels=[0,1])

    return {"thr": float(best_thr), "acc": acc, "f1": f1, "prec": prec, "rec": rec, "mcc": mcc, "cm": cm}

def save_confusion_pngs(cm, title, out_png):
    # Raw
    fig = plt.figure(figsize=(6.2, 5.2), dpi=140)
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(f"{title} — Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_xticklabels(CM_LABELS)
    ax.set_yticks([0,1]); ax.set_yticklabels(CM_LABELS)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{int(v)}", ha='center', va='center',
                color='white' if im.norm(v) > 0.5 else 'black', fontsize=10)
    fig.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(out_png); plt.close(fig)

    # Normalized
    cmn = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig2 = plt.figure(figsize=(6.2, 5.2), dpi=140)
    ax2 = fig2.add_subplot(111)
    im2 = ax2.imshow(cmn, interpolation='nearest')
    ax2.set_title(f"{title} — Normalized")
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("True")
    ax2.set_xticks([0,1]); ax2.set_xticklabels(CM_LABELS)
    ax2.set_yticks([0,1]); ax2.set_yticklabels(CM_LABELS)
    for (i, j), v in np.ndenumerate(cmn):
        ax2.text(j, i, f"{v*100:.1f}%", ha='center', va='center',
                 color='white' if im2.norm(v) > 0.5 else 'black', fontsize=10)
    fig2.colorbar(im2, ax=ax2)
    plt.tight_layout(); plt.savefig(out_png.replace(".png", "_norm.png")); plt.close(fig2)

# ==============================
# TRAIN ONE FOLD
# ==============================
def train_one_fold(name, ds, tr_idx, va_idx, ov, fold_id):
    exp_dir = os.path.join(SAVE_DIR, name)
    os.makedirs(exp_dir, exist_ok=True)

    bs, lr      = ov["batch_size"], ov["lr"]
    hidden      = ov["hidden"]
    dropout     = ov["dropout"]
    mixa, mixp  = ov["mixup_alpha"], ov["mixup_prob"]
    gpos, gneg  = ov["gamma_pos"], ov["gamma_neg"]

    tr_ds, va_ds = torch.utils.data.Subset(ds, tr_idx), torch.utils.data.Subset(ds, va_idx)
    y_train = torch.tensor([ds.y[i].item() for i in tr_idx], dtype=torch.float32)
    weights, counts = build_sampler(y_train)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(y_train), replacement=True)
    pos_weight = counts[0] / (counts[1] + 1e-6)

    tr_loader = DataLoader(tr_ds, batch_size=bs, sampler=sampler, num_workers=2, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

    model = ExpertOnEmbeddings(hidden=hidden, dropout=dropout, use_metadata=True, meta_dim=META_DIM)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        log_line(f"{name.title()}[fold {fold_id}] ✅ Using {torch.cuda.device_count()} GPUs")
    model = model.to(DEVICE)

    criterion = AsymmetricFocalLoss(gamma_pos=gpos, gamma_neg=gneg, pos_weight=pos_weight.to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    total_steps = len(tr_loader) * EPOCHS
    warmup = min(WARMUP_STEPS, max(10, total_steps // 10))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    swa_start = int(EPOCHS * SWA_START_FRAC)
    swa_model = AveragedModel(model)
    ema = EMA(model, decay=EMA_DECAY)
    scaler = torch.amp.GradScaler("cuda") if FP16 and DEVICE == "cuda" else None

    best = {"f1": -1.0}
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train(); run_loss = 0.0
        for batch in tqdm(tr_loader, desc=f"{name}[F{fold_id}] Ep{epoch}/{EPOCHS}", leave=False):
            emb  = nn.functional.normalize(batch["emb"].to(DEVICE), dim=1)
            meta = batch["meta"].to(DEVICE)
            y    = batch["y"].to(DEVICE)
            emb, meta, y = maybe_mixup(emb, meta, y, mixa, mixp)

            optimizer.zero_grad(set_to_none=True)
            if scaler:
                with torch.amp.autocast("cuda", enabled=True):
                    logits = model(emb, meta)
                    loss   = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                logits = model(emb, meta); loss = criterion(logits, y)
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()

            scheduler.step(); ema.update(model); run_loss += loss.item()

        # SWA tracking + EMA eval
        if epoch >= swa_start:
            swa_model.update_parameters(model)

        # Evaluate with EMA weights applied
        ema.apply_to(model)
        metrics = evaluate(model, va_loader)

        log_line(f"{name.title()}[F{fold_id}] Ep{epoch:02d} | loss={run_loss/len(tr_loader):.4f} | "
                 f"acc={metrics['acc']:.4f} | f1={metrics['f1']:.4f} | prec={metrics['prec']:.4f} | "
                 f"rec={metrics['rec']:.4f} | mcc={metrics['mcc']:.4f} | thr={metrics['thr']:.2f}")

        # Save best by F1
        if metrics["f1"] > best["f1"]:
            best = {**metrics, "epoch": epoch}
            # checkpoint
            ckpt_path = os.path.join(exp_dir, f"{name}_fold{fold_id}.pt")
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), ckpt_path)
            # confusion matrices
            cm_png = os.path.join(exp_dir, f"{name}_fold{fold_id}_cm.png")
            save_confusion_pngs(metrics["cm"], f"{name.title()} — Fold {fold_id}", cm_png)
            # reset early stopping
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP_PATIENCE:
            log_line(f"⏹️  Early stopping {name}[F{fold_id}] at epoch {epoch} (no improvement {EARLY_STOP_PATIENCE} epochs).")
            break

    # Optional: evaluate SWA weights (with BN update)
    try:
        update_bn(tr_loader, swa_model, device=DEVICE)
        swa_model = swa_model.to(DEVICE)
        swa_metrics = evaluate(swa_model, va_loader)
        with open(os.path.join(exp_dir, f"{name}_fold{fold_id}_swa_metrics.json"), "w") as jf:
            json.dump(swa_metrics, jf, indent=2, default=to_jsonable)
        log_line(f"{name.title()}[F{fold_id}] SWA | acc={swa_metrics['acc']:.4f} | f1={swa_metrics['f1']:.4f}")
    except Exception as e:
        log_line(f"⚠️  SWA eval skipped for {name}[F{fold_id}]: {e}")

    log_line(f"🏁 {name.title()}[F{fold_id}] Best Ep{best.get('epoch')} | Acc={best['acc']:.4f} | F1={best['f1']:.4f}")
    return best

# ==============================
# K-FOLD DRIVER
# ==============================
def train_expert_kfold(name, path):
    log_line(f"\n🚀 Expert {name.title()} — 3-Fold Stratified CV")
    ds = CourtDataset(path, meta_dim=META_DIM)
    override = EXPERT_OVERRIDES[name]
    y_all = ds.y.numpy().astype(int)

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(np.zeros_like(y_all), y_all), start=1):
        fold_results.append(train_one_fold(name, ds, tr_idx, va_idx, override, fold_id))

    accs = [fr["acc"] for fr in fold_results]
    f1s  = [fr["f1"] for fr in fold_results]
    summary = {
        "expert": name,
        "folds": fold_results,
        "acc_mean": float(np.mean(accs)),
        "acc_std":  float(np.std(accs)),
        "f1_mean":  float(np.mean(f1s)),
        "f1_std":   float(np.std(f1s)),
    }
    os.makedirs(os.path.join(SAVE_DIR, name), exist_ok=True)
    with open(os.path.join(SAVE_DIR, name, f"{name}_summary.json"), "w") as jf:
        json.dump(summary, jf, indent=2, default=to_jsonable)

    log_line(f"✅ {name.title()} | Mean Acc={summary['acc_mean']:.4f}±{summary['acc_std']:.4f} | "
             f"Mean F1={summary['f1_mean']:.4f}±{summary['f1_std']:.4f}")
    return summary

# ==============================
# MAIN
# ==============================
def main():
    results = {}
    for name, path in DATA_PATHS.items():
        results[name] = train_expert_kfold(name, path)

    with open(os.path.join(SAVE_DIR, f"summary_{RUN_TAG}.json"), "w") as jf:
        json.dump(results, jf, indent=2, default=to_jsonable)

    log_line("\n🎯 All experts trained successfully.")
    log_line(f"Artifacts per expert → {os.path.abspath(SAVE_DIR)}/<expert>/")
    log_line("  • <expert>_fold{k}.pt")
    log_line("  • <expert>_fold{k}_cm.png and _cm_norm.png")
    log_line("  • <expert>_fold{k}_swa_metrics.json (optional)")
    log_line("  • <expert>_summary.json")
    log_line(f"Global summary → {os.path.join(SAVE_DIR, f'summary_{RUN_TAG}.json')}")

if __name__ == "__main__":
    main()
