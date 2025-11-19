#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Court-MOE — Unified Expert Training with K-Fold Ensemble (768 + 777 embeddings)
Stable: Supreme, High, Tribunal
Augmented: District, Daily (metadata_v2)
"""

import os, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef
from transformers import get_cosine_schedule_with_warmup
from torch.optim.swa_utils import AveragedModel, update_bn
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

# ============================================================
# CONFIG
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = "/home/infodna/Court-MOE"
SAVE_DIR = "experts_unified_kfold1"
os.makedirs(SAVE_DIR, exist_ok=True)

RUN_TAG = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_TXT = os.path.join(SAVE_DIR, f"training_{RUN_TAG}.log")

def log_line(s):
    print(s, flush=True)
    with open(LOG_TXT, "a", encoding="utf-8") as f:
        f.write(s + "\n")

K_FOLDS, EPOCHS = 3, 35
FP16, WEIGHT_DECAY, WARMUP_STEPS = True, 5e-5, 500
SWA_START_FRAC, EMA_DECAY, EARLY_STOP_PATIENCE = 0.75, 0.999, 7
CM_LABELS = ["Accepted", "Rejected"]

# ============================================================
# PATHS & EXPERT SETTINGS
# ============================================================
DATA_PATHS_768 = {
    "supreme":  f"{BASE_DIR}/encoding/encoded_output_final/final_balanced_by_court/SupremeCourt_embeddings_final.pth",
    "high":     f"{BASE_DIR}/encoding/encoded_output_final/final_balanced_by_court/HighCourt_embeddings_final.pth",
    "tribunal": f"{BASE_DIR}/encoding/encoded_output_final/final_balanced_by_court/TribunalCourt_embeddings_final.pth",
}
DATA_AUGMENTED_777 = f"{BASE_DIR}/encoding/metadata_augmented_v3_district_daily.pth"

EXPERT_CFG = {
    # stable 768-dim experts
    "supreme":  {"dim": 768, "batch": 144, "lr": 1.6e-4, "hidden": 1536, "dropout": 0.40, "mixup_a": 0.15, "mixup_p": 0.3},
    "high":     {"dim": 768, "batch": 128, "lr": 2.0e-4, "hidden": 2048, "dropout": 0.32, "mixup_a": 0.12, "mixup_p": 0.25},
    "tribunal": {"dim": 768, "batch": 128, "lr": 2.0e-4, "hidden": 1920, "dropout": 0.30, "mixup_a": 0.12, "mixup_p": 0.22},
    # augmented 777-dim
    "district": {"dim": 777, "batch": 128, "lr": 1.2e-4, "hidden": 2400, "dropout": 0.36, "mixup_a": 0.15, "mixup_p": 0.25},
    "daily":    {"dim": 777, "batch": 160, "lr": 1.0e-4, "hidden": 2560, "dropout": 0.40, "mixup_a": 0.20, "mixup_p": 0.35},
}

TARGET_IDX = {"district": 2, "daily": 4}
LABEL_MAP = {"Accepted": 0, "Rejected": 1}

# ============================================================
# UTILITIES
# ============================================================
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(42)

def to_jsonable(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy().tolist()
    if isinstance(x, (np.float32, np.float64)): return float(x)
    if isinstance(x, (np.int32, np.int64)): return int(x)
    return x

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_line(f"🧮 Parameters: {trainable/1e6:.2f}M trainable / {total/1e6:.2f}M total")

# ============================================================
# DATASETS (with memory-mapped reading)
# ============================================================
class MemoryMappedDataset(Dataset):
    def __init__(self, path, emb_dim=768):
        log_line(f"📂 Loading mmap: {path}")
        data = torch.load(path, map_location="cpu", mmap=True)
        embs, labels = [], []
        for d in data:
            e = np.asarray(d["embeddings"], np.float32)
            if e.ndim != 1 or e.shape[0] != emb_dim: continue
            y = LABEL_MAP.get(str(d["label"]).strip(), None)
            if y is None: continue
            embs.append(torch.from_numpy(e))
            labels.append(y)
        self.emb, self.y = torch.stack(embs), torch.tensor(labels, dtype=torch.float32)
        log_line(f"✅ Loaded {len(self.emb)} samples ({emb_dim}-dim)")

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return {"emb": self.emb[i], "y": self.y[i]}

class AugmentedDataset(Dataset):
    def __init__(self, path, court_idx, emb_dim=777):
        log_line(f"📂 Loading augmented mmap: {path} | court={court_idx}")
        data = torch.load(path, map_location="cpu", mmap=True)
        embs, labels = [], []
        for d in data:
            if int(d.get("court_type_idx", -1)) != court_idx: continue
            e = np.asarray(d.get("embeddings"), np.float32)
            if e.ndim != 1 or e.shape[0] != emb_dim: continue
            y = LABEL_MAP.get(str(d["label"]).strip(), None)
            if y is None: continue
            embs.append(torch.from_numpy(e)); labels.append(y)
        self.emb, self.y = torch.stack(embs), torch.tensor(labels, dtype=torch.float32)
        log_line(f"✅ Loaded {len(self.emb)} samples ({emb_dim}-dim, court={court_idx})")

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return {"emb": self.emb[i], "y": self.y[i]}

# ============================================================
# MODEL
# ============================================================
class SEBlock(nn.Module):
    def __init__(self, dim, reduction=32):
        super().__init__()
        r = max(8, dim // reduction)
        self.fc1, self.fc2 = nn.Linear(dim, r), nn.Linear(r, dim)
        self.act, self.gate = nn.GELU(), nn.Sigmoid()
    def forward(self, x):
        w = self.gate(self.fc2(self.act(self.fc1(x))))
        return x * w

class SEResidualMLP(nn.Module):
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.se, self.norm = SEBlock(dim), nn.LayerNorm(dim)
    def forward(self, x):
        return self.norm(x + self.se(self.block(x)))

class ExpertModel(nn.Module):
    def __init__(self, input_dim, hidden, dropout):
        super().__init__()
        self.proj = nn.Linear(input_dim, input_dim)
        self.body = SEResidualMLP(input_dim, hidden, dropout)
        self.head = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(512, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    def forward(self, x):
        x = self.proj(x)
        x = self.body(x)
        return self.head(x).squeeze(-1)

# ============================================================
# TRAINING UTILS
# ============================================================
class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
    @torch.no_grad()
    def update(self, model):
        for k,v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)
    @torch.no_grad()
    def apply_to(self, model):
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)

def maybe_mixup(emb, y, a, p):
    if a <= 0 or np.random.rand() > p: return emb, y
    lam = np.random.beta(a, a)
    idx = torch.randperm(emb.size(0), device=emb.device)
    return lam*emb + (1-lam)*emb[idx], lam*y + (1-lam)*y[idx]

class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos=2.0, gamma_neg=1.0, pos_weight=None):
        super().__init__()
        self.gamma_pos, self.gamma_neg = gamma_pos, gamma_neg
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        with torch.no_grad():
            p = torch.sigmoid(logits)
            pt = torch.where(targets==1,p,1-p)
            gamma = torch.where(targets==1,torch.full_like(pt,self.gamma_pos),torch.full_like(pt,self.gamma_neg))
            mod = (1-pt).clamp(min=1e-6).pow(gamma)
        return (mod*bce).mean()

# ============================================================
# EVALUATION
# ============================================================
def evaluate(model, loader):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for b in loader:
            e, y = b["emb"].to(DEVICE), b["y"].to(DEVICE)
            e = nn.functional.normalize(e, dim=1)
            prob = torch.sigmoid(model(e))
            ys.append(y.cpu().numpy()); ps.append(prob.cpu().numpy())
    y_true, y_prob = np.concatenate(ys), np.concatenate(ps)
    best_t,best_f1=0.5,-1
    for t in np.linspace(0.3,0.7,41):
        f1=f1_score(y_true,(y_prob>t),average="macro",zero_division=0)
        if f1>best_f1: best_f1,best_t=f1,t
    y_pred=(y_prob>best_t)
    return {
        "thr":float(best_t),
        "acc":float((y_true==y_pred).mean()),
        "f1":float(best_f1),
        "prec":float(precision_score(y_true,y_pred,average="macro",zero_division=0)),
        "rec":float(recall_score(y_true,y_pred,average="macro",zero_division=0)),
        "mcc":float(matthews_corrcoef(y_true,y_pred)),
    }

# ============================================================
# TRAIN ONE FOLD
# ============================================================
def train_fold(name, ds, tr_idx, va_idx, cfg, fold_id):
    exp_dir = os.path.join(SAVE_DIR, name); os.makedirs(exp_dir, exist_ok=True)
    bs,lr,h,drop = cfg["batch"],cfg["lr"],cfg["hidden"],cfg["dropout"]
    mixa,mixp = cfg["mixup_a"],cfg["mixup_p"]
    tr_ds,va_ds = torch.utils.data.Subset(ds,tr_idx), torch.utils.data.Subset(ds,va_idx)
    y_train = torch.tensor([ds.y[i].item() for i in tr_idx])
    counts = torch.bincount(y_train.long(), minlength=2).float()
    w_pos = counts[0]/(counts[1]+1e-6)
    weights = torch.where(y_train==1,w_pos,torch.ones_like(y_train)).double()
    sampler = WeightedRandomSampler(weights, num_samples=len(y_train), replacement=True)

    tr_loader = DataLoader(tr_ds,batch_size=bs,sampler=sampler,num_workers=2,pin_memory=True)
    va_loader = DataLoader(va_ds,batch_size=bs,shuffle=False,num_workers=2,pin_memory=True)

    model = ExpertModel(cfg["dim"],h,drop)
    count_params(model)
    if torch.cuda.device_count()>1: model = nn.DataParallel(model)
    model = model.to(DEVICE)

    criterion = AsymmetricFocalLoss(pos_weight=torch.tensor([w_pos]).to(DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    total_steps = len(tr_loader)*EPOCHS
    sched = get_cosine_schedule_with_warmup(opt, min(WARMUP_STEPS,max(10,total_steps//10)), total_steps)
    swa_start = int(EPOCHS*SWA_START_FRAC)
    swa,ema = AveragedModel(model), EMA(model)
    scaler = torch.amp.GradScaler("cuda") if FP16 else None

    best,no_imp={"f1":-1},0
    for ep in range(1,EPOCHS+1):
        model.train(); run=0
        for b in tqdm(tr_loader, desc=f"{name}[F{fold_id}] Ep{ep}/{EPOCHS}", leave=False):
            e,y=b["emb"].to(DEVICE),b["y"].to(DEVICE)
            e,y=maybe_mixup(e,y,mixa,mixp)
            opt.zero_grad(set_to_none=True)
            if scaler:
                with torch.amp.autocast("cuda",enabled=True):
                    loss=criterion(model(e),y)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt); scaler.update()
            else:
                loss=criterion(model(e),y)
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            sched.step(); ema.update(model); run+=loss.item()
        if ep>=swa_start: swa.update_parameters(model)
        ema.apply_to(model)
        m=evaluate(model,va_loader)
        log_line(f"{name}[F{fold_id}] Ep{ep:02d}|loss={run/len(tr_loader):.4f}|acc={m['acc']:.4f}|f1={m['f1']:.4f}")
        if m["f1"]>best["f1"]:
            best={**m,"epoch":ep}
            torch.save(model.module.state_dict() if isinstance(model,nn.DataParallel) else model.state_dict(),
                       os.path.join(exp_dir,f"{name}_fold{fold_id}.pt"))
            no_imp=0
        else:no_imp+=1
        if no_imp>=EARLY_STOP_PATIENCE:
            log_line(f"⏹️ Early stop {name}[F{fold_id}] at Ep{ep}")
            break
    update_bn(tr_loader,swa,device=DEVICE)
    swa_metrics=evaluate(swa.to(DEVICE),va_loader)
    log_line(f"{name}[F{fold_id}] SWA|F1={swa_metrics['f1']:.4f}|Acc={swa_metrics['acc']:.4f}")
    return best

# ============================================================
# K-FOLD DRIVER
# ============================================================
def train_expert(name, dataset):
    log_line(f"\n🚀 {name.title()} Expert — {K_FOLDS}-Fold CV")
    cfg = EXPERT_CFG[name]
    y_all = dataset.y.numpy().astype(int)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    folds=[]
    for i,(tr,va) in enumerate(skf.split(np.zeros_like(y_all),y_all),1):
        folds.append(train_fold(name,dataset,tr,va,cfg,i))
    accs,f1s=[f['acc'] for f in folds],[f['f1'] for f in folds]
    mean_acc,mean_f1=np.mean(accs),np.mean(f1s)
    summ={"expert":name,"acc_mean":float(mean_acc),"f1_mean":float(mean_f1),"folds":folds}
    with open(os.path.join(SAVE_DIR,name,f"{name}_summary.json"),"w") as jf: json.dump(summ,jf,indent=2,default=to_jsonable)
    log_line(f"✅ {name}|MeanAcc={mean_acc:.4f}|MeanF1={mean_f1:.4f}")
    return summ

# ============================================================
# MAIN
# ============================================================
def main():
    results={}
    # 1. Stable experts
    for name,path in DATA_PATHS_768.items():
        ds=MemoryMappedDataset(path,emb_dim=768)
        results[name]=train_expert(name,ds)
    # 2. Augmented experts
    for name,idx in TARGET_IDX.items():
        ds=AugmentedDataset(DATA_AUGMENTED_777,idx,emb_dim=777)
        results[name]=train_expert(name,ds)
    # Save global summary
    with open(os.path.join(SAVE_DIR,f"summary_{RUN_TAG}.json"),"w") as jf:
        json.dump(results,jf,indent=2,default=to_jsonable)
    log_line("\n🎯 All experts trained successfully with K-Fold Ensemble.")

if __name__=="__main__":
    main()
