#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Court-MOE — Fine-tune District & Daily Experts on 777-dim metadata-augmented embeddings
"""

import os, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix
from transformers import get_cosine_schedule_with_warmup
from torch.optim.swa_utils import AveragedModel, update_bn
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

# ============================================================
# Safe unpickling (PyTorch ≥2.6)
# ============================================================
from numpy._core import multiarray
torch.serialization.add_safe_globals([multiarray._reconstruct, np.ndarray, dict, list])

# ============================================================
# CONFIG
# ============================================================
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AUG_PATH = "/home/infodna/Court-MOE/encoding/metadata_augmented_v2_district_daily.pth"
SAVE_ROOT = "experts_dd_kfold_final2"
os.makedirs(SAVE_ROOT, exist_ok=True)

RUN_TAG = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_TXT = os.path.join(SAVE_ROOT, f"training_{RUN_TAG}.txt")
with open(LOG_TXT, "a", encoding="utf-8") as f:
    f.write(f"\n=================== Fine-tune Run: {datetime.now()} ===================\n")

TARGET_EXPERTS = {"district": 2, "daily": 4}
K_FOLDS, EPOCHS, EMB_DIM = 3, 40, 777
FP16, WEIGHT_DECAY, WARMUP_STEPS = True, 5e-5, 500
SWA_START_FRAC, EMA_DECAY, EARLY_STOP_PATIENCE = 0.75, 0.999, 8
LABEL_MAP, CM_LABELS = {"Accepted": 0, "Rejected": 1}, ["Accepted", "Rejected"]

EXPERT_OVERRIDES = {
    "district": {"batch_size": 128, "lr": 1.2e-4, "hidden": 2304, "dropout": 0.35,
                 "mixup_alpha": 0.10, "mixup_prob": 0.25, "gamma_pos": 2.2, "gamma_neg": 1.0},
    "daily":    {"batch_size": 160, "lr": 1.0e-4, "hidden": 2560, "dropout": 0.40,
                 "mixup_alpha": 0.20, "mixup_prob": 0.35, "gamma_pos": 2.4, "gamma_neg": 1.0},
}

# ============================================================
# UTILS
# ============================================================
def log_line(s):
    print(s, flush=True)
    with open(LOG_TXT, "a", encoding="utf-8") as f:
        f.write(s + "\n")

def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(42)

def to_jsonable(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.float32, np.float64)): return float(o)
    if isinstance(o, (np.int32, np.int64)): return int(o)
    if isinstance(o, torch.Tensor): return o.detach().cpu().numpy().tolist()
    return o

# ============================================================
# LOSS — Asymmetric Focal Loss
# ============================================================
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_pos=2.0, gamma_neg=1.0, pos_weight=None):
        super().__init__()
        self.gamma_pos, self.gamma_neg = gamma_pos, gamma_neg
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        with torch.no_grad():
            p = torch.sigmoid(logits)
            pt = torch.where(targets == 1.0, p, 1.0 - p)
            gamma = torch.where(targets == 1.0,
                                torch.full_like(pt, self.gamma_pos),
                                torch.full_like(pt, self.gamma_neg))
            mod = (1 - pt).clamp(min=1e-6).pow(gamma)
        return (mod * bce).mean()

# ============================================================
# MODEL
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
    def __init__(self, dim=EMB_DIM, hidden=2048, dropout=0.3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, dim))
        self.block2 = nn.Sequential(
            nn.Linear(dim, hidden//2), nn.BatchNorm1d(hidden//2), nn.GELU(),
            nn.Dropout(dropout/1.2), nn.Linear(hidden//2, dim))
        self.se1, self.se2, self.norm = SEBlock(dim), SEBlock(dim), nn.LayerNorm(dim)
    def forward(self, x):
        res = x
        x = self.se1(self.block1(x)) + self.se2(self.block2(x))
        return self.norm(x + res)

class ExpertOnEmbeddings(nn.Module):
    def __init__(self, input_dim=EMB_DIM, hidden=2304, dropout=0.35):
        super().__init__()
        self.proj = nn.Linear(input_dim, input_dim)
        self.semlp = SEResidualMLP(input_dim, hidden, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(512, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, emb):
        x = self.proj(emb)
        x = self.semlp(x)
        return self.classifier(x).squeeze(-1)

# ============================================================
# DATASET + robust label parsing
# ============================================================
def parse_label(raw):
    if raw is None: return None
    if isinstance(raw, (int, np.integer)): return int(raw) if raw in (0,1) else None
    if isinstance(raw, bool): return int(raw)
    s = str(raw).strip().lower()
    if s in {"0","acc","accepted","accept","yes","y"}: return 0
    if s in {"1","rej","rejected","reject","no","n"}:  return 1
    return None

class CourtSubset773(Dataset):
    def __init__(self, path, target_court_idx: int, emb_dim=EMB_DIM):
        print(f"🧩 Using dataset file: {path}")
        torch.serialization.add_safe_globals([multiarray._reconstruct, dict, list, np.ndarray])
        data = torch.load(path, map_location="cpu", weights_only=False)

        total = len(data); c_match=c_dim=c_label=0
        embs, labels = [], []
        for i, d in enumerate(data):
            try:
                cid = d.get("court_type_idx", -1)
                try: cid = int(cid)
                except: cid = -1
                if cid != target_court_idx: continue
                c_match += 1

                emb = np.asarray(d.get("embeddings"), np.float32)
                if emb.ndim != 1 or emb.shape[0] != emb_dim: continue
                c_dim += 1

                y = parse_label(d.get("label"))
                if y is None: continue
                c_label += 1

                embs.append(torch.from_numpy(emb))
                labels.append(y)
            except Exception as ex:
                print(f"⚠️ Skipping idx={i}: {ex}")

        print(f"📊 Filter summary (court={target_court_idx}): "
              f"total={total}, court_match={c_match}, dim_ok={c_dim}, label_ok={c_label}")
        assert len(embs) > 0, f"No samples collected for court={target_court_idx}"
        self.emb, self.y = torch.stack(embs), torch.tensor(labels, dtype=torch.float32)
        print(f"✅ Loaded {len(self.emb)} samples | emb_dim={self.emb.shape[1]}")

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return {"emb": self.emb[i], "y": self.y[i]}

# ============================================================
# TRAIN UTILITIES
# ============================================================
def build_sampler(y):
    c = torch.bincount(y.long(), minlength=2).float()
    w_pos = c[0]/(c[1]+1e-6)
    return torch.where(y==1, w_pos, torch.ones_like(y)).double(), c

def maybe_mixup(emb, y, a, p):
    if a <= 0 or np.random.rand() > p: return emb, y
    lam = np.random.beta(a, a)
    idx = torch.randperm(emb.size(0), device=emb.device)
    return lam*emb + (1-lam)*emb[idx], lam*y + (1-lam)*y[idx]

class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {k:v.detach().clone() for k,v in model.state_dict().items() if v.dtype.is_floating_point}
        print(f"✅ EMA initialized ({len(self.shadow)} tensors)")
    @torch.no_grad()
    def update(self, model):
        for k,v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)
    @torch.no_grad()
    def apply_to(self, model):
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)

def evaluate(model, loader):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for b in loader:
            emb = b["emb"].to(DEVICE); y = b["y"].to(DEVICE)
            prob = torch.sigmoid(model(nn.functional.normalize(emb, dim=1)))
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
        "cm":confusion_matrix(y_true,y_pred,labels=[0,1])
    }

def save_cm(cm, title, out_png):
    fig,ax=plt.subplots(figsize=(6,5))
    im=ax.imshow(cm,interpolation='nearest')
    ax.set(title=title,xlabel="Predicted",ylabel="True",
           xticks=[0,1],xticklabels=CM_LABELS,yticks=[0,1],yticklabels=CM_LABELS)
    for (i,j),v in np.ndenumerate(cm):
        ax.text(j,i,str(int(v)),ha='center',va='center',
                color='white' if im.norm(v)>0.5 else 'black')
    plt.tight_layout(); plt.savefig(out_png); plt.close(fig)

# ============================================================
# TRAINING
# ============================================================
def train_one_fold(name, ds, tr_idx, va_idx, ov, fid):
    exp_dir = os.path.join(SAVE_ROOT, name); os.makedirs(exp_dir, exist_ok=True)
    bs,lr,h,drop = ov["batch_size"],ov["lr"],ov["hidden"],ov["dropout"]
    mixa,mixp,gp,gn = ov["mixup_alpha"],ov["mixup_prob"],ov["gamma_pos"],ov["gamma_neg"]

    tr_ds,va_ds = torch.utils.data.Subset(ds,tr_idx), torch.utils.data.Subset(ds,va_idx)
    y_train = torch.tensor([ds.y[i].item() for i in tr_idx])
    weights,counts = build_sampler(y_train)
    pos_weight = counts[0]/(counts[1]+1e-6)
    tr_loader = DataLoader(tr_ds,batch_size=bs,
                           sampler=WeightedRandomSampler(weights,len(y_train)),
                           num_workers=2,pin_memory=True)
    va_loader = DataLoader(va_ds,batch_size=bs,shuffle=False,num_workers=2,pin_memory=True)

    model = ExpertOnEmbeddings(EMB_DIM,h,drop)
    if torch.cuda.device_count()>1: model = nn.DataParallel(model)
    model = model.to(DEVICE)

    crit = AsymmetricFocalLoss(gamma_pos=gp,gamma_neg=gn,
                               pos_weight=torch.tensor([pos_weight]).to(DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    total = len(tr_loader)*EPOCHS
    sched = get_cosine_schedule_with_warmup(opt,min(WARMUP_STEPS,max(10,total//10)),total)
    swa_start = int(EPOCHS*SWA_START_FRAC)
    swa,ema = AveragedModel(model), EMA(model)
    scaler = torch.amp.GradScaler("cuda") if FP16 else None

    best,no_imp = {"f1":-1},0
    for ep in range(1,EPOCHS+1):
        model.train(); run=0
        for b in tqdm(tr_loader,desc=f"{name}[F{fid}] Ep{ep}/{EPOCHS}",leave=False):
            emb=b["emb"].to(DEVICE); y=b["y"].to(DEVICE)
            emb,y=maybe_mixup(emb,y,mixa,mixp)
            opt.zero_grad(set_to_none=True)
            if scaler:
                with torch.amp.autocast("cuda",enabled=True):
                    loss=crit(model(emb),y)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt); scaler.update()
            else:
                loss=crit(model(emb),y); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            sched.step(); ema.update(model); run+=loss.item()
        if ep>=swa_start: swa.update_parameters(model)
        ema.apply_to(model)
        m=evaluate(model,va_loader)
        log_line(f"{name}[F{fid}] Ep{ep:02d}|loss={run/len(tr_loader):.4f}|acc={m['acc']:.4f}|f1={m['f1']:.4f}")
        if m["f1"]>best["f1"]:
            best={**m,"epoch":ep}
            torch.save(model.module.state_dict() if isinstance(model,nn.DataParallel) else model.state_dict(),
                       os.path.join(exp_dir,f"{name}_fold{fid}.pt"))
            save_cm(m["cm"],f"{name.title()}-Fold{fid}",os.path.join(exp_dir,f"{name}_fold{fid}_cm.png"))
            no_imp=0
        else: no_imp+=1
        if no_imp>=EARLY_STOP_PATIENCE:
            log_line(f"⏹️ Early stop {name}[F{fid}] at Ep{ep}")
            break
    try:
        update_bn(tr_loader,swa,device=DEVICE)
        sm=evaluate(swa.to(DEVICE),va_loader)
        with open(os.path.join(exp_dir,f"{name}_fold{fid}_swa.json"),"w") as jf: json.dump(sm,jf,indent=2,default=to_jsonable)
    except Exception as e: log_line(f"SWA skip {name}[F{fid}]: {e}")
    log_line(f"🏁 {name}[F{fid}] Best Ep{best.get('epoch')}|Acc={best['acc']:.4f}|F1={best['f1']:.4f}")
    return best

# ============================================================
# DRIVER / MAIN
# ============================================================
def train_expert_kfold(name, idx):
    log_line(f"\n🚀 {name.title()} Expert — {K_FOLDS}-Fold Fine-Tune CV")
    ds = CourtSubset773(AUG_PATH, idx)
    ov = EXPERT_OVERRIDES[name]
    y_all = ds.y.numpy().astype(int)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    res = []
    for fid, (tr, va) in enumerate(skf.split(np.zeros_like(y_all), y_all), 1):
        res.append(train_one_fold(name, ds, tr, va, ov, fid))

    accs = [r["acc"] for r in res]
    f1s  = [r["f1"] for r in res]
    summ = {
        "expert": name,
        "folds": res,
        "acc_mean": float(np.mean(accs)),
        "f1_mean": float(np.mean(f1s))
    }

    os.makedirs(os.path.join(SAVE_ROOT, name), exist_ok=True)
    with open(os.path.join(SAVE_ROOT, name, f"{name}_summary.json"), "w") as jf:
        json.dump(summ, jf, indent=2, default=to_jsonable)
    log_line(f"✅ {name} | MeanAcc={summ['acc_mean']:.4f} | MeanF1={summ['f1_mean']:.4f}")
    return summ

def main():
    res = {}
    for n, i in TARGET_EXPERTS.items():
        res[n] = train_expert_kfold(n, i)

    # Save global summary
    with open(os.path.join(SAVE_ROOT, f"summary_{RUN_TAG}.json"), "w") as jf:
        json.dump(res, jf, indent=2, default=to_jsonable)

    log_line("\n🎯 Fine-tuning complete. Results saved.")

if __name__ == "__main__":
    main()
