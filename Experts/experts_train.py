import os, json, math, random, numpy as np
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Optional, List
from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

# =========================
# GLOBAL CONFIG
# =========================
CONFIG = {
    "save_dir": "Experts/experts_kfold_final/",
    "embeddings": {
        "supreme": "encoding/encoded_output_final/final_balanced_by_court/SupremeCourt_embeddings_final.pth",
        "high":   "encoding/encoded_output_final/final_balanced_by_court/HighCourt_embeddings_final.pth",
        "tribunal":"encoding/encoded_output_final/final_balanced_by_court/TribunalCourt_embeddings_final.pth",
        "district": "encoding/metadata_augmented_v3_district_daily.pth",
        "daily":    "encoding/metadata_augmented_v3_district_daily.pth"
    },
    "epochs": 40,
    "batch_size": 512,
    "lr": 3e-4,
    "weight_decay": 5e-3,    
    "dropout_p": 0.25,
    "mixup_alpha": 0.1,
    "warmup_ratio": 0.10,
    "early_stop_patience": 8,
    "num_workers": 8,
    "grad_clip_norm": 1.0,
    "use_ema": True,
    "ema_decay": 0.999,
    "use_swa": True,
    "swa_pct": 0.20, 
    "swa_lr": None,         
    "use_afl": False
}
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
# =========================
# UTILITIES
# =========================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def label_to_binary(label: Any) -> int:
    """Convert label to binary integer (0 or 1)"""
    if isinstance(label, str):
        label_lower = label.lower().strip()
        if label_lower in ["accepted", "accept", "1", "true", "yes"]:
            return 1
        elif label_lower in ["rejected", "reject", "0", "false", "no"]:
            return 0
        else:
            raise ValueError(f"Unknown label string: {label}")
    elif isinstance(label, (int, float)):
        return int(label)
    else:
        raise ValueError(f"Unsupported label type: {type(label)}")

def effective_num_pos_weight(y: torch.Tensor,
                             beta: float = 0.999,
                             clip_min: float = 0.5,
                             clip_max: float = 5.0) -> float:
    y_np = y.detach().cpu().numpy()
    n = len(y_np)
    n_pos = float(y_np.sum())
    n_neg = n - n_pos

    def eff_num(k: float) -> float:
        return (1 - beta ** k) / (1 - beta) if k > 0 else 1.0

    w_pos = eff_num(n) / eff_num(max(n_pos, 1.0))
    w_neg = eff_num(n) / eff_num(max(n_neg, 1.0))
    pos_w = w_neg / max(w_pos, 1e-6)

    return float(np.clip(pos_w, clip_min, clip_max))


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """MixUp directly on embedding space."""
    if alpha is None or alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]

def feature_drop(x: torch.Tensor, meta_slice: Optional[Tuple[int, int]], p: float = 0.15) -> torch.Tensor:
    """With prob p, zero-out the metadata slice to improve robustness."""
    if (meta_slice is None) or (random.random() > p):
        return x
    s, e = meta_slice
    mask = torch.ones_like(x)
    mask[:, s:e] = 0
    return x * mask


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Pick threshold that maximizes F1 over a grid."""
    best_f1, best_thr = -1.0, 0.5
    for thr in np.linspace(0.30, 0.70, 41):
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    y_pred = (y_prob >= best_thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    except ValueError:
        auc = 0.0

    return float(best_thr), {
        "acc": float(acc),
        "f1": float(best_f1),
        "prec": float(prec),
        "rec": float(rec),
        "auc": float(auc)
    }

# =========================
# DATASET / LOADING
# =========================
class EmbeddingDataset(Dataset):
    """Dataset for precomputed embeddings"""
    def __init__(self, X: torch.Tensor, y: torch.Tensor, meta_slice: Optional[Tuple[int, int]] = None):
        self.X = X.float()
        self.y = y.float()
        self.meta_slice = meta_slice

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]


def load_embeddings(path: str, court: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding file not found: {path}")

    print(f"📂 Loading embeddings for {court.upper()} from: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict) and 'X' in obj and 'y' in obj:
        X, y = obj['X'], obj['y'].float()

    elif isinstance(obj, list):
        Xs, ys = [], []
        skipped = {"court_mismatch": 0, "bad_format": 0}
        for idx, item in enumerate(obj):
            try:
                # Ensure required keys exist
                if not all(k in item for k in ["embeddings", "label", "court_type_idx"]):
                    print(f"⚠ Record {idx} missing expected keys. Found: {list(item.keys())}")
                    skipped["bad_format"] += 1
                    continue

                emb = item["embeddings"]
                # ✅ FIX: Convert string labels to binary integers
                lbl = label_to_binary(item["label"])
                court_idx = item["court_type_idx"]

                # ✅ Corrected court index mapping:
                # 0 = Supreme, 1 = High, 2 = District, 3 = Tribunal, 4 = Daily
                if court == "supreme" and court_idx != 0:
                    skipped["court_mismatch"] += 1
                    continue
                elif court == "high" and court_idx != 1:
                    skipped["court_mismatch"] += 1
                    continue
                elif court == "district" and court_idx != 2:
                    skipped["court_mismatch"] += 1
                    continue
                elif court == "tribunal" and court_idx != 3:
                    skipped["court_mismatch"] += 1
                    continue
                elif court == "daily" and court_idx != 4:
                    skipped["court_mismatch"] += 1
                    continue

                # Convert embeddings to tensor
                Xs.append(torch.as_tensor(emb, dtype=torch.float32))
                ys.append(lbl)

            except Exception as e:
                skipped["bad_format"] += 1
                if skipped["bad_format"] <= 3:
                    print(f"⚠ Error at record {idx}: {e}")
                continue

        if not Xs:
            raise ValueError(f"No valid samples for {court} in {path}")

        X, y = torch.stack(Xs), torch.tensor(ys, dtype=torch.float32)

        if skipped["court_mismatch"] or skipped["bad_format"]:
            print(
                f"   ↪ Skipped {skipped['court_mismatch']} due to mismatched court indices "
                f"and {skipped['bad_format']} malformed samples"
            )

    else:
        raise ValueError(f"Unsupported embeddings structure in: {path}")

    # Shape check
    assert X.ndim == 2 and y.ndim == 1 and X.size(0) == y.size(0), \
        f"Invalid shapes: X={X.shape}, y={y.shape}"

    pos = int(y.sum().item())
    neg = len(y) - pos
    print(f"✅ Loaded {court.upper()}: X={tuple(X.shape)} | +ve={pos}/{len(y)} ({100*pos/len(y):.1f}%) | -ve={neg} ({100*neg/len(y):.1f}%)")
    return X, y


# =========================
# MODEL
# =========================
class ExpertNet(nn.Module):
    """2-layer MLP head for frozen embeddings"""
    def __init__(self, in_dim: int, hidden: int = 512, p_drop: float = 0.25):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(hidden, 1)
        # init
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)


# =========================
# LOSSES
# =========================
class AsymmetricFocalLoss(nn.Module):
    """AFL for imbalanced binary logits. Default gn=2, gp=0 works well when negatives dominate."""
    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 2.0, clip: float = 0.05):
        super().__init__()
        self.gp, self.gn, self.clip = gamma_pos, gamma_neg, clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x_sigmoid = torch.sigmoid(logits)
        xs_neg = x_sigmoid.clamp(min=self.clip, max=1 - self.clip)
        xs_pos = 1.0 - xs_neg
        pt = xs_pos * targets + xs_neg * (1 - targets)
        w = (1 - pt) ** (self.gp * targets + self.gn * (1 - targets))
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        return (w * bce).mean()


# =========================
# EMA
# =========================
def ema_update(model: nn.Module, ema_model: nn.Module, mu: float) -> None:
    with torch.no_grad():
        for p, q in zip(model.parameters(), ema_model.parameters()):
            q.data.mul_(mu).add_(p.data, alpha=(1 - mu))


# =========================
# EVALUATION
# =========================
@torch.no_grad()
def eval_on_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    probs_all, y_all = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
        probs_all.append(probs)
        y_all.append(yb.numpy())

    probs_all = np.concatenate(probs_all)
    y_all = np.concatenate(y_all).astype(int)

    thr, m = tune_threshold(y_all, probs_all)
    m["threshold"] = thr
    return {"metrics": m, "probs": probs_all, "targets": y_all}


# =========================
# TRAINING (ONE COURT)
# =========================
@dataclass
class TrainCfg:
    court: str
    in_dim: int
    meta_slice: Optional[Tuple[int, int]]

    # Toggles
    use_afl: bool = CONFIG["use_afl"]
    use_ema: bool = CONFIG["use_ema"]
    ema_decay: float = CONFIG["ema_decay"]
    use_swa: bool = CONFIG["use_swa"]
    swa_pct: float = CONFIG["swa_pct"]
    swa_lr: Optional[float] = CONFIG["swa_lr"]

    # HPs
    epochs: int = CONFIG["epochs"]
    batch_size: int = CONFIG["batch_size"]
    lr: float = CONFIG["lr"]
    weight_decay: float = CONFIG["weight_decay"]
    mixup_alpha: float = CONFIG["mixup_alpha"]
    dropout_p: float = CONFIG["dropout_p"]
    early_stop_patience: int = CONFIG["early_stop_patience"]
    warmup_ratio: float = CONFIG["warmup_ratio"]
    save_dir: str = CONFIG["save_dir"]
    num_workers: int = CONFIG["num_workers"]
    grad_clip_norm: float = CONFIG["grad_clip_norm"]


def train_expert(court: str, X: torch.Tensor, y: torch.Tensor, cfg: TrainCfg, device: torch.device) -> List[Dict[str, Any]]:
    print(f"\n{'='*80}")
    print(f"🧠 Training {court.upper()} Expert")
    print(f"{'='*80}")
    print(f"Samples: {len(y)} | Positives: {int(y.sum())} ({100*y.sum()/len(y):.1f}%)")
    print(f"Input dim: {cfg.in_dim} | Batch size: {cfg.batch_size} | Workers: {cfg.num_workers}")
    print(f"EMA: {cfg.use_ema} | SWA: {cfg.use_swa} | MixUp alpha: {cfg.mixup_alpha}")

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_results: List[Dict[str, Any]] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n{'─'*80}\n📊 Fold {fold}/3\n{'─'*80}")

        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xv, yv = X[va_idx], y[va_idx]

        tr_ds = EmbeddingDataset(Xtr, ytr, cfg.meta_slice)
        va_ds = EmbeddingDataset(Xv, yv, cfg.meta_slice)

        tr_dl = DataLoader(
            tr_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=(cfg.num_workers > 0),
            prefetch_factor=2
        )
        va_dl = DataLoader(
            va_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=(cfg.num_workers > 0),
            prefetch_factor=2
        )

        base = ExpertNet(cfg.in_dim, hidden=512, p_drop=cfg.dropout_p).to(device)
        model = nn.DataParallel(base) if torch.cuda.device_count() > 1 else base
        base_ref = model.module if isinstance(model, nn.DataParallel) else model

        # EMA
        ema_model = None
        if cfg.use_ema:
            ema_model = deepcopy(base).to(device)
            for p in ema_model.parameters():
                p.requires_grad_(False)
            print(f"✅ EMA enabled (decay={cfg.ema_decay})")

        # SWA
        swa_model, swa_start = None, None
        if cfg.use_swa:
            swa_model = torch.optim.swa_utils.AveragedModel(base).to(device)
            swa_start = int((1.0 - cfg.swa_pct) * cfg.epochs)
            print(f"✅ SWA enabled (starts at epoch {swa_start + 1})")

        # Loss
        pos_w = effective_num_pos_weight(ytr)
        print(f"⚖  pos_weight = {pos_w:.3f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
        afl = AsymmetricFocalLoss() if cfg.use_afl else None
        if cfg.use_afl:
            print("🎯 Using Asymmetric Focal Loss")

        # Optim & Schedulers
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        total_steps = len(tr_dl) * cfg.epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=cfg.lr,
            total_steps=total_steps,
            pct_start=cfg.warmup_ratio,
            anneal_strategy="cos"
        )
        swa_sched = None
        if cfg.use_swa:
            swa_sched = torch.optim.swa_utils.SWALR(
                opt, anneal_strategy="cos", anneal_epochs=1, swa_lr=(cfg.swa_lr or cfg.lr)
            )

        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        best = {"f1": -1.0, "who": "raw", "epoch": -1, "metrics": None}
        patience = 0

        for ep in range(cfg.epochs):
            # ===== TRAIN =====
            model.train()
            run_loss, n_batches = 0.0, 0
            pbar = tqdm(tr_dl, desc=f"Epoch {ep+1}/{cfg.epochs}", leave=False)

            for xb, yb in pbar:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).unsqueeze(1)

                # Regularizations
                xb = feature_drop(xb, cfg.meta_slice, p=0.15)
                if cfg.mixup_alpha > 0:
                    xb, yb = mixup(xb, yb, alpha=cfg.mixup_alpha)
                xb = xb + 0.01 * torch.randn_like(xb)

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(xb)
                    loss = (afl or criterion)(logits, yb)

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
                scaler.step(opt); scaler.update()

                # EMA update before LR step
                if cfg.use_ema:
                    ema_update(base_ref, ema_model, mu=cfg.ema_decay)

                # Step OneCycle unless in SWA phase
                if not (cfg.use_swa and ep >= swa_start):
                    scheduler.step()

                run_loss += loss.item(); n_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # SWA update after epoch (SWA phase)
            if cfg.use_swa and ep >= swa_start:
                swa_model.update_parameters(base_ref)
                swa_sched.step()

            avg_loss = run_loss / max(1, n_batches)

            # ===== VALIDATE (choose best among EMA/SWA/Raw) =====
            candidates = []
            if cfg.use_ema:
                candidates.append(("ema", ema_model))
            if cfg.use_swa and ep >= swa_start:
                candidates.append(("swa", swa_model))
            candidates.append(("raw", base_ref))

            best_ep = None
            for tag, m in candidates:
                out = eval_on_loader(m, va_dl, device)
                f1 = out["metrics"]["f1"]
                if (best_ep is None) or (f1 > best_ep["metrics"]["f1"]):
                    best_ep = {"who": tag, "metrics": out["metrics"]}

            metrics = best_ep["metrics"]
            current_lr = opt.param_groups[0]["lr"]
            print(f"Epoch {ep+1:02d} | loss={avg_loss:.4f} | "
                  f"F1={metrics['f1']:.4f} | AUC={metrics['auc']:.4f} | "
                  f"Acc={metrics['acc']:.4f} | Prec={metrics['prec']:.4f} | Rec={metrics['rec']:.4f} | "
                  f"Thr={metrics['threshold']:.3f} | LR={current_lr:.6f} | best={best_ep['who']}")

            # Save best checkpoint
            if metrics["f1"] > best["f1"] + 1e-6:
                best.update({
                    "f1": metrics["f1"],
                    "who": best_ep["who"],
                    "epoch": ep + 1,
                    "metrics": metrics
                })
                patience = 0

                save_dir = os.path.join(cfg.save_dir, court)
                os.makedirs(save_dir, exist_ok=True)

                to_save = ema_model if best["who"] == "ema" else (swa_model if best["who"] == "swa" else base_ref)
                ckpt = {
                    "model_state_dict": to_save.state_dict(),
                    "best_source": best["who"],
                    "metrics": best["metrics"],
                    "epoch": best["epoch"],
                    "fold": fold,
                    "config": asdict(cfg)
                }
                ckpt_path = os.path.join(save_dir, f"{court}_fold{fold}.pt")
                torch.save(ckpt, ckpt_path)
                with open(os.path.join(save_dir, f"{court}_fold{fold}_metrics.json"), "w") as f:
                    json.dump(best["metrics"], f, indent=2)
                print(f"✅ Saved best ({best['who']}) — F1={best['f1']:.4f} @ epoch {best['epoch']}")

            else:
                patience += 1
                if patience >= cfg.early_stop_patience:
                    print(f"⏹  Early stopping (patience={patience})")
                    break

        # Fold summary
        print(f"\n📈 Fold {fold} — Best F1={best['f1']:.4f} (from {best['who']} @ epoch {best['epoch']})")
        print(f"   Metrics: {best['metrics']}")
        fold_results.append(best["metrics"])

    # Court summary
    print(f"\n{'='*80}")
    print(f"✅ {court.upper()} Training Complete")
    print(f"{'='*80}")
    avg_f1 = float(np.mean([r["f1"] for r in fold_results]))
    std_f1 = float(np.std([r["f1"] for r in fold_results]))
    avg_auc = float(np.mean([r.get("auc", 0.0) for r in fold_results]))
    std_auc = float(np.std([r.get("auc", 0.0) for r in fold_results]))
    avg_acc = float(np.mean([r["acc"] for r in fold_results]))
    print(f"F1   : {avg_f1:.4f} ± {std_f1:.4f}")
    print(f"AUC  : {avg_auc:.4f} ± {std_auc:.4f}")
    print(f"Acc  : {avg_acc:.4f}")
    print("Per-fold F1:", [f"{r['f1']:.4f}" for r in fold_results])

    return fold_results


# =========================
# ORCHESTRATOR
# =========================
def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*80}")
    print("🚀 Court-MOE Expert Training System — Final")
    print(f"{'='*80}")
    print(f"Device: {device} | GPUs: {torch.cuda.device_count()} | Workers/loader: {CONFIG['num_workers']}")
    print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {p.name} ({p.total_memory/1e9:.1f} GB)")

    # Validate files
    print(f"\n{'─'*80}\nValidating embedding files...\n{'─'*80}")
    for court, path in CONFIG["embeddings"].items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✅ {court:10s} -> {path} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {court:10s} -> {path} (MISSING)")

    # Train all experts
    results_all: Dict[str, List[Dict[str, Any]]] = {}

    for court in ["supreme", "high", "tribunal", "district", "daily"]:
        try:
            in_dim = 768 if court in ["supreme", "high", "tribunal"] else 777
            meta_slice = None if in_dim == 768 else (768, 777)
            emb_path = CONFIG["embeddings"][court]
            if not os.path.exists(emb_path):
                print(f"\n⚠  Skipping {court}: file not found → {emb_path}")
                continue

            X, y = load_embeddings(emb_path, court)
            cfg = TrainCfg(court=court, in_dim=in_dim, meta_slice=meta_slice)
            results = train_expert(court, X, y, cfg, device)
            results_all[court] = results

        except Exception as e:
            print(f"\n❌ Error training {court}: {e}")
            import traceback; traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print("🎉 ALL COURTS — TRAINING COMPLETE")
    print(f"{'='*80}")
    if results_all:
        print(f"\n{'Court':<12} {'Avg F1':<10} {'Avg AUC':<10} {'Avg Acc':<10}")
        print("-"*44)
        for court, folds in results_all.items():
            avg_f1 = float(np.mean([r["f1"] for r in folds]))
            avg_auc = float(np.mean([r.get("auc", 0.0) for r in folds]))
            avg_acc = float(np.mean([r["acc"] for r in folds]))
            print(f"{court.upper():<12} {avg_f1:<10.4f} {avg_auc:<10.4f} {avg_acc:<10.4f}")

        os.makedirs(CONFIG["save_dir"], exist_ok=True)
        with open(os.path.join(CONFIG["save_dir"], "training_summary.json"), "w") as f:
            json.dump(results_all, f, indent=2)
        print(f"\n📄 Summary saved → {os.path.join(CONFIG['save_dir'], 'training_summary.json')}\n")
    else:
        print("⚠  No courts were trained.\n")

if __name__ == "__main__":
    main()