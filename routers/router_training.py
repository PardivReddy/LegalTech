import os, random, argparse
from collections import Counter
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ------------------------------------------------------------
# ✅ PyTorch 2.6+ deserialization fix (for NumPy objects)
# ------------------------------------------------------------
from numpy._core import multiarray
torch.serialization.add_safe_globals([multiarray._reconstruct, np.ndarray])

# ------------------------------------------------------------
# AMP compatibility
# ------------------------------------------------------------
try:
    _autocast_ctx = lambda enabled=True: torch.amp.autocast("cuda", enabled=enabled)
    _GradScaler = torch.amp.GradScaler
except AttributeError:
    from torch.cuda import amp as _amp_legacy
    _autocast_ctx = lambda enabled=True: _amp_legacy.autocast(enabled=enabled)
    _GradScaler = _amp_legacy.GradScaler

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    print(f"🔢 Seed set to {seed}")

def l2_normalize(x, eps=1e-12):
    return x / x.norm(p=2, dim=1, keepdim=True).clamp(min=eps)

def compute_probs_entropy(logits, temperature=1.0, eps=1e-12):
    scaled = logits / (temperature + eps)
    probs = torch.softmax(scaled, dim=-1)
    ent = -torch.sum(probs * torch.log(probs + eps), dim=-1)
    return probs, ent

def topk_acc(logits, y, k=(1,2,3)):
    with torch.no_grad():
        maxk = max(k)
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(y.view(1, -1).expand_as(pred))
        res = []
        for kk in k:
            res.append(correct[:kk].reshape(-1).float().mean().mul_(100.0))
        return res

def get_temperature(epoch, total_epochs, T_max=8.0, T_min=0.01):
    t = min(max(epoch, 0), total_epochs)
    cosine = 0.5 * (1.0 + np.cos(np.pi * t / total_epochs))
    return T_min + (T_max - T_min) * cosine

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class EncodedDataset(Dataset):
    def __init__(self, records):
        self.samples = []
        for r in records:
            emb, lbl = r.get("embeddings"), r.get("court_type_idx")
            if emb is None or lbl is None:
                continue
            self.samples.append((np.asarray(emb, np.float32), int(lbl)))
        print(f"📦 Dataset ready: {len(self.samples):,} samples")

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

# ------------------------------------------------------------
# Router model (augmented 773-dim input)
# ------------------------------------------------------------
class RouterBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2, drop_path_rate=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.se = nn.Sequential(
            nn.Linear(out_dim, max(8, out_dim // 4)),
            nn.ReLU(),
            nn.Linear(max(8, out_dim // 4), out_dim),
            nn.Sigmoid()
        )
        self.drop_path_rate = drop_path_rate

    def forward(self, x):
        h = self.fc(x)
        h = h * self.se(h)
        if self.training and torch.rand(1, device=h.device) < self.drop_path_rate:
            return h
        return h + 0.2 * self.proj(x)

class RouterMLP(nn.Module):
    def __init__(self, in_dim=773, num_classes=5, dropout=0.2, droppath=0.1):
        super().__init__()
        self.in_ln = nn.LayerNorm(in_dim)
        self.block1 = RouterBlock(in_dim, 1024, dropout, droppath)
        self.block2 = RouterBlock(1024, 768, dropout, droppath)
        self.block3 = RouterBlock(768, 512, dropout, droppath)
        self.block4 = RouterBlock(512, 256, dropout, droppath)
        self.out = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.in_ln(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.out(x)

# ------------------------------------------------------------
# Mixup
# ------------------------------------------------------------
def mixup_embeddings(x, y, alpha=0.2):
    if alpha <= 0: return x, None
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, (y, y[idx], lam)

# ------------------------------------------------------------
# Train / Validate
# ------------------------------------------------------------
def train_epoch(model, loader, opt, device, scaler, loss_fn, T, entropy_lambda, mixup_alpha):
    model.train()
    total_loss, logits_cat, y_cat, ent_vals = 0, [], [], []
    pbar = tqdm(loader, ncols=120, desc="Train")
    for X, y in pbar:
        X, y = X.to(device, torch.float32, non_blocking=True), y.to(device, non_blocking=True)
        X = l2_normalize(X)
        X_mix, mix_info = mixup_embeddings(X, y, mixup_alpha)
        opt.zero_grad(set_to_none=True)
        with _autocast_ctx(True):
            logits = model(X_mix)
            if mix_info:
                ya, yb, lam = mix_info
                loss = lam * loss_fn(logits, ya) + (1 - lam) * loss_fn(logits, yb)
            else:
                loss = loss_fn(logits, y)
            probs, ent = compute_probs_entropy(logits, T)
            loss = loss + entropy_lambda * ent.mean()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(opt); scaler.update()

        total_loss += loss.item() * X.size(0)
        logits_cat.append(logits.detach().cpu())
        y_cat.append(y.cpu())
        ent_vals.append(ent.detach().cpu().numpy())

        acc_now = (torch.cat(logits_cat).argmax(-1) == torch.cat(y_cat)).float().mean().item() * 100.0
        ent_mean = float(np.mean(np.concatenate(ent_vals))) if len(ent_vals) > 0 else 0.0
        pbar.set_postfix(loss=f"{total_loss/len(loader.dataset):.4f}",
                         acc=f"{acc_now:.2f}%", ent=f"{ent_mean:.3f}")

    logits_all = torch.cat(logits_cat); y_all = torch.cat(y_cat)
    acc1, acc2, acc3 = topk_acc(logits_all, y_all)
    return total_loss/len(loader.dataset), acc1.item(), (acc2.item(), acc3.item())

@torch.no_grad()
def validate_epoch(model, loader, device, loss_fn, T):
    model.eval()
    total_loss, logits_cat, y_cat = 0, [], []
    for X, y in tqdm(loader, ncols=120, desc="Val"):
        X, y = X.to(device, torch.float32, non_blocking=True), y.to(device, non_blocking=True)
        X = l2_normalize(X)
        logits = model(X)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * X.size(0)
        logits_cat.append(logits.cpu()); y_cat.append(y.cpu())
    logits_all = torch.cat(logits_cat); y_all = torch.cat(y_cat)
    acc1, acc2, acc3 = topk_acc(logits_all, y_all)
    return total_loss/len(loader.dataset), acc1.item(), (acc2.item(), acc3.item())

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="encoding/metadata_augmented_district_daily.pth")
    p.add_argument("--out_dir", default="routers/router_meta_boosted")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--max_lr", type=float, default=7e-4)
    p.add_argument("--mixup_alpha", type=float, default=0.2)
    p.add_argument("--label_smoothing", type=float, default=0.03)
    p.add_argument("--entropy_lambda", type=float, default=2e-4)
    p.add_argument("--temp_init", type=float, default=8.0)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using {torch.cuda.device_count()} GPU(s)")

    # ---- load augmented embeddings safely ----
    records = torch.load(args.data_path, map_location="cpu", weights_only=False)
    ds = EncodedDataset(records)
    n = len(ds); val_n = int(0.1*n); tr_n = n - val_n
    tr_ds, val_ds = random_split(ds, [tr_n, val_n], generator=torch.Generator().manual_seed(42))
    num_classes = max(r["court_type_idx"] for r in records) + 1
    print(f"📈 Train/Val → {tr_n:,}/{val_n:,} | Classes={num_classes}")

    tr_labels = [tr_ds[i][1].item() for i in range(tr_n)]
    cnt = Counter(tr_labels)
    cw = torch.tensor([1 / max(cnt.get(i, 1), 1) for i in range(num_classes)], dtype=torch.float32)
    cw = (cw / cw.mean()).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = RouterMLP(in_dim=773, num_classes=num_classes).to(device)
    if torch.cuda.device_count() > 1:
        print(f"⚡ Multi-GPU mode on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    base = model.module if isinstance(model, nn.DataParallel) else model

    opt = AdamW(model.parameters(), lr=args.max_lr, weight_decay=0.02)
    sched = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=args.max_lr/20)
    scaler = _GradScaler(enabled=True)
    swa_start = int(0.67 * args.epochs)
    swa_model = AveragedModel(base)
    swa_sched = SWALR(opt, swa_lr=args.max_lr/2)

    best = 0.0
    for ep in range(1, args.epochs + 1):
        T = get_temperature(ep, args.epochs, T_max=args.temp_init, T_min=0.01)
        print(f"\n===== Epoch {ep}/{args.epochs} | Temp={T:.3f} =====")

        tl, ta, (t2, t3) = train_epoch(model, tr_loader, opt, device, scaler,
                                       loss_fn, T, args.entropy_lambda, args.mixup_alpha)
        vl, va, (v2, v3) = validate_epoch(model, val_loader, device, loss_fn, T)
        print(f"Train Top-1 {ta:.2f}% | Val Top-1 {va:.2f}% | Top-2 {v2:.2f}% | Top-3 {v3:.2f}%")

        if va > best:
            best = va
            os.makedirs(args.out_dir, exist_ok=True)
            torch.save(base.state_dict(), f"{args.out_dir}/best_router.pt")
            print(f"💾 Saved best model ({best:.2f}%)")

        if ep >= swa_start:
            swa_model.update_parameters(base)
            swa_sched.step()
        else:
            sched.step()

    print("\n🔁 Finalizing SWA weights ...")
    update_bn(tr_loader, swa_model, device=device)
    torch.save(swa_model.state_dict(), f"{args.out_dir}/swa_router.pt")
    print(f"✅ Training complete | Best Val Acc {best:.2f}%")

if __name__ == "__main__":
    main()
