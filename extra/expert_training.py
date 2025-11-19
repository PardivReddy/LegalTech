import os, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModel, get_cosine_schedule_with_warmup
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix
from torch.optim.swa_utils import AveragedModel
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_ENCODER_PATH = "encoding/legalbert_finetuned_courts"

DATA_PATHS = {
    "supreme":  "encoding/encoded_output_final/final_balanced_by_court/SupremeCourt_embeddings_final.pth",
    "high":     "encoding/encoded_output_final/final_balanced_by_court/HighCourt_embeddings_final.pth",
    "district": "encoding/encoded_output_final/final_balanced_by_court/DistrictCourt_embeddings_final.pth",
    "tribunal": "encoding/encoded_output_final/final_balanced_by_court/TribunalCourt_embeddings_final.pth",
    "daily":    "encoding/encoded_output_final/final_balanced_by_court/DailyOrderCourt_embeddings_final.pth"
}

SAVE_DIR = "experts_finetuned"
os.makedirs(SAVE_DIR, exist_ok=True)

LOG_FILE = "expert_training_metrics1.txt"
with open(LOG_FILE, "a") as f:
    f.write(f"\n=================== Training Run: {datetime.now()} ===================\n")

EPOCHS = 30
BATCH_SIZE = 128
LR = 2e-4
META_DIM = 5
EMB_DIM = 768
FP16 = True
VAL_RATIO = 0.15
LABEL_MAP = {"Accepted": 0, "Rejected": 1}
FOCAL_GAMMA = 2.0

# ============================================================
# FOCAL LOSS
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_factor = (1 - pt).pow(self.gamma)
        return (focal_factor * bce_loss).mean()

# ============================================================
# MODEL DEFINITIONS
# ============================================================

class SEBlock(nn.Module):
    def __init__(self, dim, reduction=32):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.fc2 = nn.Linear(dim // reduction, dim)
        self.act = nn.GELU(); self.sig = nn.Sigmoid()
    def forward(self, x):
        w = self.sig(self.fc2(self.act(self.fc1(x))))
        return x * w

class SEResidualMLP(nn.Module):
    def __init__(self, dim=EMB_DIM, hidden=2048, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.se = SEBlock(dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        res = x
        x = self.drop(self.act(self.bn1(self.fc1(x))))
        x = self.drop(self.fc2(x))
        x = self.se(x)
        return self.norm(x + res)

class LegalBERTExpert(nn.Module):
    def __init__(self, base_model_path, use_metadata=True, meta_dim=META_DIM, unfreeze_layers=2):
        super().__init__()
        print(f"🔹 Loading LegalBERT base from {base_model_path} ...")
        self.encoder = AutoModel.from_pretrained(base_model_path)
        self.dim = EMB_DIM

        # 🔓 Unfreeze last 'unfreeze_layers' transformer blocks
        for param in self.encoder.parameters(): param.requires_grad = False
        for block in self.encoder.encoder.layer[-unfreeze_layers:]:
            for param in block.parameters(): param.requires_grad = True
        print(f"✅ Unfrozen last {unfreeze_layers} layers for fine-tuning.\n")

        self.encoder_proj = nn.Linear(self.dim, self.dim)
        self.use_metadata = use_metadata

        if use_metadata:
            self.meta_proj = nn.Sequential(
                nn.Linear(meta_dim, 64), nn.GELU(),
                nn.Linear(64, 128), nn.LayerNorm(128)
            )
            combined = self.dim + 128
        else:
            combined = self.dim

        self.se_mlp = SEResidualMLP(dim=self.dim)
        self.classifier = nn.Sequential(
            nn.Linear(combined, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
        print("✅ Expert initialized with SE-Residual head, BatchNorm, and Focal Loss.\n")

    def forward(self, embeddings, metadata=None):
        x = self.encoder_proj(embeddings)
        x = self.se_mlp(x)
        if self.use_metadata and metadata is not None:
            m = self.meta_proj(metadata)
            x = torch.cat([x, m], dim=-1)
        return self.classifier(x).squeeze(-1)

# ============================================================
# DATASET
# ============================================================

class CourtDataset(Dataset):
    def __init__(self, data_path):
        print(f"📦 Loading dataset from: {data_path}")
        data = torch.load(data_path, map_location="cpu")
        embeddings_list, labels_list, metadata_list = [], [], []
        for i, d in enumerate(data):
            try:
                emb = torch.tensor(d["embeddings"], dtype=torch.float32)
                if emb.ndim != 1: continue
                label_val = LABEL_MAP.get(str(d["label"]).strip(), -1)
                if label_val not in [0, 1]: continue
                embeddings_list.append(emb)
                labels_list.append(int(label_val))
                meta = torch.tensor(d.get("metadata", torch.zeros(META_DIM)), dtype=torch.float32)
                metadata_list.append(meta)
            except Exception as e:
                print(f"⚠️ Skipping sample {i}: {e}")
        self.embeddings = torch.stack(embeddings_list)
        self.labels = torch.tensor(labels_list, dtype=torch.float)
        self.metadata = torch.stack(metadata_list)
        print(f"✅ Loaded {len(self.embeddings)} samples | Emb dim={self.embeddings.shape[1]} | Meta dim={self.metadata.shape[1]}\n")

    def __len__(self): return len(self.embeddings)
    def __getitem__(self, idx):
        return {
            "embeddings": self.embeddings[idx],
            "labels": self.labels[idx],
            "metadata": self.metadata[idx]
        }

# ============================================================
# HELPERS
# ============================================================

def stratified_split(dataset, val_ratio=0.15, seed=42):
    labels = dataset.labels.numpy().astype(int)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds   = torch.utils.data.Subset(dataset, val_idx)
    print(f"📊 Stratified Split | train={len(train_ds)} | val={len(val_ds)}")
    return train_ds, val_ds

def log_to_file(text: str):
    with open(LOG_FILE, "a") as f: f.write(text + "\n")

# ============================================================
# TRAINING & EVAL
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train(); total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad(set_to_none=True)
        emb = batch["embeddings"].to(DEVICE)
        meta = batch["metadata"].to(DEVICE)
        y = batch["labels"].to(DEVICE)
        emb = torch.nn.functional.normalize(emb, dim=1)
        with torch.amp.autocast("cuda", enabled=FP16):
            logits = model(emb, meta)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            emb = batch["embeddings"].to(DEVICE)
            meta = batch["metadata"].to(DEVICE)
            y = batch["labels"].to(DEVICE)
            emb = torch.nn.functional.normalize(emb, dim=1)
            logits = model(emb, meta)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    acc = (y_true == y_pred).mean()
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    print(f"🧾 Confusion Matrix:\n{cm}")
    print(f"📊 Acc={acc:.4f} | F1={f1:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | MCC={mcc:.4f}\n")
    return acc, f1, prec, rec, mcc, cm

# ============================================================
# MAIN TRAIN LOOP
# ============================================================

def train_expert(court_name, data_path):
    print(f"\n🚀 Training {court_name.title()} Expert (Fine-Tuned LegalBERT)\n{'='*70}")
    log_to_file(f"\n=== {court_name.title()} Expert Training ===")

    dataset = CourtDataset(data_path)
    train_ds, val_ds = stratified_split(dataset, VAL_RATIO)
    labels_train = dataset.labels[train_ds.indices]
    class_counts = torch.bincount(labels_train.long(), minlength=2).float()
    pos_weight = torch.tensor([class_counts[0]/(class_counts[1]+1e-6)], dtype=torch.float)
    log_to_file(f"Class counts: {class_counts.tolist()} | pos_weight={pos_weight.item():.3f}")

    sample_weights = torch.where(labels_train==1, class_counts[0], class_counts[1])
    sampler = WeightedRandomSampler(weights=sample_weights.double(), num_samples=len(labels_train), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = LegalBERTExpert(base_model_path=BASE_ENCODER_PATH, use_metadata=True, meta_dim=META_DIM)
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)
        print(f"✅ Using {torch.cuda.device_count()} GPUs\n")
    model.to(DEVICE)

    criterion = FocalLoss(gamma=FOCAL_GAMMA, pos_weight=pos_weight.to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-5)
    total_steps = len(train_loader)*EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, 500, total_steps)
    swa_model = AveragedModel(model)
    swa_start = int(EPOCHS*0.7)
    scaler = torch.amp.GradScaler("cuda")

    best_acc = 0
    for epoch in range(EPOCHS):
        print(f"\n🧠 Epoch {epoch+1}/{EPOCHS}")
        loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        scheduler.step()
        if epoch >= swa_start: swa_model.update_parameters(model)
        acc, f1, prec, rec, mcc, cm = evaluate(model, val_loader)
        log_line = (f"{court_name.title()} | Epoch {epoch+1} | "
                    f"Loss={loss:.4f} | Acc={acc:.4f} | F1={f1:.4f} | "
                    f"Prec={prec:.4f} | Rec={rec:.4f} | MCC={mcc:.4f}")
        log_to_file(log_line)

        if acc > best_acc:
            best_acc = acc
            path = os.path.join(SAVE_DIR, f"{court_name}_expert.pt")
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), path)
            log_to_file(f"💾 Saved best model for {court_name} → {path}")

    log_to_file(f"🏁 {court_name.title()} Expert finished | Best Acc={best_acc:.4f}")
    print(f"🏁 {court_name.title()} Expert training done | Best Acc={best_acc:.4f}\n{'='*70}")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("🔥 Starting Fine-Tuned LegalBERT Expert Training (GPUs 1 & 2)...\n")
    for court_name, path in DATA_PATHS.items():
        train_expert(court_name, path)
    print(f"\n🎯 All experts trained successfully.\nMetrics logged in: {LOG_FILE}")
