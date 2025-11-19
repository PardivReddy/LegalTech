import os, torch, numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

BASE_ENCODER_PATH = "encoding/legalbert_finetuned_courts"
MODEL_DIR = "experts_legalbert_finetuned"
DATA_PATHS = {
    "supreme":  "encoding/encoded_output_final/final_balanced_by_court/SupremeCourt_embeddings_final.pth",
    "high":     "encoding/encoded_output_final/final_balanced_by_court/HighCourt_embeddings_final.pth",
    "district": "encoding/encoded_output_final/final_balanced_by_court/DistrictCourt_embeddings_final.pth",
    "tribunal": "encoding/encoded_output_final/final_balanced_by_court/TribunalCourt_embeddings_final.pth",
    "daily":    "encoding/encoded_output_final/final_balanced_by_court/DailyOrderCourt_embeddings_final.pth",
}
CM_SAVE_DIR = "experts_confusion_matrices_only"
os.makedirs(CM_SAVE_DIR, exist_ok=True)

LABEL_MAP = {"Accepted": 0, "Rejected": 1}
META_DIM = 5
EMB_DIM = 768

# ============================================================
# MODEL DEFINITION (Same as Training)
# ============================================================

import torch.nn as nn

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
        self.encoder = AutoModel.from_pretrained(base_model_path)
        self.dim = EMB_DIM
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
        data = torch.load(data_path, map_location="cpu")
        embeddings_list, labels_list, metadata_list = [], [], []
        for d in data:
            emb = torch.tensor(d["embeddings"], dtype=torch.float32)
            if emb.ndim != 1:
                continue
            label_val = LABEL_MAP.get(str(d["label"]).strip(), -1)
            if label_val not in [0, 1]:
                continue
            embeddings_list.append(emb)
            labels_list.append(label_val)
            meta = torch.tensor(d.get("metadata", torch.zeros(META_DIM)), dtype=torch.float32)
            metadata_list.append(meta)
        self.embeddings = torch.stack(embeddings_list)
        self.labels = torch.tensor(labels_list, dtype=torch.float)
        self.metadata = torch.stack(metadata_list)
    def __len__(self): return len(self.embeddings)
    def __getitem__(self, idx):
        return {
            "embeddings": self.embeddings[idx],
            "labels": self.labels[idx],
            "metadata": self.metadata[idx]
        }

# ============================================================
# CONFUSION MATRIX GENERATION
# ============================================================

def plot_confusion_matrix(court_name, cm):
    labels = ["Accepted", "Rejected"]
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.title(f"{court_name.title()} Expert - Raw Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(CM_SAVE_DIR, f"{court_name}_raw.png"))
    plt.close()

    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Purples", cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.title(f"{court_name.title()} Expert - Normalized Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(CM_SAVE_DIR, f"{court_name}_normalized.png"))
    plt.close()

# ============================================================
# EVALUATE EACH EXPERT
# ============================================================

def evaluate_expert(court_name, model_path, data_path):
    print(f"\n📊 Generating confusion matrix for {court_name.title()} Expert")
    model = LegalBERTExpert(BASE_ENCODER_PATH).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    dataset = CourtDataset(data_path)
    loader = DataLoader(dataset, batch_size=128)

    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in loader:
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
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    acc = (y_true == y_pred).mean()
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"🧾 Confusion Matrix:\n{cm}")
    print(f"📈 Acc={acc:.4f} | F1={f1:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | MCC={mcc:.4f}\n")
    plot_confusion_matrix(court_name, cm)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    for court in DATA_PATHS.keys():
        model_path = os.path.join(MODEL_DIR, f"{court}_expert.pt")
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found for {court}: {model_path}")
            continue
        evaluate_expert(court, model_path, DATA_PATHS[court])
    print(f"\n✅ Confusion matrices saved in: {CM_SAVE_DIR}")
