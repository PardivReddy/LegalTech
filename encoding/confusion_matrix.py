import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from numpy._core import multiarray
import os


torch.serialization.add_safe_globals([multiarray._reconstruct, np.ndarray])

class EncodedDataset(Dataset):
    def __init__(self, records):
        self.samples = []
        for r in records:
            emb, lbl = r.get("embeddings"), r.get("court_type_idx")
            if emb is not None and lbl is not None:
                self.samples.append((np.asarray(emb, np.float32), int(lbl)))
        print(f"📦 Dataset ready: {len(self.samples):,} samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


class RouterBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(out_dim),
            torch.nn.Dropout(dropout)
        )
        self.proj = torch.nn.Linear(in_dim, out_dim) if in_dim != out_dim else torch.nn.Identity()
        self.se = torch.nn.Sequential(
            torch.nn.Linear(out_dim, max(8, out_dim // 4)),
            torch.nn.ReLU(),
            torch.nn.Linear(max(8, out_dim // 4), out_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        h = self.fc(x)
        h = h * self.se(h)
        return h + 0.2 * self.proj(x)

class RouterMLP(torch.nn.Module):
    def __init__(self, in_dim=773, num_classes=5):
        super().__init__()
        self.in_ln = torch.nn.LayerNorm(in_dim)
        self.block1 = RouterBlock(in_dim, 1024)
        self.block2 = RouterBlock(1024, 768)
        self.block3 = RouterBlock(768, 512)
        self.block4 = RouterBlock(512, 256)
        self.out = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.in_ln(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.out(x)

def main():
    data_path = "encoding/metadata_augmented_district_daily.pth"
    model_path = "routers/router_meta_boosted/best_router.pt"
    save_path = "routers/router_meta_boosted/confusion_matrix.png"

    print(f"🔍 Loading data from: {data_path}")
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    ds = EncodedDataset(data)
    loader = DataLoader(ds, batch_size=1024, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🧠 Loading router from: {model_path}")
    model = RouterMLP(in_dim=773, num_classes=5).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print("⚙️ Running inference...")
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in tqdm(loader):
            X = X.to(device, torch.float32)
            logits = model(X)
            preds = logits.argmax(dim=-1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())

    print("✅ Inference complete!")
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    labels = ["Supreme Court", "High Court", "District Court", "Tribunal", "Daily Orders"]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=labels, yticklabels=labels)
    plt.title("Court Routing Confusion Matrix")
    plt.xlabel("Predicted Court")
    plt.ylabel("True Court")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Confusion matrix saved at: {save_path}")

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels, digits=3))

if __name__ == "__main__":
    main()
