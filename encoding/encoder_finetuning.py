import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL = "nlpaueb/legal-bert-base-uncased"   
DATA_PATH = "encoding/encoded_output_final/aggregated_doc_embeddings_fixed.pth"
SAVE_DIR = "legalbert_finetuned_courts"

EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5
MAX_LEN = 256
WEIGHT_DECAY = 0.01
SEED = 42

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
encoder = AutoModel.from_pretrained(BASE_MODEL)

for name, param in encoder.named_parameters():
    if any(f"layer.{i}." in name for i in range(8)):
        param.requires_grad = False


print("[INFO] Loading dataset for supervised fine-tuning...")
data = torch.load(DATA_PATH, map_location="cpu")

texts = []
labels = []
for d in data:
   
    if "text" in d:
        texts.append(d["text"])
    else:
        texts.append(f"Case ID {d['case_id']}")  
    labels.append(int(d["court_type_idx"]))

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
num_classes = len(np.unique(labels))
print(f"[INFO] Fine-tuning on {len(texts)} documents across {num_classes} courts.")


class LegalDataset(Dataset):
    def __init__(self, docs, labels, tokenizer, max_len):
        self.docs = docs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.docs)
    def __getitem__(self, idx):
        text = str(self.docs[idx])[:4000]
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

dataset = LegalDataset(texts, labels, tokenizer, MAX_LEN)


labels_tensor = torch.tensor(labels)
class_counts = torch.bincount(labels_tensor)
weights = 1.0 / (class_counts.float() + 1e-6)
sample_weights = weights[labels_tensor]
sampler = WeightedRandomSampler(sample_weights.numpy(), num_samples=len(sample_weights), replacement=True)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)


class LegalBERTClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.encoder.config.hidden_size, num_classes)
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.fc(self.dropout(pooled))

model = LegalBERTClassifier(encoder, num_classes)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[0, 1])

model.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

model.train()
for epoch in range(1, EPOCHS + 1):
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for batch in pbar:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels_b = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels_b)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels_b).sum().item()
        total += labels_b.size(0)
        total_loss += loss.item() * input_ids.size(0)
        pbar.set_postfix({"loss": f"{total_loss/total:.4f}", "acc": f"{correct/total:.3f}"})
    print(f"[E{epoch}] Train Acc={correct/total:.4f} | Loss={total_loss/total:.4f}")


os.makedirs(SAVE_DIR, exist_ok=True)
model.module.encoder.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"[INFO] ✅ Fine-tuned LegalBERT saved to: {SAVE_DIR}")
