#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Court-MOE Inference (Multi-GPU Parallelization + Top-K Routing)

✔ Router: SE-Residual (773-dim input; 4 blocks)
✔ Experts: 3-fold ensembles, 768/777 dims
✔ AMP + tqdm + diagnostics + Top-K routing
✔ Multi-GPU across GPU 1 & 2 (skips GPU 0)
"""

import os, sys, json, torch, torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Tuple

# =========================================================
# === Router architecture =================================
# =========================================================
class SEBlock(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(1, dim // reduction)
        self.fc = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, dim), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x.mean(dim=0, keepdim=True))
        return x * w

class BiasOnly(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.bias   = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        return x * self.weight + self.bias

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
        out = self.fc0(x)
        out = self.act(out)
        out = self.fc2(out)
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

# =========================================================
# === Expert model ========================================
# =========================================================
class ExpertMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)

# =========================================================
# === Loaders =============================================
# =========================================================
def _infer_in_dim_from_state_dict(sd: Dict[str, torch.Tensor]) -> int:
    for k,v in sd.items():
        if "weight" in k and v.ndim==2: return v.shape[1]
    return 768

def load_router(device_ids):
    path = "routers/router_meta_boosted_61.64/best_router.pt"
    ckpt = torch.load(path, map_location="cpu")
    model = SERouterTrue()
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model = nn.DataParallel(model, device_ids=device_ids)   # ✅ Multi-GPU
    model.to(f"cuda:{device_ids[0]}")
    model.eval()
    print(f"✅ Loaded router (GPUs {device_ids}) input=773")
    return model

def load_expert_ensemble(court: str, device_ids):
    base = f"Experts/experts_kfold/{court}"
    folds = [f"{base}/{court}_fold{i}.pt" for i in (1,2,3)]
    models=[]
    for f in folds:
        sd = torch.load(f, map_location="cpu")
        state = sd.get("model", sd)

        # ✅ Force correct input dims for District/Daily
        if court in ["district", "daily"]:
            in_dim = 777
        else:
            in_dim = _infer_in_dim_from_state_dict(state)

        m = ExpertMLP(in_dim)
        try:
            m.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"⚠️ Warning while loading {court} model: {e}")
        m = nn.DataParallel(m, device_ids=device_ids)
        m.to(f"cuda:{device_ids[0]}")
        m.eval()
        models.append(m)

    print(f"✅ Loaded {court} ensemble ({len(models)} folds, in_dim={in_dim})")
    return models, in_dim
# =========================================================
# === Helpers =============================================
# =========================================================
def _pad_to_dim(x:torch.Tensor,want:int)->torch.Tensor:
    if x.shape[1]==want: return x
    pad=want-x.shape[1]
    if pad<0: raise ValueError(f"Input {x.shape[1]} > expected {want}")
    return torch.cat([x,torch.zeros(x.size(0),pad,device=x.device,dtype=x.dtype)],1)

@torch.inference_mode()
def ensemble_predict(models:List[nn.Module],x:torch.Tensor)->torch.Tensor:
    probs=[torch.sigmoid(m(x)).squeeze(-1) for m in models]
    return torch.stack(probs,0).mean(0)

# =========================================================
# === Inference (Top-K) ===================================
# =========================================================
@torch.inference_mode()
def predict_batch(embeddings,router,experts,device_ids,top_k=1,batch_size=4096):
    device = f"cuda:{device_ids[0]}"
    court_idx2name={0:"supreme",1:"high",2:"tribunal",3:"district",4:"daily"}
    results={"router_court":[], "expert_prob":[], "expert_label":[]}

    for i in tqdm(range(0,len(embeddings),batch_size),desc="🔁 Routing batches"):
        x=embeddings[i:i+batch_size].to(device).float()
        x_router=torch.cat([x,torch.zeros(x.size(0),5,device=device)],1)
        with torch.amp.autocast(device_type="cuda",enabled=True):
            router_logits=router(x_router)
            probs=torch.softmax(router_logits,1)
            topk_probs,topk_idx=torch.topk(probs,k=top_k,dim=1)

        for k_rank in range(topk_idx.shape[1]):
            ids=topk_idx[:,k_rank]
            for cid,cname in court_idx2name.items():
                mask=(ids==cid)
                if not mask.any(): continue
                x_court=x[mask]
                models,want_dim=experts[cname]
                if cname in ["district","daily"]: want_dim=777
                x_court=_pad_to_dim(x_court,want_dim)
                with torch.amp.autocast(device_type="cuda",enabled=True):
                    probs_e=ensemble_predict(models,x_court)
                weight=topk_probs[mask,k_rank].unsqueeze(1)
                weighted=probs_e*weight.squeeze()
                labels=(weighted>=0.5).int()
                for p,l in zip(weighted.tolist(),labels.tolist()):
                    results["router_court"].append(cname)
                    results["expert_prob"].append(round(p,4))
                    results["expert_label"].append(int(l))
    return results

# =========================================================
# === Diagnostics =========================================
# =========================================================
@torch.inference_mode()
def inspect_router_distribution(router,embeddings,device_ids):
    device=f"cuda:{device_ids[0]}"
    n=min(5000,len(embeddings))
    x=embeddings[:n].to(device)
    x_router=torch.cat([x,torch.zeros(n,5,device=device)],1)
    with torch.amp.autocast(device_type="cuda",enabled=True):
        probs=torch.softmax(router(x_router),1).mean(0)
    print("\n🔍 Mean Router Probabilities (Supreme, High, Tribunal, District, Daily):")
    print([round(v.item(),4) for v in probs])

# =========================================================
# === Main ================================================
# =========================================================
def main(top_k: int = 1):
    # ✅ Use only GPUs 1 & 2
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    device_ids = [0, 1]   # inside visible device list, maps to physical GPU1 & GPU2
    device = f"cuda:{device_ids[0]}"
    print(f"🚀 Using GPUs {device_ids} (physical 1 & 2)")

    # === Load router and experts ===
    router = load_router(device_ids)
    experts = {c: load_expert_ensemble(c, device_ids)
               for c in ["supreme", "high", "tribunal", "district", "daily"]}

    # === Load embeddings ===
    emb_path = "encoding/encoded_output_final/encoded_cases_final_balanced_minority.pth"
    print(f"📦 Loading embeddings from: {emb_path}")
    data = torch.load(emb_path, map_location="cpu")

    # --- Robust flattening for list[list[dict]] or other structures ---
    if isinstance(data, dict) and "embeddings" in data:
        emb = data["embeddings"]
        embeddings = emb if torch.is_tensor(emb) else torch.tensor(emb)
    elif isinstance(data, list):
        flattened = []
        for block in data:
            if isinstance(block, list):
                for item in block:
                    if isinstance(item, dict) and "embeddings" in item:
                        emb = item["embeddings"]
                        flattened.append(torch.tensor(emb) if not torch.is_tensor(emb) else emb)
                    elif torch.is_tensor(item):
                        flattened.append(item)
            elif isinstance(block, dict) and "embeddings" in block:
                emb = block["embeddings"]
                flattened.append(torch.tensor(emb) if not torch.is_tensor(emb) else emb)
            elif torch.is_tensor(block):
                flattened.append(block)
        embeddings = torch.stack(flattened)
        print(f"✅ Flattened {len(flattened)} embedding records from nested list.")
    elif torch.is_tensor(data):
        embeddings = data
    else:
        raise ValueError("Unsupported .pth structure — expected dict, list, or tensor.")

    embeddings = embeddings.float()
    print(f"✅ Loaded embeddings: shape = {embeddings.shape}")

    # === Router diagnostics ===
    inspect_router_distribution(router, embeddings, device_ids)

    # === Prediction ===
    results = predict_batch(embeddings, router, experts, device_ids, top_k=top_k)

    # === Summary ===
    courts = ["supreme", "high", "tribunal", "district", "daily"]
    counts = {c: 0 for c in courts}
    for c in results["router_court"]:
        counts[c] += 1
    print(f"\n=== Routing Distribution (Top-{top_k}) ===")
    for c in courts:
        print(f"{c:<9}: {counts[c]}")

    print("\n=== Sample Outputs (first 10) ===")
    for i in range(min(10, len(results["router_court"]))):
        print({
            "court": results["router_court"][i],
            "prob_accept": results["expert_prob"][i],
            "label": results["expert_label"][i],
        })

    # === Save results ===
    os.makedirs("results", exist_ok=True)
    torch.save(results, "results/inference_results.pt")
    with open("results/inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n💾 Results saved to results/inference_results.pt and .json")
# =========================================================
# === Entry-point =========================================
# =========================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_k", type=int, default=1,
                    help="Top-K experts to use (1 = hard routing)")
    args = ap.parse_args()
    main(top_k=args.top_k)
