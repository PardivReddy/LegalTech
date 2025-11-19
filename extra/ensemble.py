#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Court-MOE — K-Fold Ensemble Inference (final stable)
"""
import os, json, torch, numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, matthews_corrcoef
from datetime import datetime
# ---- PyTorch ≥ 2.6 safe-unpickling ----
from numpy._core import multiarray
torch.serialization.add_safe_globals([multiarray._reconstruct, np.ndarray, dict, list])

# ============================================================
# CONFIG
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

EXPERT_DIR="/home/infodna/Court-MOE/Experts/experts_kfold"
SAVE_DIR="ensemble_results"; os.makedirs(SAVE_DIR,exist_ok=True)
AUG_PATH="/home/infodna/Court-MOE/encoding/metadata_augmented_v2_district_daily.pth"
DATA_PATHS_768={
 "supreme":"/home/infodna/Court-MOE/encoding/encoded_output_final/final_balanced_by_court/SupremeCourt_embeddings_final.pth",
 "high":"/home/infodna/Court-MOE/encoding/encoded_output_final/final_balanced_by_court/HighCourt_embeddings_final.pth",
 "tribunal":"/home/infodna/Court-MOE/encoding/encoded_output_final/final_balanced_by_court/TribunalCourt_embeddings_final.pth"
}
RUN_TAG=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_TXT=os.path.join(SAVE_DIR,f"ensemble_{RUN_TAG}.log")

EXPERTS={
 "supreme":{"dim":768,"folds":3},
 "high":{"dim":768,"folds":3},
 "tribunal":{"dim":768,"folds":3},
 "district":{"dim":777,"folds":3},
 "daily":{"dim":777,"folds":3},
}
EXPERT_HIDDEN={
 "supreme":{"hidden":1536,"dropout":0.40},
 "high":{"hidden":2304,"dropout":0.30},
 "tribunal":{"hidden":1920,"dropout":0.30},
 "district":{"hidden":2400,"dropout":0.36},
 "daily":{"hidden":2560,"dropout":0.40},
}

# ============================================================
# UTILS
# ============================================================
def log_line(s):
    print(s,flush=True)
    with open(LOG_TXT,"a",encoding="utf-8") as f:f.write(s+"\n")
def to_jsonable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,torch.Tensor):return x.detach().cpu().numpy().tolist()
    if isinstance(x,(np.float32,np.float64)):return float(x)
    if isinstance(x,(np.int32,np.int64)):return int(x)
    return x
def set_seed(seed=42):
    import random
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
set_seed(42)

# ============================================================
# MODEL (same as training)
# ============================================================
class SEBlock(nn.Module):
    def __init__(self,dim,reduction=32):
        super().__init__();r=max(8,dim//reduction)
        self.fc1, self.fc2=nn.Linear(dim,r),nn.Linear(r,dim)
        self.act, self.gate=nn.GELU(),nn.Sigmoid()
    def forward(self,x):
        w=self.gate(self.fc2(self.act(self.fc1(x))));return x*w
class SEResidualMLP(nn.Module):
    def __init__(self,dim,hidden,dropout):
        super().__init__()
        self.fc1=nn.Linear(dim,hidden);self.bn1=nn.BatchNorm1d(hidden)
        self.fc2=nn.Linear(hidden,dim);self.act=nn.GELU();self.drop=nn.Dropout(dropout)
        self.se, self.norm=SEBlock(dim),nn.LayerNorm(dim)
    def forward(self,x):
        res=x;x=self.drop(self.act(self.bn1(self.fc1(x))));x=self.drop(self.fc2(x));x=self.se(x)
        return self.norm(x+res)
class ExpertOnEmbeddings(nn.Module):
    def __init__(self,emb_dim,hidden,dropout,use_metadata=True,meta_dim=5):
        super().__init__()
        self.use_metadata=use_metadata;self.proj=nn.Linear(emb_dim,emb_dim)
        self.semlp=SEResidualMLP(emb_dim,hidden,dropout)
        if use_metadata:
            self.meta=nn.Sequential(nn.Linear(meta_dim,64),nn.GELU(),nn.Linear(64,128),nn.LayerNorm(128))
            in_dim=emb_dim+128
        else: in_dim=emb_dim
        self.classifier=nn.Sequential(
            nn.Linear(in_dim,512),nn.BatchNorm1d(512),nn.GELU(),nn.Dropout(0.30),nn.Linear(512,1))
    def forward(self,emb,meta=None):
        x=self.proj(emb);x=self.semlp(x)
        if self.use_metadata and meta is not None:
            m=self.meta(meta);x=torch.cat([x,m],dim=-1)
        return self.classifier(x).squeeze(-1)

# ============================================================
# DATASET
# ============================================================
class MemoryMappedEvalDataset(torch.utils.data.Dataset):
    def __init__(self,path,emb_dim):
        data=torch.load(path,map_location="cpu",weights_only=False)
        embs,labels=[],[]
        for d in data:
            if "embeddings" not in d or "label" not in d:continue
            e=np.asarray(d["embeddings"],np.float32)
            if e.ndim!=1 or e.shape[0]!=emb_dim:continue
            lbl=str(d["label"]).strip().lower()
            if lbl in ("accepted","accept","acc","0","yes"):y=0
            elif lbl in ("rejected","reject","rej","1","no"):y=1
            else:continue
            embs.append(torch.from_numpy(e));labels.append(y)
        if len(embs)==0:
            log_line(f"⚠️ Empty dataset for emb_dim={emb_dim} at {path}")
            self.emb=torch.zeros(0,emb_dim);self.meta=torch.zeros(0,5);self.y=torch.zeros(0)
            return
        self.emb=torch.stack(embs);self.meta=torch.zeros(len(self.emb),5)
        self.y=torch.tensor(labels,dtype=torch.float32)
        log_line(f"✅ Loaded {len(self.emb)} samples ({emb_dim}-dim)")
    def __len__(self):return len(self.y)
    def __getitem__(self,i):return{"emb":self.emb[i],"meta":self.meta[i],"y":self.y[i]}

# ============================================================
# ENSEMBLE
# ============================================================
@torch.no_grad()
def ensemble_predict(models,emb,meta=None):
    preds=[torch.sigmoid(m(emb,meta)) for m in models]
    return torch.mean(torch.stack(preds),dim=0)
def evaluate(models,loader):
    if len(loader.dataset)==0:return {"acc":0,"f1":0,"prec":0,"rec":0,"mcc":0,"thr":0}
    ys,ps=[],[]
    for b in tqdm(loader,leave=False):
        e=b["emb"].to(DEVICE);m=b["meta"].to(DEVICE)
        e=nn.functional.normalize(e,dim=1);p=ensemble_predict(models,e,m)
        ys.append(b["y"].numpy());ps.append(p.cpu().numpy())
    y_true=np.concatenate(ys);y_prob=np.concatenate(ps)
    best_t,best_f1=0.5,-1
    for t in np.linspace(0.3,0.7,41):
        f1=f1_score(y_true,(y_prob>t),average="macro",zero_division=0)
        if f1>best_f1:best_f1,best_t=f1,t
    y_pred=(y_prob>best_t)
    return{
      "thr":float(best_t),
      "acc":float(accuracy_score(y_true,y_pred)),
      "f1":float(best_f1),
      "prec":float(precision_score(y_true,y_pred,average="macro",zero_division=0)),
      "rec":float(recall_score(y_true,y_pred,average="macro",zero_division=0)),
      "mcc":float(matthews_corrcoef(y_true,y_pred)),
    }

# ============================================================
# DRIVER
# ============================================================
def load_expert_models(expert,emb_dim):
    folder=os.path.join(EXPERT_DIR,expert);models=[]
    for i in range(1,4):
        path=os.path.join(folder,f"{expert}_fold{i}.pt")
        if not os.path.exists(path):log_line(f"⚠️ Missing fold: {path}");continue
        cfg=EXPERT_HIDDEN[expert];use_meta=True if emb_dim==768 else False
        model=ExpertOnEmbeddings(emb_dim,cfg["hidden"],cfg["dropout"],use_metadata=use_meta)
        state=torch.load(path,map_location="cuda");model.load_state_dict(state,strict=False)
        model.eval().to(DEVICE);models.append(model)
    log_line(f"📦 Loaded {len(models)} folds for {expert}");return models

def main():
    log_line(f"\n================ Ensemble Run {datetime.now()} ================")
    results={}
    for name,cfg in EXPERTS.items():
        emb_dim=cfg["dim"];log_line(f"\n🚀 Evaluating {name.title()} ({emb_dim}-dim)")
        models=load_expert_models(name,emb_dim)
        if len(models)==0:log_line(f"❌ No folds for {name}");continue
        # dataset selection
        if emb_dim==768:
            ds_path=DATA_PATHS_768.get(name,"")
            if not os.path.exists(ds_path):
                log_line(f"⚠️ Dataset missing for {name}, skipping.")
                continue
            ds=MemoryMappedEvalDataset(ds_path,emb_dim)
        else:
            ds=MemoryMappedEvalDataset(AUG_PATH,emb_dim)
        loader=torch.utils.data.DataLoader(ds,batch_size=128,shuffle=False,num_workers=2,pin_memory=True)
        m=evaluate(models,loader);results[name]=m
        log_line(f"✅ {name.title()} | Acc={m['acc']:.4f} | F1={m['f1']:.4f} | MCC={m['mcc']:.4f}")
    with open(os.path.join(SAVE_DIR,f"ensemble_metrics_{RUN_TAG}.json"),"w") as jf:
        json.dump(results,jf,indent=2,default=to_jsonable)
    log_line("\n🎯 Ensemble complete.")

if __name__=="__main__":main()
