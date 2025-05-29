#!/usr/bin/env python3
import os, glob
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

import rater_feedback_utils
from rater_feedback_utils import process_rater_specified_trajectories

# -----------------------------------------
# Load cached features and metadata
# -----------------------------------------
CACHE_DIR = 'cache/val'
cache_files = sorted(glob.glob(os.path.join(CACHE_DIR, 'cache_*.pt')))
records = []
for cf in cache_files:
    data = torch.load(cf)
    N = data['features'].shape[0]
    for i in range(N):
        records.append({
            'feature':    data['features'][i],
            'past':       data['past'][i],
            'intent':     data['intent'][i],
            'future':     data['future'][i],
            'best':       data['best'][i],
            'raw_trajs':  data['raw_trajs'][i],
            'raw_scores': data['raw_scores'][i],
            'init_speed': data['init_speeds'][i],
        })
# Filter only those frames with human labels
labeled = [r for r in records if len(r['raw_scores']) > 0]
print(f"Loaded {len(records)} total records, {len(labeled)} labeled for RFS evaluation.")

# -----------------------------------------
# Build & load model head + intent embedding
# -----------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load('model_final.pth', map_location=device)
# strip prefixes
head_sd = {k.split('head.',1)[1] if k.startswith('head.') else k: v 
           for k,v in ckpt['head_state_dict'].items()}
intent_sd = {k.split('intent_emb.',1)[1] if k.startswith('intent_emb.') else k: v
             for k,v in ckpt['intent_emb_state_dict'].items()}
# feature_dim from checkpoint
feature_dim = ckpt['feature_dim']
# define head-only model
class HeadModel(nn.Module):
    def __init__(self, feature_dim, intent_dim=16, future_len=20):
        super().__init__()
        total = feature_dim + 16*2 + intent_dim  # past always 16x2
        self.head = nn.Sequential(
            nn.Linear(total, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, future_len*2)
        )
        self.future_len = future_len
    def forward(self, feat, past, intent_emb):
        past_flat = past.view(past.size(0), -1)
        x = torch.cat([feat, past_flat, intent_emb], dim=1)
        return self.head(x).view(-1, self.future_len, 2)

# instantiate and load
head_model = HeadModel(feature_dim).to(device).eval()
intent_emb = nn.Embedding(4, 16).to(device)
head_model.head.load_state_dict(head_sd)
intent_emb.load_state_dict(intent_sd)
print('Loaded cached head & intent embedding.')

# -----------------------------------------
# Metrics over all labeled frames
# -----------------------------------------
freq, secs, P = 4, 5, 3
ADEs, RFSs = [], []
transform = T.Compose([T.Resize((128,128)), T.ToTensor()])

for rec in tqdm(labeled, desc='Eval'):    
    feat = rec['feature'].unsqueeze(0).to(device)
    past = rec['past'].unsqueeze(0).to(device)
    emb  = intent_emb(rec['intent'].unsqueeze(0).to(device))
    # forward
    with torch.no_grad():
        pred = head_model(feat, past, emb).cpu().numpy()[0]  # [T,2]
    # ADE
    gt = rec['future'].numpy()
    mask = np.ones(gt.shape[0], dtype=bool)
    d = np.linalg.norm(pred - gt, axis=-1)
    ADEs.append(d.mean())
    # RFS
    proc_trajs, proc_scores = process_rater_specified_trajectories(
        [rec['raw_trajs']], [np.array(rec['raw_scores'])],
        target_num_waypoints=freq*secs,
        target_num_trajectories_per_batch=P
    )
    pred_t = pred[None,None]  # [1,1,T,2]
    probs  = np.ones((1,1))
    init_v = np.array([rec['init_speed']])
    out = rater_feedback_utils.get_rater_feedback_score(
        pred_t, probs, proc_trajs, proc_scores, init_v,
        frequency=freq, length_seconds=secs,
        output_trust_region_visualization=False
    )
    RFSs.append(out['rater_feedback_score'][0])

print(f"\nOverall ADE: {np.mean(ADEs):.3f} ± {np.std(ADEs):.3f} m")
print(f"Overall RFS: {np.mean(RFSs):.3f} ± {np.std(RFSs):.3f}\n")
