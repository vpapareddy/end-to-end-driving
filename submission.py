#!/usr/bin/env python3
import os
import io
import glob
import struct
import math
import re
import tarfile
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

from waymo_open_dataset.protos.end_to_end_driving_data_pb2 import E2EDFrame
from waymo_open_dataset.protos import end_to_end_driving_submission_pb2 as wod_e2ed_submission_pb2

# -----------------------------------------
# Pure-Python TFRecord reader
# -----------------------------------------
def python_tfrecord_iterator(path):
    with tf.io.gfile.GFile(path, "rb") as f:
        while True:
            header = f.read(8)
            if not header:
                break
            length = struct.unpack("<Q", header)[0]
            f.read(4)
            data = f.read(length)
            f.read(4)
            yield data

# -----------------------------------------
# Feature Extraction & Caching (test shards)
# -----------------------------------------
class ShardFeatureExtractor:
    def __init__(self, shards, transform, cache_dir, device):
        self.shards = shards
        self.transform = transform
        self.cache_dir = cache_dir
        self.device = device
        os.makedirs(cache_dir, exist_ok=True)
        weights = ResNet50_Weights.IMAGENET1K_V1
        backbone = resnet50(weights=weights)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1]).to(device).eval()
        self.future_len = 20

    def extract(self):
        for idx, shard_path in enumerate(self.shards):
            cache_path = os.path.join(self.cache_dir, f"cache_{idx:02d}.pt")
            if os.path.exists(cache_path):
                print(f"Skipping existing cache: {cache_path}")
                continue
            print(f"Extracting features for shard {idx}: {shard_path}")
            feats, pasts, intents, futs, bests = [], [], [], [], []
            frame_names = []
            for raw in tqdm(python_tfrecord_iterator(shard_path), desc="Records", unit="rec"):
                msg = E2EDFrame(); msg.ParseFromString(raw)
                frame_names.append(msg.frame.context.name)

                # images → features
                imgs = []
                for cam in msg.frame.images:
                    im = Image.open(io.BytesIO(cam.image)).convert("RGB")
                    imgs.append(self.transform(im).to(self.device))
                imgs = torch.stack(imgs)
                with torch.no_grad():
                    f = self.cnn(imgs)
                    f = f.view(len(msg.frame.images), -1).view(-1).cpu()
                feats.append(f)

                # past trajectory
                pasts.append(torch.tensor(
                    list(zip(msg.past_states.pos_x, msg.past_states.pos_y)),
                    dtype=torch.float32
                ))

                # intent
                intents.append(torch.tensor(msg.intent, dtype=torch.long))

                # future (pad/trunc)
                ft = list(zip(msg.future_states.pos_x, msg.future_states.pos_y))
                if len(ft) >= self.future_len:
                    ft = ft[:self.future_len]
                else:
                    pad = ft[-1] if ft else (0.0, 0.0)
                    ft += [pad] * (self.future_len - len(ft))
                futs.append(torch.tensor(ft, dtype=torch.float32))

                # best-rated (pad/trunc)
                valid = [
                    (t.pos_x, t.pos_y, t.preference_score)
                    for t in msg.preference_trajectories
                    if t.preference_score >= 0 and len(t.pos_x) > 0
                ]
                if valid:
                    px, py, _ = max(valid, key=lambda x: x[2])
                    coords = list(zip(px, py))
                else:
                    coords = []
                if len(coords) >= self.future_len:
                    coords = coords[:self.future_len]
                else:
                    pad = coords[-1] if coords else (0.0, 0.0)
                    coords += [pad] * (self.future_len - len(coords))
                bests.append(torch.tensor(coords, dtype=torch.float32))

            # save cache
            cache_data = {
                'features':    torch.stack(feats),
                'past':        torch.stack(pasts),
                'intent':      torch.stack(intents),
                'future':      torch.stack(futs),
                'best':        torch.stack(bests),
                'frame_names': frame_names,
            }
            torch.save(cache_data, cache_path)
            print(f"Saved cache to {cache_path}")

# -----------------------------------------
# Head-only model (must match training)
# -----------------------------------------
class E2EModel(nn.Module):
    def __init__(self, feature_dim, past_len=16, intent_emb_dim=16, future_len=20):
        super().__init__()
        total_dim = feature_dim + past_len*2 + intent_emb_dim
        self.head = nn.Sequential(
            nn.Linear(total_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, future_len*2),
        )
        self.future_len = future_len

    def forward(self, feat, past, intent_emb):
        past_flat = past.view(past.size(0), -1)
        x = torch.cat([feat, past_flat, intent_emb], dim=1)
        return self.head(x).view(-1, self.future_len, 2)

# -----------------------------------------
# Main: extract & submit
# -----------------------------------------
if __name__ == "__main__":
    # 1) Extract & cache test shards
    DATASET_FOLDER = "gs://waymo_open_dataset_end_to_end_camera_v_1_0_0"
    test_shards    = sorted(tf.io.gfile.glob(os.path.join(DATASET_FOLDER, "test*.tfrecord*")))
    cache_dir      = os.path.join(os.path.dirname(__file__), "cache/test")
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform      = T.Compose([T.Resize((128,128)), T.ToTensor()])
    ShardFeatureExtractor(test_shards, transform, cache_dir, device).extract()

    # 2) Load model_final.pth
    ckpt = torch.load("model_final.pth", map_location=device)
    feature_dim = ckpt["feature_dim"]
    model       = E2EModel(feature_dim).to(device).eval()
    head_sd = {k.split("head.",1)[1] if k.startswith("head.") else k: v
               for k,v in ckpt["head_state_dict"].items()}
    model.head.load_state_dict(head_sd)
    intent_emb = nn.Embedding(4,16).to(device)
    emb_sd = {k.split("intent_emb.",1)[1] if k.startswith("intent_emb.") else k: v
              for k,v in ckpt["intent_emb_state_dict"].items()}
    intent_emb.load_state_dict(emb_sd)
    print("Loaded model_final.pth → head + intent_emb")

    # 3) Build flat list of FrameTrajectoryPredictions for JSON frames
    list_path = os.path.join(os.path.dirname(__file__), "test_sequence_frames_for_submission_as_list.txt")
    with open(list_path) as f:
        required_frames = [l.strip() for l in f if l.strip()]
    required_set = set(required_frames)

    predictions = []
    counts = []
    cache_files = sorted(
        glob.glob(os.path.join(cache_dir, "cache_*.pt")),
        key=lambda p: int(re.search(r'cache_(\d+)\.pt$', os.path.basename(p)).group(1))
    )
    for cf in cache_files:
        data = torch.load(cf)
        feats = data['features']
        pasts = data['past']
        intents = data['intent']
        frame_names = data['frame_names']

        sel_indices = [i for i, name in enumerate(frame_names) if name in required_set]
        counts.append(len(sel_indices))

        for i in sel_indices:
            f = feats[i].unsqueeze(0).to(device)
            p = pasts[i].unsqueeze(0).to(device)
            it = intents[i].unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(f, p, intent_emb(it)).cpu().numpy()[0]
            traj = wod_e2ed_submission_pb2.TrajectoryPrediction(
                pos_x=np.ascontiguousarray(pred[:,0], dtype=np.float32),
                pos_y=np.ascontiguousarray(pred[:,1], dtype=np.float32),
            )
            ftp = wod_e2ed_submission_pb2.FrameTrajectoryPredictions(
                frame_name=frame_names[i],
                trajectory=traj
            )
            predictions.append(ftp)
        print(f"Cache {os.path.basename(cf)} → {len(sel_indices)} predictions")

    total_expected = len(required_frames)
    total_actual = len(predictions)
    print(f"# predictions after filtering: {total_actual} (expected {total_expected})")

    # 4) Pack for submission (mirror Waymo guide exactly)
    # num_submission_shards = 1  # Please modify accordingly.
    # submission_file_base = os.path.join(os.path.dirname(__file__), "submission")
    # if not os.path.exists(submission_file_base):
    #     os.makedirs(submission_file_base)
    # sub_file_names = [
    #     os.path.join(submission_file_base, part)
    #     for part in [f'part{i}' for i in range(num_submission_shards)]
    # ]

    # YOU NEED TO PACK IT WITH THIS EXACT shard_filename FOR SUBMISSION TO BE DETECTED WITHOUT ERRORS
    num_submission_shards = 1  # still 1
    submission_file_base = os.path.join(os.path.dirname(__file__), "submission")
    if not os.path.exists(submission_file_base):
        os.makedirs(submission_file_base)

    # — rename your single shard to exactly what Waymo expects —
    shard_filename = "mysubmission.binproto-00000-of-00001"
    sub_file_names = [os.path.join(submission_file_base, shard_filename)]

    submissions = []
    num_predictions_per_shard = math.ceil(len(predictions) / num_submission_shards)
    for i in range(num_submission_shards):
        start = i * num_predictions_per_shard
        end = (i + 1) * num_predictions_per_shard
        submissions.append(
            wod_e2ed_submission_pb2.E2EDChallengeSubmission(
                predictions=predictions[start:end]
            )
        )

    for i, shard in enumerate(submissions):
        shard.submission_type = wod_e2ed_submission_pb2.E2EDChallengeSubmission.SubmissionType.E2ED_SUBMISSION
        shard.authors[:] = ['Vik Papareddy']
        shard.affiliation = 'N/A'
        shard.account_name = 'vik.papareddy@gmail.com'
        shard.unique_method_name = 'FrozenResNet50'
        shard.method_link = 'https://github.com/vpapareddy/end-to-end-driving'
        shard.description = 'Frozen ResNet50 backbone to extract per-view features + lightweight MLP head'
        shard.uses_public_model_pretraining = True
        shard.public_model_names.extend(['ResNet50'])
        shard.num_model_parameters = "40M"
        with tf.io.gfile.GFile(sub_file_names[i], 'wb') as fp:
            fp.write(shard.SerializeToString())
        print(f'Wrote submission shard {i}: {sub_file_names[i]}')

    # 5) Tar + gzip all parts into one archive
    tar_path = os.path.join(submission_file_base, 'submission.tar.gz')
    with tarfile.open(tar_path, 'w:gz') as tar:
        for fname in sub_file_names:
            tar.add(fname, arcname=os.path.basename(fname))
    print('\nDone! Final submission archive at:', tar_path)
