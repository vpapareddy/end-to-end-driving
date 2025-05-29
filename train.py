import os, io, glob, struct
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as T
from waymo_open_dataset.protos.end_to_end_driving_data_pb2 import E2EDFrame
from PIL import Image
import tensorflow as tf  # for GCS streaming and file IO
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import numpy as np

# -----------------------------------------
# Pure-Python TFRecord reader
# -----------------------------------------
def python_tfrecord_iterator(path):
    """
    Yields raw bytes from a TFRecord file, supporting local paths or GCS URIs.
    """
    with tf.io.gfile.GFile(path, "rb") as f:
        while True:
            header = f.read(8)
            if not header:
                break
            length = struct.unpack("<Q", header)[0]
            f.read(4)  # skip length CRC
            data = f.read(length)
            f.read(4)  # skip data CRC
            yield data

# -----------------------------------------
# Feature Extraction & Caching
# -----------------------------------------
class ShardFeatureExtractor:
    def __init__(self, shards, transform, cache_dir, device):
        self.shards = shards
        self.transform = transform
        self.cache_dir = cache_dir
        self.device = device
        os.makedirs(cache_dir, exist_ok=True)
        # CNN backbone, pretrained on ImageNet
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
            raw_trajs, raw_scores, init_speeds = [], [], []
            for raw in tqdm(python_tfrecord_iterator(shard_path), desc="Records", unit="rec"):
                msg = E2EDFrame(); msg.ParseFromString(raw)
                # --- collect raw rater trajectories & scores ---
                traj_list, score_list = [], []
                for traj in msg.preference_trajectories:
                    if traj.preference_score >= 0 and len(traj.pos_x) > 0:
                        coords = np.stack([traj.pos_x, traj.pos_y], axis=1)  # [T,2]
                        traj_list.append(coords)
                        score_list.append(traj.preference_score)
                raw_trajs.append(traj_list)
                raw_scores.append(np.array(score_list, dtype=np.float32))
                # initial speed from last past state
                vx, vy = msg.past_states.vel_x[-1], msg.past_states.vel_y[-1]
                init_speeds.append(np.hypot(vx, vy))

                # decode & transform images
                imgs = []
                for cam in msg.frame.images:
                    im = Image.open(io.BytesIO(cam.image)).convert("RGB")
                    imgs.append(self.transform(im).to(self.device))
                imgs = torch.stack(imgs)
                with torch.no_grad():
                    f = self.cnn(imgs)                # [V,2048,1,1]
                    f = f.view(len(msg.frame.images), -1)  # [V,2048]
                    f = f.view(-1).cpu()                 # [V*2048]
                feats.append(f)
                # past trajectory
                past = torch.tensor(list(zip(msg.past_states.pos_x, msg.past_states.pos_y)), dtype=torch.float32)
                pasts.append(past)
                # intent
                intents.append(torch.tensor(msg.intent, dtype=torch.long))
                # future, pad/truncate to fixed length
                ft = list(zip(msg.future_states.pos_x, msg.future_states.pos_y))
                if len(ft) >= self.future_len:
                    ft = ft[:self.future_len]
                else:
                    pad = ft[-1] if ft else (0.0, 0.0)
                    ft += [pad] * (self.future_len - len(ft))
                futs.append(torch.tensor(ft, dtype=torch.float32))
                # best-rated human trajectory
                valid = [ (traj.pos_x, traj.pos_y, traj.preference_score)
                          for traj in msg.preference_trajectories
                          if traj.preference_score >= 0 and len(traj.pos_x)>0 ]
                if valid:
                    px, py, _ = max(valid, key=lambda x: x[2])
                    coords_best = list(zip(px, py))
                else:
                    coords_best = []
                if len(coords_best) >= self.future_len:
                    coords_best = coords_best[:self.future_len]
                else:
                    pad = coords_best[-1] if coords_best else (0.0, 0.0)
                    coords_best += [pad] * (self.future_len - len(coords_best))
                bests.append(torch.tensor(coords_best, dtype=torch.float32))

            # save cache with metadata
            cache_data = {
                'features':     torch.stack(feats),
                'past':         torch.stack(pasts),
                'intent':       torch.stack(intents),
                'future':       torch.stack(futs),
                'best':         torch.stack(bests),
                'raw_trajs':    raw_trajs,
                'raw_scores':   raw_scores,
                'init_speeds':  init_speeds,
            }
            torch.save(cache_data, cache_path)
            print(f"Saved cache to {cache_path}")

# -----------------------------------------
# Cache-Based Dataset
# -----------------------------------------
class CacheDataset(Dataset):
    def __init__(self, cache_files):
        self.records = []
        for f in cache_files:
            d = torch.load(f)
            N = d['features'].shape[0]
            for i in range(N):
                self.records.append((
                    d['features'][i],
                    d['past'][i],
                    d['intent'][i],
                    d['future'][i],
                    d['best'][i]
                ))
    def __len__(self):
        return len(self.records)
    def __getitem__(self, idx):
        return self.records[idx]

# -----------------------------------------
# Model Definition (head only)
# -----------------------------------------
class E2EModel(nn.Module):
    def __init__(self, feature_dim, past_len=16, intent_emb_dim=16, future_len=20):
        super().__init__()
        total_dim = feature_dim + past_len*2 + intent_emb_dim
        # MLP head with dropout for regularization
        self.head = nn.Sequential(
            nn.Linear(total_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, future_len*2)
        )
        self.future_len = future_len
    def forward(self, feat, past, intent_emb):
        past_flat = past.view(past.size(0), -1)
        x = torch.cat([feat, past_flat, intent_emb], dim=1)
        return self.head(x).view(-1, self.future_len, 2)

# -----------------------------------------
# Training Loop with Cache & Phases
# -----------------------------------------
if __name__ == '__main__':
    DATASET_FOLDER = 'gs://waymo_open_dataset_end_to_end_camera_v_1_0_0'
    train_shards = tf.io.gfile.glob(os.path.join(DATASET_FOLDER, 'training*.tfrecord*'))[:93]
    val_shards   = tf.io.gfile.glob(os.path.join(DATASET_FOLDER, 'val*.tfrecord*'))[:1]
    test_shards  = tf.io.gfile.glob(os.path.join(DATASET_FOLDER, 'test*.tfrecord*'))[:1]

    dirs = {'train':'cache/train','val':'cache/val','test':'cache/test'}
    for d in dirs.values(): os.makedirs(d, exist_ok=True)

    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = T.Compose([T.Resize((128,128)), T.ToTensor()])

    # Phase 1: Cache extraction
    ShardFeatureExtractor(train_shards, transform, dirs['train'], device).extract()
    ShardFeatureExtractor(val_shards,   transform, dirs['val'],   device).extract()
    ShardFeatureExtractor(test_shards,  transform, dirs['test'],  device).extract()

    # Phase 2: Load caches in streaming chunks to limit memory
    train_cache_files = sorted(glob.glob(os.path.join(dirs['train'], 'cache_*.pt')))
    val_cache_files   = sorted(glob.glob(os.path.join(dirs['val'],   'cache_*.pt')))

    val_ds   = CacheDataset(val_cache_files)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Phase 3: Train & tune streaming over train cache files
    feature_dim = torch.load(train_cache_files[0])['features'].shape[1]
    model = E2EModel(feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    intent_emb_layer = nn.Embedding(4,16).to(device)
    chunk_size = 30
    best_val_ade = float('inf')

    for epoch in range(1,4):
        model.train(); total_loss = 0
        for start in range(0, len(train_cache_files), chunk_size):
            chunk_files = train_cache_files[start:start+chunk_size]
            train_ds = CacheDataset(chunk_files)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            for feat,past,intent,fut,best in tqdm(train_loader, desc=f"Epoch {epoch} Train"): 
                feat, past = feat.to(device), past.to(device)
                intent_emb = intent_emb_layer(intent.to(device))
                fut, best = fut.to(device), best.to(device)
                pred = model(feat,past,intent_emb)
                loss = criterion(pred, fut) + 0.3*criterion(pred,best)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item()
            del train_ds, train_loader
        avg_train_loss = total_loss / (len(train_cache_files) * batch_size)
        print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}")

        # Val ADE & checkpoint best
        model.eval(); ta,de = 0,0
        with torch.no_grad():
            for feat,past,intent,fut,_ in val_loader:
                feat, past = feat.to(device), past.to(device)
                intent_emb = intent_emb_layer(intent.to(device))
                fut = fut.to(device)
                pred = model(feat,past,intent_emb)
                ade = torch.mean(torch.norm(pred-fut,dim=-1),dim=-1)
                ta += ade.sum().item(); de += ade.size(0)
        val_ade = ta/de
        print(f"Epoch {epoch} Val ADE: {val_ade:.4f}")

        if val_ade < best_val_ade:
            best_val_ade = val_ade
            torch.save({
                'head_state_dict': model.state_dict(),
                'intent_emb_state_dict': intent_emb_layer.state_dict()
            }, 'best_model.pth')
            print(f"→ New best ADE: {best_val_ade:.4f}, saved best_model.pth")

    # Phase 4: Final train on train+val streaming in chunks
    # all_cache_files = train_cache_files + val_cache_files
    # for epoch in range(1,5):
    #     model.train(); tl = 0
    #     for start in range(0, len(all_cache_files), chunk_size):
    #         chunk_files = all_cache_files[start:start+chunk_size]
    #         ds_chunk = CacheDataset(chunk_files)
    #         loader_chunk = DataLoader(ds_chunk, batch_size=batch_size, shuffle=True)
    #         for feat,past,intent,fut,best in tqdm(loader_chunk, desc=f"Final Epoch {epoch}"):
    #             feat, past = feat.to(device), past.to(device)
    #             intent_emb = intent_emb_layer(intent.to(device))
    #             fut, best = fut.to(device), best.to(device)
    #             pred = model(feat,past,intent_emb)
    #             loss = criterion(pred,fut) + 0.3*criterion(pred,best)
    #             optimizer.zero_grad(); loss.backward(); optimizer.step()
    #             tl += loss.item()
    #         del ds_chunk, loader_chunk
    #     print(f"Final Epoch {epoch} Loss: {tl/len(all_cache_files)/batch_size:.4f}")

    # Phase 5: Save trained head & embedding
    ckpt = {
        'head_state_dict':       model.state_dict(),
        'intent_emb_state_dict': intent_emb_layer.state_dict(),
        'optimizer_state_dict':  optimizer.state_dict(),
        'feature_dim':           feature_dim,
        'future_len':            model.future_len,
    }
    save_path = 'model_final.pth'
    torch.save(ckpt, save_path)
    print(f"Saved model & embedding checkpoint to {save_path}")
