# Architecture: Mel Spectrogram (librosa) → InstanceNorm → EfficientNet-B0 → GeM → Linear(10)
# Key: On-the-fly mashup augmentation + SpecAugment + Mixup
# No torchaudio dependency - uses librosa for all audio processing.

import os, glob, random, warnings, time, gc
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import librosa
import timm

from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore")

# helper functions
from caching import cache_path, precompute_cache
from helper import load_wav, load_wav_tta, spec_augment, wav_to_mel

# datasets
from dataset import MashupDataset, ValDataset, TestDataset


# config
SEED = 23456543
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {DEVICE}")

DATA_ROOT = "/kaggle/input/competitions/jan-2026-dl-gen-ai-project/messy_mashup"
OUTPUT_DIR = "./outputs"

STEMS_DIR = os.path.join(DATA_ROOT, "genres_stems")
NOISE_DIR = os.path.join(DATA_ROOT, "ESC-50-master", "audio")
TEST_DIR = os.path.join(DATA_ROOT, "mashups")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")

# audio params
SR = 22050 # Sampling Rate
DURATION = 10.0 # 10s
TARGET_LEN = int(SR * DURATION) # 220500 samples
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 20
FMAX = 8000

GENRES = sorted(['blues', 'classical', 'country', 'disco', 'hiphop',
                 'jazz', 'metal', 'pop', 'reggae', 'rock'])


GENRES2IDX = {g : i for i, g in enumerate(GENRES)}
IDX2GENRES = {i : g for g, i in GENRES2IDX.items()}
STEMS = ["drums", "vocals", "bass", "other"]

SAMPLES_PER_GENRE = 1000 # 10k mashups per epoch
BATCH_SIZE = 32
EPOCHS = 35
LR = 1e-3
WEIGHT_DECAY = 1e-4 # l2-reg
LABEL_SMOOTHING = 0.1 # better generalization
NUM_WORKERS = 4
MIXUP_ALPHA = 0.4

# from EDA
STEM_WEIGHTS = {
    "drums": 0.45,
    "vocals": 0.30,
    "bass": 0.15,
    "other": 0.10
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Will train on {SAMPLES_PER_GENRE * 10} mashups per epoch")

# data index and train_val split
stem_index = {g : {st: [] for st in STEMS} for g in GENRES}
song_index = {g : [] for g in GENRES}

for genre in GENRES:
    genre_path = os.path.join(STEMS_DIR, genre)
    songs = sorted(s for s in os.listdir(genre_path)
                    if os.path.isdir(os.path.join(genre_path, s)))

    for song in songs:
        song_dir = os.path.join(genre_path, song)
        available_stems = []
        for stem in STEMS:
            filepath = os.path.join(song_dir, f"{stem}.wav")
            if os.path.exists(filepath):
                stem_index[genre][stem].append(filepath)
                available_stems.append(stem)

        if available_stems:
            song_index[genre].append({
                'dir': song_dir,
                "stems": available_stems
            })


noise_files = sorted(glob.glob(os.path.join(NOISE_DIR, "*.wav")))
print(f"Found {len(noise_files)} noise clips from ESC-50")

train_stems = {g : {st: [] for st in STEMS} for g in GENRES}
val_songs = {g: [] for g in GENRES}

for genre in GENRES:
    songs = song_index[genre].copy()
    random.shuffle(songs)
    split_point = int(0.85 * len(songs))

    # slice
    train_list = songs[:split_point]
    val_list = songs[split_point:]
    val_songs[genre] = val_list

    # only use stems from training songs
    train_dirs = {s['dir'] for s in train_list}
    for stem in STEMS:
        train_stems[genre][stem] = [
            fp for fp in stem_index[genre][stem]
            if os.path.dirname(fp) in train_dirs
        ]

    print(f"   {genre}: {len(train_list)} train, {len(val_list)} val")

# Model
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=(-2, -1)).pow(1.0 / self.p)

class GenreClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.inst_norm = nn.InstanceNorm2d(1) # volume norm
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=True,
            in_chans=1, num_classes=0, global_pool=''
        )
        num_features = self.backbone.num_features # 1280 for effnet-b0
        self.gem = GeM(p=3.0)
        self.head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x, augment=False):
        x = self.inst_norm(x)
        if augment:
            x = spec_augment(x)
        features = self.backbone(x)
        pooled = self.gem(features)
        return self.head(pooled)

# sanity check
model = GenreClassifier().to(DEVICE)
dummy = torch.randn(2, 1, N_MELS, TARGET_LEN // HOP_LENGTH + 1).to(DEVICE)
with torch.no_grad():
    out = model(dummy)
print(f"Output shape: {out.shape}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
del dummy, out
gc.collect()
torch.cuda.empty_cache()

def mixup_data(x, y, alpha=0.4):
    # Mix two samples with random weight from Beta distribution
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    # Weighted loss for mixed labels
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
    n_mels=N_MELS, f_min=FMIN, f_max=FMAX
).to(DEVICE)
db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80).to(DEVICE)

def wav_batch_to_mel(wav_batch):
    # wav_batch: (B, T) on DEVICE
    mel = mel_transform(wav_batch)          # (B, n_mels, frames)
    mel_db = db_transform(mel)
    return mel_db.unsqueeze(1)              # (B, 1, n_mels, frames)

def train_one_epoch(model, loader, optimizer, scaler, criterion):
    model.train()
    total_loss = 0
    num_samples = 0

    for wav, labels in tqdm(loader, desc="Train", leave=False):
        wav = wav.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()

        # mel/db in fp32 — avoids amin=1e-10 underflowing under fp16
        mel = wav_batch_to_mel(wav)
        assert torch.isfinite(mel).all(), "non-finite values in mel"
        
        with autocast():
            # apply mixup 50% of the time
            if random.random() < 0.5:
                mel_mixed, y_a, y_b, lam = mixup_data(mel, labels, MIXUP_ALPHA)
                logits = model(mel_mixed, augment=True)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                logits = model(mel, augment=True)
                loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(labels)
        num_samples += len(labels)

    return total_loss / num_samples


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    for wav, labels in loader:
        wav = wav.to(DEVICE)
        mel = wav_batch_to_mel(wav)

        with autocast():
            logits = model(mel, augment=False)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

    f1 = f1_score(all_labels, all_preds, average="macro")
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return f1, acc, np.array(all_preds), np.array(all_labels)

train_dataset = MashupDataset(train_stems, noise_files, SAMPLES_PER_GENRE, augment=True)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
    prefetch_factor=4,
)

val_dataset = ValDataset(val_songs)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)
print(f"Validation samples: {len(val_dataset)}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
scaler = GradScaler()

best_f1 = 0.0
history = {
    "loss": [],
    "val_f1": [],
    "val_acc": [],
    "lr": []
}

for epoch in range(1, EPOCHS + 1):
    start_time = time.time()
    
    loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion)
    scheduler.step()

    val_f1, val_acc, _, _ = evaluate(model, val_loader)
    lr = scheduler.get_last_lr()[0]
    elapsed = time.time() - start_time

    # track history
    history['loss'].append(loss)
    history['val_f1'].append(val_f1)
    history['val_acc'].append(val_acc)
    history['lr'].append(lr)

    # save best model
    tag = ""
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_cnn.pth'))
        tag = " [best]"

    print(f"E{epoch:02d}/{EPOCHS} | loss={loss:.4f} | f1={val_f1:.4f} | "
          f"acc={val_acc:.4f} | lr={lr:.6f} | {elapsed:.0f}s{tag}")

print(f"\nBest validation F1: {best_f1:.4f}")
