import os
import random
import glob
import warnings
import time, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import torchaudio
import librosa
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore")

SEED = 23456543
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {DEVICE}")

# Config
DATA_ROOT = "/kaggle/input/competitions/jan-2026-dl-gen-ai-project/messy_mashup"
OUTPUT_DIR = "./outputs"

STEMS_DIR = os.path.join(DATA_ROOT, "genres_stems")
NOISE_DIR = os.path.join(DATA_ROOT, "ESC-50-master", "audio")
TEST_DIR = os.path.join(DATA_ROOT, "mashups")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")

# AST expects 16kHz audio
SR = 16000
DURATION = 10.0
TARGET_LEN = int(SR * DURATION)

# genres
GENRES = sorted(['blues', 'classical', 'country', 'disco', 'hiphop',
                 'jazz', 'metal', 'pop', 'reggae', 'rock'])
GENRES2IDX = {g : i for i, g in enumerate(GENRES)}
IDX2GENRES = {i : g for g, i in GENRES2IDX.items()}
STEMS = ["drums", "vocals", "bass", "other"]

# training params - less than CNNs cuz this is very heavy (transformer)
SAMPLES_PER_GENRE = 800 # 8000 mashups per epoch
BATCH_SIZE = 8
ACCUM_STEPS = 4 # effective batch = 8 * 4 = 32
EPOCHS = 20
LR_BACKBONE = 1e-5 # lower to preserve pretrained weights
LR_HEAD = 1e-3 # high for new classifier head
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0
NUM_WORKERS = 4
WARMUP_EPOCHS = 2

STEM_WEIGHTS = {
    "drums": 0.45,
    "vocals": 0.30,
    "bass": 0.15,
    "other": 0.10
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Will train on {SAMPLES_PER_GENRE * 10} mashups/epoch")
print(f"Effective batch size: {BATCH_SIZE} x {ACCUM_STEPS} = {BATCH_SIZE * ACCUM_STEPS}")

# splitting
stem_index = {g : {st : [] for st in STEMS} for g in GENRES}
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
                'stems': available_stems
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

    print(f"{genre}: {len(train_list)} train, {len(val_list)} val")

# model
AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

feature_extractor = ASTFeatureExtractor.from_pretrained(AST_MODEL_NAME)
print("Feature extractor loaded")

class ASTGenreClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.ast = ASTForAudioClassification.from_pretrained(
            AST_MODEL_NAME,
            num_labels=num_classes,
            ignore_mismatched_sizes = True # head size changes from 527 -> 10
        )

    def forward(self, x):
        # input_values shape: (batch, 1024, 128) - from feature extractor
        outputs = self.ast(input_values=x)
        return outputs.logits

model = ASTGenreClassifier(num_classes=10).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"AST parameters: {total_params / 1e6:.1f}M")

# collator functions
class ASTCollator:
    def __init__(self, feature_extractor, sr=16000):
        self.fe = feature_extractor
        self.sr = sr

    def __call__(self, batch):
        waveforms, labels = zip(*batch)

        waveforms_np = [w.numpy() for w in waveforms]
        inputs = self.fe(
            waveforms_np,
            sampling_rate=self.sr,
            return_tensors="pt",
            padding="max_length",
            max_length=1024,
            truncation=True
        )

        if isinstance(labels[0], int) or isinstance(labels[0], np.integer):
            labels_out = torch.tensor(labels, dtype=torch.long)
        else:
            labels_out = labels

        return inputs["input_values"], labels_out

class ASTTestCollator:
    # same but for test set where labels are string IDs
    
    def __init__(self, feature_extractor, sr=16000):
        self.fe = feature_extractor
        self.sr = sr

    def __call__(self, batch):
        waveforms, ids = zip(*batch)
        waveforms_np = [w.numpy() for w in waveforms]
        inputs = self.fe(
            waveforms_np,
            sampling_rate=self.sr,
            return_tensors="pt",
            padding="max_length",
            max_length=1024,
            truncation=True,
        )
        return inputs["input_values"], list(ids)

print("Collators ready")

# Training
def train_one_epoch(model, loader, optimizer, scaler, criterion):
    model.train()
    total_loss = 0.0
    num_samples = 0
    optimizer.zero_grad()

    for step, (input_values, labels) in enumerate(tqdm(loader, desc="Train", leave=False)):
        input_values = input_values.to(DEVICE)
        labels = labels.to(DEVICE)

        with autocast():
            logits = model(input_values)
            loss = criterion(logits, labels) / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() *  ACCUM_STEPS * input_values.size(0)
        num_samples += len(labels)

    return total_loss / num_samples

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    for input_values, labels in loader:
        input_values = input_values.to(DEVICE)
        with autocast():
            logits = model(input_values)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

    f1 = f1_score(all_labels, all_preds, average="macro")
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return f1, acc, np.array(all_preds), np.array(all_labels)

# Optimizer with Differential Learning Rate
backbone_params = []
head_params = []

for name, param in model.named_parameters():
    if 'classifier' in name:
        head_params.append(param)
    else:
        backbone_params.append(param)

print(f"Backbone: {sum(p.numel() for p in backbone_params) / 1e6:.1f}M params (lr={LR_BACKBONE})")
print(f"Head: {sum(p.numel() for p in head_params)} params (lr={LR_HEAD})")

# different learning rates for backbone vs head
optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': LR_BACKBONE},
    {'params': head_params, 'lr': LR_HEAD},
], weight_decay=WEIGHT_DECAY)

# cosine schedule with warmup
def get_lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        # linear warmup
        return (epoch + 1) / WARMUP_EPOCHS
    # cosine decay after warmup
    progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
scaler = GradScaler()

collator = ASTCollator(feature_extractor, sr=SR)

val_ds = ValDataset(val_songs)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        collate_fn=collator)
print(f"Validation samples: {len(val_ds)}")

best_f1 = 0.0
history = {'loss': [], 'val_f1': [], 'val_acc': [], 'lr': []}

for epoch in range(1, EPOCHS + 1):
    start_time = time.time()

    # new mashups every epoch
    train_ds = MashupDataset(train_stems, noise_files, SAMPLES_PER_GENRE, augment=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=True, collate_fn=collator)

    loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion)
    scheduler.step()
    
    val_f1, val_acc, _, _ = evaluate(model, val_loader)
    lr = optimizer.param_groups[0]['lr']
    elapsed = time.time() - start_time

    # track history
    history['loss'].append(loss)
    history['val_f1'].append(val_f1)
    history['val_acc'].append(val_acc)
    history['lr'].append(lr)

    # save best
    tag = ""
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_ast.pth'))
        tag = " [best]"

    print(f"E{epoch:02d}/{EPOCHS} | loss={loss:.4f} | f1={val_f1:.4f} | "
          f"acc={val_acc:.4f} | lr={lr:.6f} | {elapsed:.0f}s{tag}")

print(f"\nBest validation F1: {best_f1:.4f}")
