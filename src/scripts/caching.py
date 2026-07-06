# caching raw waveforms so that the CPU doesn't have to read and decode audio files on the fly.

import os
from tqdm.auto import tqdm
import numpy as np
import librosa

DATA_ROOT = "/kaggle/input/competitions/jan-2026-dl-gen-ai-project/messy_mashup"
NOISE_DIR = os.path.join(DATA_ROOT, "ESC-50-master", "audio")

SR = 22050 # Sampling Rate
DURATION = 10.0 # 10s
TARGET_LEN = int(SR * DURATION) # 220500 samples
GENRES = sorted(['blues', 'classical', 'country', 'disco', 'hiphop',
                 'jazz', 'metal', 'pop', 'reggae', 'rock'])
STEMS = ["drums", "vocals", "bass", "other"]
CACHE_DIR = "./wav-cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_path(orig_path):
    key = orig_path.replace("/", "_")
    return os.path.join(CACHE_DIR, key + ".npy")

def precompute_cache(filepaths):
    for fp in tqdm(filepaths, desc="Caching Waveforms"):
        cp = cache_path(fp)
        if os.path.exists(cp):
            continue
        try:
            y, _ = librosa.load(fp, sr=SR, mono=True)
        except Exception:
            y = np.zeros(TARGET_LEN, dtype=np.float32)
        np.save(cp, y.astype(np.float32))

stem_index = {g : {st: [] for st in STEMS} for g in GENRES}
noise_files = sorted(glob.glob(os.path.join(NOISE_DIR, "*.wav")))

# gather every file that will ever be loaded
all_files = set()
for genre in GENRES:
    for stem in STEMS:
        all_files.update(stem_index[genre][stem])
all_files.update(noise_files)

precompute_cache(list(all_files))