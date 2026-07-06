def load_wav(path, sr=SR, target_len=TARGET_LEN):
    cp = cache_path(path)
    try:
        y = np.load(cp)
    except Exception:
        y = np.zeros(target_len, dtype=np.float32)
    # clip or pad happens per-call since crop is random each time
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        start = random.randint(0, len(y) - target_len)
        y = y[start:start + target_len]
    return y.astype(np.float32)

def wav_to_mel(y, sr=SR):
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX
    )
    
    S_db = librosa.power_to_db(S, ref=np.max, top_db=80)
    return S_db.astype(np.float32)

def load_wav_tta(path, sr=SR, target_len=TARGET_LEN):
    cp = cache_path(path)
    try:
        y = np.load(cp)
    except Exception:
        try:
            y, _ = librosa.load(path, sr=sr, mono=True)
        except Exception:
            y = np.zeros(target_len, dtype=np.float32)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        start = random.randint(0, len(y) - target_len)  # random, not center — this is what makes TTA rounds differ
        y = y[start:start + target_len]
    return y.astype(np.float32)

def spec_augment(spec, freq_mask=27, time_mask=80, n_freq=2, n_time=2):
    _, _, n_mels, n_frames = spec.shape
    aug = spec.clone()

    # mask freq
    for _ in range(n_freq):
        f = random.randint(0, freq_mask)
        f0 = random.randint(0, max(0, n_mels - f))
        aug[:, :, f0: f0+f, :] = 0

    # mask time
    for _ in range(n_time):
        t = random.randint(0, time_mask)
        t0 = random.randint(0, max(0, n_frames - t))
        aug[:, :, :, t0:t0+t] = 0

    return aug
