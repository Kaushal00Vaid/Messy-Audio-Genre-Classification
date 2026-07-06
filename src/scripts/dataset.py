class MashupDataset(Dataset):
    def __init__(self, stem_idx, noise_files, samples_per_genre=1000, augment=True):
        self.stem_idx = stem_idx
        self.noise_files = noise_files
        self.augment = augment
        self.samples = []
        for genre in GENRES:
            for _ in range(samples_per_genre):
                self.samples.append(GENRES2IDX[genre])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        genre_idx = self.samples[idx]
        genre = IDX2GENRES[genre_idx]

        # stem mixing
        stems_wav = []
        for stem_type in STEMS:
            available = self.stem_idx[genre][stem_type]
            if not available:
                continue
            wav = load_wav(random.choice(available))
            # weight stems by importance
            gain = random.uniform(0.5, 1.5) * (STEM_WEIGHTS[stem_type] / 0.25)
            stems_wav.append(wav * gain)

        if not stems_wav:
            mel = np.zeros((N_MELS, TARGET_LEN // HOP_LENGTH + 1), dtype=np.float32)
            return torch.from_numpy(mel).unsqueeze(0), genre_idx

        # mix all up
        mix = np.sum(stems_wav, axis=0)

        if self.augment:
            mix = np.roll(mix, random.randint(-SR, SR))

            # add ESC noise (0-2 at random SNR)
            for _ in range(random.randint(0, 2)):
                noise = load_wav(random.choice(self.noise_files))
                snr_db = random.uniform(5.0, 25.0)
                sig_pwr = np.mean(mix ** 2) + 1e-10
                nse_pwr = np.mean(noise ** 2) + 1e-10
                scale = np.sqrt(sig_pwr / (nse_pwr * 10 ** (snr_db / 10)))
                mix = mix + noise * scale

            if random.random() < 0.3:
                mix = np.clip(mix * random.uniform(1.2, 3.0), -1, 1)

        # normalise
        peak = np.max(np.abs(mix))
        if peak > 1e-6:
            mix = mix / peak * random.uniform(0.7, 1.0)

        # mel = wav_to_mel(mix)
        # return torch.from_numpy(mel).unsqueeze(0), genre_idx
        return torch.from_numpy(mix).float(), genre_idx

class ValDataset(Dataset):
    def __init__(self, song_idx):
        self.items = []
        for genre in GENRES:
            for song in song_idx[genre]:
                self.items.append((song, GENRES2IDX[genre]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        song_info, label = self.items[idx]
        stems = [load_wav(os.path.join(song_info['dir'], f"{st}.wav")) for st in song_info['stems']]
        mix = np.sum(stems, axis=0)
        peak = np.max(np.abs(mix))
        if peak > 1e-6:
            mix = mix / peak
        # mel = wav_to_mel(mix)
        # return torch.from_numpy(mel).unsqueeze(0), label
        return torch.from_numpy(mix).float(), label


class TestDataset(Dataset):
    def __init__(self, test_dir, test_csv):
        self.df = pd.read_csv(test_csv, dtype={'id': str})
        self.paths = []
        for _, row in self.df.iterrows():
            path = None
            for pattern in [f"song{str(row['id']).zfill(4)}.wav",
                            f"{row['id']}.wav",
                            f"song{row['id']}.wav"
                           ]:
                p = os.path.join(test_dir, pattern)
                if os.path.exists(p):
                    path = p
                    break
            self.paths.append(path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.paths[idx]
        if path:
            # mel = wav_to_mel(load_wav(path))
            # mel_tensor = torch.from_numpy(mel).unsqueeze(0)
            wav = load_wav_tta(path)
            peak = np.max(np.abs(wav))
            if peak > 1e-6:
                wav = wav / peak
            wav_tensor = torch.from_numpy(wav).float()
        else:
            wav_tensor = torch.zeros(TARGET_LEN, dtype=torch.float32)
        return wav_tensor, str(self.df.iloc[idx]['id'])