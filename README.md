<h1 align="center"> DL-Genai Project 26t1 </h1>

<p align="center"><i>Name: Kaushal Vaid </i> </p>
<p align="center"><i>Roll-No: 24f2000681 </i></p>

# Classification on Messy Mashups

> Robust music genre classification under heavily degraded, noisy, real-world mashup conditions.

---

## Deployment Link

https://kaushal00vaid-messy-music-genre-classifier.hf.space

---

## Problem Statement

Given multi-stem audio tracks (drums, vocals, bass, other) from 10 music genres, classify each track correctly, but the catch is the **test data is a chaotic mess**: stems are cross-mixed across genres, tempo-shifted, and layered with heavy environmental noise (sirens, dogs, babies crying, vacuums, etc. from the ESC-50 dataset).

The core challenge is bridging the **domain gap** between clean training data and brutally degraded test conditions.

**Evaluation Metric:** Macro F1-Score  
**Best Leaderboard Score:** `0.80933`

---

## Models Used

| Model               | Type                     | Score        |
| ------------------- | ------------------------ | ------------ |
| ScratchCNN (v1)     | 4-block custom CNN       | ~0.10 val F1 |
| CustomCNN (v2)      | Residual + Attention CNN | 0.70105      |
| ResNet50            | Pretrained baseline      | 0.49205      |
| ResNet18            | Fine-tuned               | 0.80869      |
| **EfficientNet-B3** | **Fine-tuned (best)**    | **0.80933**  |

---

## Preprocessing Pipeline

**V1 (Failed):**

- Summed stems 50% of the time
- Single noise clip at high SNR (5–20 dB)
- 1-channel grayscale Mel-Spectrogram

**V2 (Robust Pipeline):**

- 100% cross-stem mixing across genres
- Per-stem tempo augmentation (±10% time-stretch before mixing)
- Aggressive noise injection at SNR 0–15 dB (sometimes multiple clips layered)
- 3-channel dynamic feature extraction: Mel + delta + delta-delta
- MixUp augmentation on spectrograms and labels
- OneCycleLR scheduler with warmup
- 5-Crop Test-Time Augmentation (TTA) at inference

---

## Folder Structure

```
├── notebooks/
│   ├── dl-24f2000681-notebook-t1202...ipynb   # Main training notebook
│   ├── milestone-1.ipynb
│   ├── milestone-2.ipynb
│   ├── milestone-4.ipynb
│   ├── milestone-5.ipynb
│   └── resnet50_baseline_0.49205.ipynb
├── reports/
│   ├── 24f2000681_DG_T12026.pdf               # Final report
│   ├── milestone-1.pdf
│   ├── milestone-2.pdf
│   ├── milestone-3.pdf
│   ├── milestone-4.pdf
│   └── milestone-5.pdf
├── src/
│   ├── EDA.ipynb
│   ├── EfficientNet_Architecture.png
│   ├── ScratchCNN_Baseline_Architecture.png
│   ├── ScratchCNN_Final_Architecture.png
│   └── resnet18_architecture.png
└── README.md
```

---

## References

- _DL for Audio Classification_ — Seth Adams
- _Audio Signal Processing_ — Valerio Velardo
- EfficientNet Architecture — GeeksForGeeks
- DLGenAI Course Lectures, CampusX (100 Days of DL)
