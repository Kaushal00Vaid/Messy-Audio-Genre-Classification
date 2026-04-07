import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import gradio as gr
from pathlib import Path
import torchvision.models as models

# ─────────────────────────────────────────────
# CONFIG  (must match exactly what you trained)
# ─────────────────────────────────────────────
CFG = dict(
    sr          = 22050,
    duration    = 10,
    n_mels      = 128,
    n_fft       = 2048,
    hop_length  = 512,
    fmax        = 8000,
    img_size    = 224,
    num_classes = 10,
    tta_n_crops = 5,
    genres      = ['blues', 'classical', 'country', 'disco', 'hiphop',
                   'jazz', 'metal', 'pop', 'reggae', 'rock'],
)

GENRE_TO_IDX = {g: i for i, g in enumerate(CFG['genres'])}
IDX_TO_GENRE = {i: g for g, i in GENRE_TO_IDX.items()}
TTA_OFFSETS  = [0.0, 0.25, 0.5, 0.75, 1.0]   # 5-crop positions

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────
# PREPROCESSING  (identical to training)
# ─────────────────────────────────────────────

def load_audio(path, sr, duration, offset_frac=None):
    y, _ = librosa.load(path, sr=sr, mono=True)
    n = len(y)
    target = sr * duration

    if n < target:
        repeats = (target // n) + 1
        y = np.tile(y, repeats)[:target]
    else:
        if offset_frac is not None:
            start = int(offset_frac * (n - target))
        else:
            start = random.randint(0, n - target)
        y = y[start: start + target]

    return y.astype(np.float32)


def audio_to_melspec(y: np.ndarray, cfg: dict) -> np.ndarray:
    mel    = librosa.feature.melspectrogram(
        y=y, sr=cfg['sr'], n_mels=cfg['n_mels'],
        n_fft=cfg['n_fft'], hop_length=cfg['hop_length'], fmax=cfg['fmax']
    )
    meldb  = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    delta1 = librosa.feature.delta(meldb, width=9).astype(np.float32)
    delta2 = librosa.feature.delta(meldb, width=9, order=2).astype(np.float32)
    return np.stack([meldb, delta1, delta2], axis=0)   # (3, n_mels, T)


def normalise_channels(spec: np.ndarray) -> np.ndarray:
    out = np.empty_like(spec)
    for c in range(spec.shape[0]):
        lo, hi = spec[c].min(), spec[c].max()
        out[c] = (spec[c] - lo) / (hi - lo + 1e-6)
    return out


def spec_to_tensor(spec: np.ndarray, img_size: int) -> torch.Tensor:
    spec = normalise_channels(spec)
    t    = torch.tensor(spec, dtype=torch.float32)
    t    = F.interpolate(
        t.unsqueeze(0), size=(img_size, img_size),
        mode='bilinear', align_corners=False
    ).squeeze(0)
    return t


# ─────────────────────────────────────────────
# MODEL  (EfficientNet-B3, same arch as training)
# ─────────────────────────────────────────────

class EfficientNetB3Classifier(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.4):
        super().__init__()
        backbone = models.efficientnet_b3(weights=None)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ─────────────────────────────────────────────
# LOAD CHECKPOINT
# The .ckpt is a PyTorch Lightning checkpoint.
# We strip the "model." prefix from state_dict keys.
# ─────────────────────────────────────────────

def load_model(ckpt_path: str) -> EfficientNetB3Classifier:
    model = EfficientNetB3Classifier(CFG['num_classes'])
    ckpt  = torch.load(ckpt_path, map_location='cpu')

    # Lightning wraps weights under "model." key
    raw_sd = ckpt.get('state_dict', ckpt)
    sd = {k.replace('model.', '', 1): v for k, v in raw_sd.items()
          if k.startswith('model.')}

    model.load_state_dict(sd, strict=True)
    model.eval()
    return model.to(DEVICE)


CKPT_PATH = 'model.ckpt'   # place your .ckpt here (see Steps below)
model = load_model(CKPT_PATH)
print(f'Model loaded on {DEVICE}')


# ─────────────────────────────────────────────
# INFERENCE  (5-crop TTA, same as notebook)
# ─────────────────────────────────────────────

@torch.no_grad()
def predict(audio_path: str) -> dict:
    if audio_path is None:
        return {"error": "No file uploaded"}

    crops = []
    for offset in TTA_OFFSETS:
        y      = load_audio(audio_path, CFG['sr'], CFG['duration'], offset_frac=offset)
        spec   = audio_to_melspec(y, CFG)
        tensor = spec_to_tensor(spec, CFG['img_size'])
        crops.append(tensor)

    batch  = torch.stack(crops).to(DEVICE)          # (5, 3, 224, 224)
    logits = model(batch)                            # (5, 10)
    probs  = F.softmax(logits, dim=1).mean(dim=0)   # (10,)  averaged TTA

    pred_idx   = probs.argmax().item()
    confidence = probs[pred_idx].item()

    # return dict for Gradio Label component
    return {IDX_TO_GENRE[i]: float(probs[i]) for i in range(CFG['num_classes'])}


# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────

with gr.Blocks(title='Music Genre Classifier') as demo:
    gr.Markdown(
        """
        # 🎵 Music Genre Classifier
        Upload a `.wav` audio file (any length — the model uses a 10-second window with 5-crop TTA).

        **Supported genres:** Blues · Classical · Country · Disco · Hip-hop · Jazz · Metal · Pop · Reggae · Rock
        """
    )

    with gr.Row():
        audio_input = gr.Audio(type='filepath', label='Upload WAV file')

    predict_btn = gr.Button('Predict Genre', variant='primary')

    label_output = gr.Label(num_top_classes=5, label='Predicted Genre (top-5 probabilities)')

    predict_btn.click(fn=predict, inputs=audio_input, outputs=label_output)

    gr.Examples(
        examples=[],   # add sample .wav paths here if you want
        inputs=audio_input
    )

demo.launch()