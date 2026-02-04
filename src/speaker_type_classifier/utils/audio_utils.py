from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np


def load_audio_mono(path: Path) -> Tuple[np.ndarray, int]:
    """
    Load audio as mono float32. Returns (audio, sr).
    Uses soundfile if available .
    """
    path = Path(path)

    try:
        import soundfile as sf
        x, sr = sf.read(str(path), always_2d=True)
        x = x.mean(axis=1).astype(np.float32)
        return x, int(sr)
    except Exception:
        pass

    # fallback wav-only
    from scipy.io import wavfile
    sr, x = wavfile.read(str(path))
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    else:
        x = x.astype(np.float32)
    return x, int(sr)


def resample_audio(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return x.astype(np.float32)

    # torchaudio if available
    try:
        import torch
        import torchaudio
        xt = torch.from_numpy(x).unsqueeze(0)  # (1,T)
        y = torchaudio.functional.resample(xt, orig_sr, target_sr)
        return y.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception:
        pass

    # fallback scipy
    from scipy.signal import resample_poly
    import math
    g = math.gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    y = resample_poly(x, up, down)
    return y.astype(np.float32)


def crop_or_pad(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) >= target_len:
        return x[:target_len].astype(np.float32)
    out = np.zeros(target_len, dtype=np.float32)
    out[: len(x)] = x.astype(np.float32)
    return out
