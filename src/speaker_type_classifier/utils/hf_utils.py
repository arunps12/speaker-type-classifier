from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class HFEmbedder:
    model_name: str
    device: str = "cuda"     # cuda|cpu|auto
    pooling: str = "mean"    # mean|first
    use_fp16: bool = True

    def __post_init__(self):
        import torch
        from transformers import AutoProcessor, AutoModel

        if self.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.device

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self._device)
        self.model.eval()

        self._use_fp16 = bool(self.use_fp16 and self._device == "cuda")
        self._torch = torch

    @property
    def device_resolved(self) -> str:
        return self._device

    def embed_batch(self, audios_16k: List[np.ndarray]) -> np.ndarray:
        """
        audios_16k: list of 1D float32 arrays at 16kHz sampling rate
        Returns: (B, D) float32
        """
        torch = self._torch

        inputs = self.processor(
            audios_16k,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        for k in inputs:
            inputs[k] = inputs[k].to(self._device)

        with torch.no_grad():
            if self._use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model(**inputs)
            else:
                out = self.model(**inputs)

            hidden = out.last_hidden_state  # (B, T, D)

            if self.pooling == "first":
                emb = hidden[:, 0, :]
            else:
                emb = hidden.mean(dim=1)

        return emb.detach().float().cpu().numpy().astype(np.float32)
