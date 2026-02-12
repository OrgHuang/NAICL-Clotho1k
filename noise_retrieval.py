import os
import torch
import torchaudio
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from model.beats.BEATs import BEATs, BEATsConfig


# =========================
# Config
# =========================
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Data structure
# =========================
@dataclass
class NoiseItem:
    audio_path: str
    caption: str
    embedding: torch.Tensor  # (D,)


# =========================
# Utility functions
# =========================
def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Load audio and resample to target_sr.
    Return: Tensor [1, T]
    """
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def mean_pooling(features: torch.Tensor, padding_mask: torch.Tensor):
    """
    features: (B, T, D)
    padding_mask: (B, T)
    """
    mask = (~padding_mask).unsqueeze(-1)
    features = features * mask
    return features.sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor):
    """
    a: (D,)
    b: (N, D)
    """
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return torch.matmul(b, a)


# =========================
# BEATs Encoder
# =========================
class BEATsEncoder:
    def __init__(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cfg = BEATsConfig(checkpoint["cfg"])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.model.to(DEVICE)

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (1, T)
        return: (D,)
        """
        wav = wav.to(DEVICE)
        padding_mask = torch.zeros_like(wav).bool()
        feats = self.model.extract_features(
            wav, padding_mask=padding_mask
        )[0]  # (B, T, D)
        pooled = mean_pooling(feats, padding_mask)
        return pooled.squeeze(0).cpu()


# =========================
# Noise Knowledge Base
# =========================
class NoiseKnowledgeBase:
    def __init__(self, beats_ckpt: str):
        self.encoder = BEATsEncoder(beats_ckpt)
        self.noise_items: List[NoiseItem] = []

    def add_noise(self, audio_path: str, caption: str):
        wav = load_audio(audio_path)
        emb = self.encoder.encode(wav)
        self.noise_items.append(
            NoiseItem(
                audio_path=audio_path,
                caption=caption,
                embedding=emb,
            )
        )

    def build_from_list(self, noise_list: List[Dict]):
        """
        noise_list example:
        [
            {
                "audio_path": ".../white_noise_01.wav",
                "caption": "Broadband white noise with no semantic events."
            },
            ...
        ]
        """
        for item in noise_list:
            self.add_noise(item["audio_path"], item["caption"])

    def retrieve(self, query_audio_path: str, topk: int = 3):
        wav = load_audio(query_audio_path)
        query_emb = self.encoder.encode(wav)

        embeddings = torch.stack(
            [n.embedding for n in self.noise_items], dim=0
        )  # (N, D)

        sims = cosine_similarity(query_emb, embeddings)
        topk_idx = torch.topk(sims, k=topk).indices.tolist()

        results = []
        for idx in topk_idx:
            item = self.noise_items[idx]
            results.append(
                {
                    "audio_path": item.audio_path,
                    "caption": item.caption,
                    "similarity": sims[idx].item(),
                }
            )
        return results

