"""Stage 2: attention-only Transformer for Omniglot in-context learning.

Run from project root:
    python src/02_transformer_model.py
    python src/02_transformer_model.py --smoke-train

Requires Stage 1 files in data/raw and data/processed.
This is a PyTorch adaptation of Singh et al. (2024), not a bit-identical
reimplementation of the original JAX architecture.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class SavedICLDataset(Dataset):
    """Load Stage 1 indices; look up frozen ResNet features for three image tokens."""

    def __init__(self, root: Path, split: str, raw_file: Path):
        npz_path = root / f"{split}_sequences.npz"
        if not npz_path.exists() or not raw_file.exists():
            raise FileNotFoundError(f"Missing {npz_path} or {raw_file}. Run Stage 1 first.")
        with np.load(npz_path, allow_pickle=False) as z:
            self.classes = z["class_ids"].astype(np.int64)
            self.exemplars = z["exemplar_ids"].astype(np.int64)
            self.labels = z["context_labels"].astype(np.int64)
            self.targets = z["targets"].astype(np.int64)
        with h5py.File(raw_file, "r") as h5:
            self.features = np.asarray(h5["resnet18/224/feat"], dtype=np.float32)
        n = len(self.targets)
        if self.classes.shape != (n, 3) or self.exemplars.shape != (n, 3) or self.labels.shape != (n, 2):
            raise ValueError("Stage 1 arrays have unexpected shapes")
        if not (np.isfinite(self.features).all() and n > 0):
            raise ValueError("Invalid features or empty split")
        if np.any(self.classes < 0) or np.any(self.classes >= len(self.features)):
            raise ValueError("Class index out of bounds")
        if np.any(self.exemplars < 0) or np.any(self.exemplars >= self.features.shape[1]):
            raise ValueError("Exemplar index out of bounds")
        if np.any(self.labels < 0) or np.any(self.labels >= 5) or np.any(self.targets < 0) or np.any(self.targets >= 5):
            raise ValueError("Label out of range")
        if np.any(self.classes[:, 0] == self.classes[:, 1]):
            raise ValueError("Context classes must differ")
        if np.any(self.labels[:, 0] == self.labels[:, 1]):
            raise ValueError("Context labels must differ")
        matches0 = self.classes[:, 2] == self.classes[:, 0]
        matches1 = self.classes[:, 2] == self.classes[:, 1]
        if not np.all(matches0 ^ matches1):
            raise ValueError("Query must match exactly one context class")
        expected = np.where(matches0, self.labels[:, 0], self.labels[:, 1])
        if not np.array_equal(expected, self.targets):
            raise ValueError("Incorrect Stage 1 target labels")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        x = self.features[self.classes[i], self.exemplars[i]].copy()
        return {
            "image_features": torch.from_numpy(x),       # [3, 512]
            "context_labels": torch.from_numpy(self.labels[i].copy()),  # [2]
            "target": torch.tensor(int(self.targets[i]), dtype=torch.long),
        }


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """Rotate query/key vectors using their token positions. x: [B,H,T,D]."""
    d = x.shape[-1]
    if d % 2:
        raise ValueError("RoPE requires even head dimension")
    positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
    frequencies = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    angles = positions[:, None] * frequencies[None, :]
    cos = angles.cos().to(x.dtype)[None, None, :, :]
    sin = angles.sin().to(x.dtype)[None, None, :, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class CausalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads or (d_model // n_heads) % 2:
            raise ValueError("d_model must be divisible by n_heads and head_dim even")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, ablate_heads: tuple[int, ...] = ()):
        b, t, d = x.shape
        reshape = lambda y: y.reshape(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = reshape(self.q_proj(x)), reshape(self.k_proj(x)), reshape(self.v_proj(x))
        q, k = apply_rope(q), apply_rope(k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.ones((t, t), dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal_mask[None, None], torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        head_outputs = attention @ v  # [B,H,T,D_head]
        if ablate_heads:
            if any(h < 0 or h >= self.n_heads for h in ablate_heads):
                raise ValueError(f"Invalid head index in {ablate_heads}")
            # Zero per-head contribution AFTER attention and BEFORE output projection.
            head_mask = torch.ones((1, self.n_heads, 1, 1), device=x.device, dtype=x.dtype)
            head_mask[:, list(ablate_heads)] = 0
            head_outputs = head_outputs * head_mask
        joined = head_outputs.transpose(1, 2).contiguous().view(b, t, d)
        return self.out_proj(joined), attention


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = CausalAttention(d_model, n_heads)

    def forward(self, x, ablate_heads=()):
        update, pattern = self.attn(self.norm(x), ablate_heads)
        return x + update, pattern


class InductionTransformer(nn.Module):
    """2-layer, 8-head attention-only classifier with accessible attention maps."""

    def __init__(self, feature_dim=512, d_model=64, n_heads=8, n_layers=2, n_labels=5):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_labels = n_labels
        self.image_projection = nn.Linear(feature_dim, d_model)
        self.label_embedding = nn.Embedding(n_labels, d_model)
        self.blocks = nn.ModuleList([AttentionBlock(d_model, n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_labels)

    def encode(self, images, context_labels):
        if images.ndim != 3 or images.shape[1:] != (3, self.feature_dim):
            raise ValueError(f"Expected images [B,3,{self.feature_dim}], got {tuple(images.shape)}")
        if context_labels.shape != (images.shape[0], 2):
            raise ValueError("Expected context_labels [B,2]")
        if images.dtype != torch.float32:
            images = images.float()
        image_tokens = self.image_projection(images)
        label_tokens = self.label_embedding(context_labels)
        # image0, label0, image1, label1, query
        return torch.stack((image_tokens[:, 0], label_tokens[:, 0], image_tokens[:, 1],
                            label_tokens[:, 1], image_tokens[:, 2]), dim=1)

    def forward(self, image_features, context_labels, return_attention=False, ablate=None):
        x = self.encode(image_features, context_labels)
        maps = []
        ablate = {} if ablate is None else ablate
        if any(layer < 0 or layer >= len(self.blocks) for layer in ablate):
            raise ValueError("Invalid ablation layer")
        for layer, block in enumerate(self.blocks):
            x, pattern = block(x, tuple(ablate.get(layer, ())))
            if return_attention:
                maps.append(pattern)
        logits = self.classifier(self.final_norm(x[:, -1]))
        return (logits, torch.stack(maps, dim=1)) if return_attention else logits


def run_checks(model, batch, device):
    model.eval()
    images = batch["image_features"].to(device)
    labels = batch["context_labels"].to(device)
    targets = batch["target"].to(device)
    with torch.no_grad():
        logits, patterns = model(images, labels, return_attention=True)
        assert logits.shape == (len(images), 5)
        assert patterns.shape == (len(images), 2, 8, 5, 5)
        assert torch.isfinite(logits).all() and torch.isfinite(patterns).all()
        assert torch.allclose(patterns.sum(dim=-1), torch.ones_like(patterns[..., 0]), atol=1e-5)
        assert torch.all(patterns[..., torch.triu(torch.ones(5, 5, dtype=torch.bool, device=device), diagonal=1)] == 0)
        # Causality: changing final query image must not change earlier attention rows.
        altered = images.clone()
        altered[:, -1] = altered[:, -1] + 1.0
        _, changed = model(altered, labels, return_attention=True)
        assert torch.allclose(patterns[..., :4, :], changed[..., :4, :], atol=1e-6)
        # Output head ablation must leave finite predictions and preserve output shape.
        ablated = model(images, labels, ablate={1: (0,)})
        assert ablated.shape == logits.shape and torch.isfinite(ablated).all()
        # Labels should be in the expected range and target must be one of the two labels.
        assert torch.all((targets[:, None] == labels).any(dim=1))
    print("[PASS] Output dimensions: logits [B,5], attention [B,2,8,5,5]")
    print("[PASS] Causal attention mask and attention normalization")
    print("[PASS] Future-token perturbation cannot alter earlier attention")
    print("[PASS] Individual attention-head ablation")
    print("[PASS] Finite model outputs and valid targets")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/omniglot_resnet18_randomized_order_s0.h5"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-train", action="store_true", help="One optimizer step to check gradients, not Stage 3 training")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("=" * 70)
    print("STAGE 2: TWO-LAYER ATTENTION-ONLY TRANSFORMER")
    print("=" * 70)
    print("[INFO] Device: CPU" if not torch.cuda.is_available() else "[INFO] Device: CUDA")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_file = args.processed / "stage1_manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Missing {manifest_file}. Run Stage 1 first.")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    ds = SavedICLDataset(args.processed, "train", args.raw)
    print(f"[INFO] Loaded {len(ds):,} training sequences; feature dimension={ds.features.shape[-1]}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    model = InductionTransformer(feature_dim=ds.features.shape[-1]).to(device)
    print(f"[INFO] Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
    run_checks(model, batch, device)
    if args.smoke_train:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999))
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["image_features"].to(device), batch["context_labels"].to(device))
        loss = F.cross_entropy(logits, batch["target"].to(device))
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        print(f"[PASS] Single-step gradient/optimizer smoke test: loss={loss.item():.5f}")
    print("[SUCCESS] Stage 2 architecture and integration checks passed.")
    print("[NOTE] Model is untrained; Stage 3 will implement full training and evaluation.")


if __name__ == "__main__":
    main()
