"""Stage 1: prepare reproducible Omniglot ICL data from Singh et al. features.

Run from project root: python src/01_prepare_omniglot.py
Dependencies: numpy, torch, h5py
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

SOURCE_URL = (
    'https://raw.githubusercontent.com/aadityasingh/icl-dynamics/main/'
    'omniglot_resnet18_randomized_order_s0.h5'
)


def download_if_missing(path: Path) -> None:
    if path.exists():
        print(f'[OK] Found source embeddings: {path} ({path.stat().st_size / 2**20:.1f} MiB)')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    print(f'[DOWNLOAD] {SOURCE_URL}')
    try:
        urllib.request.urlretrieve(SOURCE_URL, partial)
        with h5py.File(partial, 'r') as h5:
            assert any(True for _ in list_datasets(h5)), 'No HDF5 datasets found'
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    print(f'[OK] Downloaded {path.stat().st_size / 2**20:.1f} MiB')


def list_datasets(group, prefix=''):
    for key, value in group.items():
        name = f'{prefix}/{key}' if prefix else key
        if isinstance(value, h5py.Dataset):
            yield name, value
        elif isinstance(value, h5py.Group):
            yield from list_datasets(value, name)


def read_embeddings(path: Path, dataset_key: str | None) -> tuple[np.ndarray, str]:
    with h5py.File(path, 'r') as h5:
        entries = [(name, tuple(ds.shape), str(ds.dtype)) for name, ds in list_datasets(h5)]
        print('[INFO] HDF5 datasets:')
        for name, shape, dtype in entries:
            print(f'  {name}: shape={shape}, dtype={dtype}')
        if dataset_key:
            if dataset_key not in h5:
                raise ValueError(f'Dataset key {dataset_key!r} not found. Choose one from above.')
            selected = dataset_key
        else:
            candidates = [name for name, shape, _ in entries if len(shape) == 3 and shape[0] >= 150 and shape[1] >= 5]
            if len(candidates) != 1:
                raise ValueError(f'Expected one [classes, exemplars, features] dataset, found {candidates}. Use --dataset-key.')
            selected = candidates[0]
        features = np.asarray(h5[selected], dtype=np.float32)
    if features.ndim != 3 or features.shape[0] < 150 or features.shape[1] < 5:
        raise ValueError(f'Unexpected feature shape {features.shape}; need [>=150, >=5, feature_dim]')
    if not np.isfinite(features).all():
        raise ValueError('Embeddings contain NaN or infinity')
    if np.max(np.linalg.norm(features, axis=-1)) == 0:
        raise ValueError('All embeddings are zero')
    return features, selected


def label_pair_split(seed: int, labels: int = 5):
    pairs = np.asarray(list(itertools.combinations(range(labels), 2)), dtype=np.int64)
    rng = np.random.default_rng(seed)
    pairs = pairs[rng.permutation(len(pairs))]
    return pairs[:8], pairs[8:]


class ICLSequenceDataset(Dataset):
    """Deterministically sampled 2-way/1-shot sequences, with 5 logical tokens.

    Each item: image_features [3,D], class_ids [3], labels [2], target,
    image_exemplar_ids [3]. Token order: image0,label0,image1,label1,query.
    The actual 5 x d_model embedding sequence is constructed in Stage 2.
    """

    def __init__(self, features: np.ndarray, class_ids: np.ndarray,
                 exemplar_ids: np.ndarray, allowed_pairs: np.ndarray,
                 count: int, seed: int):
        self.features = features
        self.class_ids = np.asarray(class_ids, dtype=np.int64)
        self.exemplar_ids = np.asarray(exemplar_ids, dtype=np.int64)
        self.allowed_pairs = np.asarray(allowed_pairs, dtype=np.int64)
        self.count = count
        self.seed = seed
        if len(self.class_ids) < 2 or len(self.exemplar_ids) < 1 or len(self.allowed_pairs) < 1:
            raise ValueError('Not enough classes, exemplars or label pairs')

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        # Per-index RNG: deterministic regardless of DataLoader worker count/order.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        selected = rng.choice(self.class_ids, size=2, replace=False)
        pair = self.allowed_pairs[int(rng.integers(len(self.allowed_pairs)))].copy()
        if rng.integers(2):
            pair = pair[::-1].copy()
        # Swap order of context classes independently of which is the query.
        query_side = int(rng.integers(2))
        ex0 = int(rng.choice(self.exemplar_ids))
        ex1 = int(rng.choice(self.exemplar_ids))
        query_ex = int(rng.choice(self.exemplar_ids))
        c0, c1 = map(int, selected)
        cq = (c0, c1)[query_side]
        target = int(pair[query_side])
        vectors = np.stack((self.features[c0, ex0], self.features[c1, ex1],
                            self.features[cq, query_ex])).astype(np.float32)
        return {
            'image_features': torch.from_numpy(vectors),
            'class_ids': torch.tensor([c0, c1, cq], dtype=torch.long),
            'exemplar_ids': torch.tensor([ex0, ex1, query_ex], dtype=torch.long),
            'context_labels': torch.tensor(pair.copy(), dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long),
        }


def validate_split(name: str, ds: ICLSequenceDataset, allowed: set[tuple[int, int]]):
    seen_query_sides = set()
    for i in range(min(1000, len(ds))):
        x = ds[i]
        c = x['class_ids'].tolist()
        labels = x['context_labels'].tolist()
        target = int(x['target'])
        assert c[0] != c[1], f'{name}: context classes repeat'
        assert c[2] in c[:2], f'{name}: query absent from context'
        assert labels[0] != labels[1], f'{name}: labels repeat'
        assert tuple(sorted(labels)) in allowed, f'{name}: label pair leakage'
        assert target == labels[c[:2].index(c[2])], f'{name}: target mismatch'
        assert x['image_features'].shape == (3, ds.features.shape[-1])
        seen_query_sides.add(c[:2].index(c[2]))
    assert seen_query_sides == {0, 1}, f'{name}: query not balanced across positions'
    x = ds[0]
    y = ds[0]
    assert all(torch.equal(x[k], y[k]) for k in x), f'{name}: not reproducible'
    print(f'[PASS] {name}: {len(ds)} sequences, 1000-or-fewer checked')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('data/raw/omniglot_resnet18_randomized_order_s0.h5'))
    parser.add_argument('--output', type=Path, default=Path('data/processed'))
    parser.add_argument('--dataset-key', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-samples', type=int, default=78400)
    parser.add_argument('--eval-samples', type=int, default=5000)
    args = parser.parse_args()
    if args.train_samples < 1 or args.eval_samples < 1:
        parser.error('Sample counts must be positive')

    print('=' * 70)
    print('STAGE 1: OMNIGLOT IN-CONTEXT LEARNING DATA PREPARATION')
    print('=' * 70)
    download_if_missing(args.source)
    features, key = read_embeddings(args.source, args.dataset_key)
    n_classes, n_exemplars, d_feature = features.shape
    # The original H5 was already randomly ordered using the authors' seed 0.
    train_classes = np.arange(0, 50, dtype=np.int64)
    test_classes = np.arange(n_classes - 100, n_classes, dtype=np.int64)
    train_exemplars = np.asarray([0], dtype=np.int64)
    val_exemplars = np.arange(1, 5, dtype=np.int64)
    train_pairs, heldout_pairs = label_pair_split(args.seed)
    assert set(train_classes).isdisjoint(set(test_classes))
    assert set(train_exemplars).isdisjoint(set(val_exemplars))
    assert set(map(tuple, train_pairs)).isdisjoint(set(map(tuple, heldout_pairs)))
    print(f'[INFO] Embeddings: {n_classes} classes x {n_exemplars} exemplars x {d_feature} features')
    print(f'[INFO] Train classes: {len(train_classes)}; held-out classes: {len(test_classes)}')
    print(f'[INFO] Training label pairs: {train_pairs.tolist()}')
    print(f'[INFO] Held-out label pairs: {heldout_pairs.tolist()}')

    config = {
        'train': (train_classes, train_exemplars, train_pairs, args.train_samples, args.seed + 10),
        'test_relabel': (train_classes, train_exemplars, heldout_pairs, args.eval_samples, args.seed + 20),
        'test_exemplars': (train_classes, val_exemplars, train_pairs, args.eval_samples, args.seed + 30),
        'test_classes': (test_classes, train_exemplars, train_pairs, args.eval_samples, args.seed + 40),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        'source_url': SOURCE_URL,
        'source_path': str(args.source),
        'h5_dataset_key': key,
        'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'feature_shape': list(features.shape),
        'seed': args.seed,
        'train_classes': train_classes.tolist(),
        'test_classes': test_classes.tolist(),
        'train_exemplars': train_exemplars.tolist(),
        'val_exemplars': val_exemplars.tolist(),
        'train_label_pairs': train_pairs.tolist(),
        'heldout_label_pairs': heldout_pairs.tolist(),
        'splits': {},
        'note': 'Precomputed features are from authors. Sampled sequences and pair partition are our PyTorch adaptation.',
    }
    for name, (classes, exemplars, pairs, count, seed) in config.items():
        ds = ICLSequenceDataset(features, classes, exemplars, pairs, count, seed)
        validate_split(name, ds, set(map(tuple, pairs.tolist())))
        # Save only indices and labels, not repeated high-dimensional vectors.
        cls = np.empty((count, 3), dtype=np.int32)
        ex = np.empty((count, 3), dtype=np.int16)
        labs = np.empty((count, 2), dtype=np.int16)
        targets = np.empty(count, dtype=np.int16)
        for i in range(count):
            item = ds[i]
            cls[i] = item['class_ids'].numpy()
            ex[i] = item['exemplar_ids'].numpy()
            labs[i] = item['context_labels'].numpy()
            targets[i] = int(item['target'])
        filename = f'{name}_sequences.npz'
        np.savez_compressed(args.output / filename,
                            class_ids=cls, exemplar_ids=ex, context_labels=labs, targets=targets)
        manifest['splits'][name] = {'file': filename, 'count': count, 'seed': seed}
        print(f'[SAVE] {args.output / filename}')
    path = args.output / 'stage1_manifest.json'
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'[SAVE] {path}')
    print('[SUCCESS] All Stage 1 validations passed.')


if __name__ == '__main__':
    main()
