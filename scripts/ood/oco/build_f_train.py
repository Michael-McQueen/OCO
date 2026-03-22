import os
import pickle
import sys
from typing import Dict, Iterable, Set, Tuple

import torch
from tqdm import tqdm

# ---- Make openood importable from repo root ----
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, _REPO_ROOT)

from openood.datasets import get_dataloader
from openood.networks import get_network
from openood.utils import setup_config

Pattern = Tuple[Tuple[int, int], ...]


def to_frequency_pattern(slot_indices: Iterable[int]) -> Pattern:
    """Convert one sample's slot argmax classes into the paper's F_i form."""
    counts: Dict[int, int] = {}
    for cls in slot_indices:
        cls_i = int(cls)
        counts[cls_i] = counts.get(cls_i, 0) + 1
    return tuple(sorted(counts.items(), key=lambda x: x[0]))


def build_f_train(config, output_path: str):
    print('[OCO Stage B] Loading network...')
    net = get_network(config.network)
    net.eval()
    net.cuda()

    print('[OCO Stage B] Loading ID train dataloader...')
    loader_dict = get_dataloader(config)
    train_loader = loader_dict['train']

    f_train_patterns: Set[Pattern] = set()
    n_total = 0
    n_multiclass = 0

    with torch.no_grad():
        for batch in tqdm(train_loader, desc='Building F_train'):
            data = batch['data'].cuda()
            out = net(data, return_oco_dict=True)

            slot_logits = out['slot_logits']
            slot_indices = slot_logits.softmax(dim=-1).argmax(dim=-1)  # [B, K]

            for row in slot_indices.cpu().tolist():
                n_total += 1
                pattern = to_frequency_pattern(row)
                if len(pattern) >= 2:
                    f_train_patterns.add(pattern)
                    n_multiclass += 1

    if n_total == 0:
        raise RuntimeError('ID train dataloader is empty; cannot build F_train.')

    cache_obj = {
        'f_train_patterns': sorted(f_train_patterns),
    }

    print(f'[OCO Stage B] Processed {n_total} training images.')
    print(f'[OCO Stage B] Multi-class training images: {n_multiclass}')
    print(f'[OCO Stage B] Unique multi-class F_train patterns: '
          f'{len(f_train_patterns)}')

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(cache_obj, f)
    print(f'[OCO Stage B] F_train saved -> {output_path}')


if __name__ == '__main__':
    # Parse --output before handing control to OpenOOD's setup_config.
    import copy

    _argv_orig = copy.copy(sys.argv)
    output_path = './results/oco_f_train.pkl'
    _clean_argv = [sys.argv[0]]

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--output':
            if i + 1 < len(sys.argv):
                output_path = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        else:
            _clean_argv.append(sys.argv[i])
            i += 1
    sys.argv = _clean_argv

    config = setup_config()

    # Ensure Stage B uses the trained Stage A checkpoint if provided.
    ckpt = getattr(config.network, 'checkpoint', None)
    if ckpt not in (None, 'none'):
        config.network.pretrained = True

    build_f_train(config, output_path=output_path)
