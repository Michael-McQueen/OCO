
import os
import pickle
from typing import Any, Dict, Iterable, Set, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.postprocessors.base_postprocessor import BasePostprocessor

Pattern = Tuple[Tuple[int, int], ...]


class OCOPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.cache_path = getattr(self.args, 'f_train_cache',
                                  './results/oco_f_train.pkl')

        self.f_train_patterns: Set[Pattern] = set()
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return

        if not os.path.exists(self.cache_path):
            raise FileNotFoundError(
                f"OCO F_train cache not found at '{self.cache_path}'.\n"
                "Run Stage B first: "
                "python scripts/ood/oco/build_f_train_minimal_strict.py --config <...>"
            )

        with open(self.cache_path, 'rb') as f:
            cache_obj = pickle.load(f)

        if not isinstance(cache_obj, dict):
            raise RuntimeError('Invalid OCO cache format: expected dict.')
        if 'f_train_patterns' not in cache_obj:
            raise RuntimeError("Invalid OCO cache: missing key 'f_train_patterns'.")

        self.f_train_patterns = self._normalize_patterns(cache_obj['f_train_patterns'])

        if comm.is_main_process():
            print(f'[OCO] Loaded cache: {len(self.f_train_patterns)} '
                  f'multi-class F_train patterns from {self.cache_path}')
        self.setup_flag = True

    @staticmethod
    def _to_frequency_pattern(slot_indices: Iterable[int]) -> Pattern:
        counts: Dict[int, int] = {}
        for cls in slot_indices:
            cls_i = int(cls)
            counts[cls_i] = counts.get(cls_i, 0) + 1
        return tuple(sorted(counts.items(), key=lambda x: x[0]))

    @staticmethod
    def _normalize_patterns(raw_patterns) -> Set[Pattern]:
        patterns: Set[Pattern] = set()
        for p in raw_patterns:
            normalized = tuple(
                sorted(((int(cls), int(freq)) for cls, freq in p),
                       key=lambda x: x[0]))
            if len(normalized) >= 2:
                patterns.add(normalized)
        return patterns

    def _route_scene(self, pattern: Pattern) -> str:
        if len(pattern) == 1:
            return 'S1'
        if pattern in self.f_train_patterns:
            return 'S2'
        return 'S3'

    @staticmethod
    def _class_evidence(slot_indices_1d: torch.Tensor,
                        slot_prior_1d: torch.Tensor) -> Dict[int, float]:
        """p_c^max: highest slot confidence for each class in F_t."""
        ev: Dict[int, float] = {}
        for cls, p in zip(slot_indices_1d.tolist(), slot_prior_1d.tolist()):
            cls_i = int(cls)
            p_f = float(p)
            if cls_i not in ev or p_f > ev[cls_i]:
                ev[cls_i] = p_f
        return ev

    @staticmethod
    def _bel(p1: float, p2: float) -> float:
        """Paper Eq. (7): Bel(c', c)."""
        return p1 * p2 + p1 * (1.0 - p2) + (1.0 - p1) * p2

    def _s_amb(self, evidence: Dict[int, float]) -> float:
        """Paper Eq. (8): average pairwise belief with dominant class."""
        if len(evidence) <= 1:
            return float(max(evidence.values())) if evidence else 0.0

        dominant_cls = max(evidence.items(), key=lambda x: x[1])[0]
        p_dom = float(evidence[dominant_cls])

        total = 0.0
        count = 0
        for cls, p_cls in evidence.items():
            if cls == dominant_cls:
                continue
            total += self._bel(p_dom, float(p_cls))
            count += 1

        return total / max(count, 1)

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        net.eval()
        out = net(data, return_oco_dict=True)

        logits = out['logits']
        slots_logits = out['slot_logits']

        # Scene-level confidence P_t
        p_scene, pred = logits.softmax(dim=-1).max(dim=-1)

        # Slot-level top1 confidence/class for each slot
        slot_prior, slot_indices = slots_logits.softmax(dim=-1).max(dim=-1)

        # p_t^max
        p_slot_max = slot_prior.max(dim=-1).values

        B = data.size(0)
        scores = torch.zeros((B,), device=data.device)

        for b in range(B):
            pattern = self._to_frequency_pattern(slot_indices[b].tolist())
            scene = self._route_scene(pattern)

            # S1
            if scene == 'S1':
                scores[b] = p_scene[b] * p_slot_max[b]
                continue

            # S2
            if scene == 'S2':
                evidence = self._class_evidence(slot_indices[b], slot_prior[b])
                scores[b] = self._s_amb(evidence)
                continue

            # S3
            scores[b] = p_slot_max[b]

        return pred, scores

    def inference(self, net: nn.Module, data_loader: DataLoader,
                  progress: bool = True):
        pred_list, conf_list, label_list = [], [], []
        for batch in tqdm(data_loader,
                          disable=not progress or not comm.is_main_process()):
            data = batch['data'].cuda()
            label = batch['label'].cuda()
            pred, conf = self.postprocess(net, data)
            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())

        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)
        return pred_list, conf_list, label_list
