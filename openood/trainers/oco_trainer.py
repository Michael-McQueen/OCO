import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import openood.utils.comm as comm
from openood.utils import Config


def _get_module(net):
    """Unwrap DDP to access the underlying module."""
    if isinstance(net, torch.nn.parallel.DistributedDataParallel):
        return net.module
    return net


class OCOTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:

        self.net          = net
        self.train_loader = train_loader
        self.config       = config

        # --- Trainer args -------------------------
        trainer_args      = getattr(config.trainer, 'trainer_args', {})
        self.recon_weight = float(getattr(trainer_args, 'recon_loss_weight', 1.0))

        # --- Optimizer: AdamW on trainable params only --------------------
        raw_net = _get_module(net)
        self.trainable_params = [
            p for name, p in raw_net.named_parameters()
            if 'visual_encoder' not in name and p.requires_grad
        ]

        opt_cfg = config.optimizer
        lr_init        = float(getattr(opt_cfg, 'lr',           4e-4))
        lr_final       = float(getattr(opt_cfg, 'final_lr',     4e-5))
        weight_decay   = float(getattr(opt_cfg, 'weight_decay', 0.05))
        self.num_epochs = int(getattr(opt_cfg, 'num_epochs',    20))
        warmup_epochs   = int(getattr(opt_cfg, 'warmup_epochs', 1))

        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=lr_init,
            weight_decay=weight_decay,
        )

        # --- Cosine LR schedule: lr_init → lr_final ----------------------
        total_steps  = self.num_epochs * len(train_loader)
        warmup_steps = warmup_epochs  * len(train_loader)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            scale    = lr_final / lr_init
            return scale + (1.0 - scale) * cosine

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )

    # -----------------------------------------------------------------------
    def train_epoch(self, epoch_idx):
        self.net.train()
        # Keep backbone frozen even in train() mode
        _get_module(self.net).visual_encoder.eval()

        loss_avg   = 0.0
        recon_avg  = 0.0
        ce_avg     = 0.0
        train_iter = iter(self.train_loader)

        for step in tqdm(range(1, len(train_iter) + 1),
                         desc='Epoch {:03d}: '.format(epoch_idx),
                         position=0,
                         leave=True,
                         disable=not comm.is_main_process()):

            batch  = next(train_iter)
            data   = batch['data'].cuda()
            target = batch['label'].cuda()

            # Forward 
            out = self.net(data, return_oco_dict=True)
            logits       = out['logits']        # [B, C]
            feat_tokens  = out['feat_tokens']   # [B, 196, 768]  (DINO tokens, detached)
            recon_tokens = out['recon_tokens']  # [B, 196, 768]

            # Losses
            loss_ce    = F.cross_entropy(logits, target)
            loss_recon = F.mse_loss(recon_tokens, feat_tokens.detach())
            loss       = loss_ce + self.recon_weight * loss_recon

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.trainable_params, max_norm=1.0
            )
            self.optimizer.step()
            self.scheduler.step()

            # EMA for display
            with torch.no_grad():
                loss_avg  = loss_avg  * 0.8 + float(loss)       * 0.2
                recon_avg = recon_avg * 0.8 + float(loss_recon) * 0.2
                ce_avg    = ce_avg    * 0.8 + float(loss_ce)    * 0.2

        metrics = {
            'epoch_idx':  epoch_idx,
            'loss':       self._gather(loss_avg),
            'loss_ce':    self._gather(ce_avg),
            'loss_recon': self._gather(recon_avg),
        }
        return self.net, metrics

    def _gather(self, value):

        all_vals = comm.gather(value) 
        if not all_vals:             
            return 0.0
        return float(np.mean([x for x in all_vals]))
