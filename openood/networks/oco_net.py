import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init



# Inlined from DINOSAUR/models/model.py  

class _MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 residual=False, layer_order='none'):
        super().__init__()
        self.residual    = residual
        self.layer_order = layer_order
        if residual:
            assert input_dim == output_dim

        self.layer1    = nn.Linear(input_dim, hidden_dim)
        self.layer2    = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout   = nn.Dropout(p=0.1)

        if layer_order in ('pre', 'post'):
            self.norm = nn.LayerNorm(input_dim)
        else:
            assert layer_order == 'none'

    def forward(self, x):
        inp = x
        if self.layer_order == 'pre':
            x = self.norm(x)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.dropout(x)
        if self.residual:
            x = x + inp
        if self.layer_order == 'post':
            x = self.norm(x)
        return x


class _SA(nn.Module):
    """Vanilla Slot Attention — from DINOSAUR/models/model.py (class SA)."""

    def __init__(self, num_slots, slot_dim, slot_att_iter,
                 query_opt=False, input_dim=None):
        super().__init__()
        assert input_dim is not None, 'input_dim required'
        self.num_slots     = num_slots
        self.scale         = slot_dim ** -0.5
        self.iters         = slot_att_iter
        self.slot_dim      = slot_dim
        self.query_opt     = query_opt

        # Slot initialisation
        if query_opt:
            self.slots = nn.Parameter(torch.Tensor(1, num_slots, slot_dim))
            init.xavier_uniform_(self.slots)
        else:
            self.slots_mu        = nn.Parameter(torch.randn(1, 1, slot_dim))
            self.slots_logsigma  = nn.Parameter(torch.zeros(1, 1, slot_dim))
            init.xavier_uniform_(self.slots_mu)
            init.xavier_uniform_(self.slots_logsigma)

        # Slot Attention layers
        self.Q           = nn.Linear(slot_dim, slot_dim, bias=False)
        self.norm        = nn.LayerNorm(slot_dim)
        self.update_norm = nn.LayerNorm(slot_dim)
        self.gru         = nn.GRUCell(slot_dim, slot_dim)
        self.mlp         = _MLP(slot_dim, 4 * slot_dim, slot_dim,
                                residual=True, layer_order='pre')

        # Key / Value projections
        self.K = nn.Linear(slot_dim, slot_dim, bias=False)
        self.V = nn.Linear(slot_dim, slot_dim, bias=False)

        # Input preprocessing MLP (Slot style: 768 -> 768 -> 256)
        self.initial_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, slot_dim),
            nn.LayerNorm(slot_dim),
        )
        self.final_layer = nn.Linear(slot_dim, slot_dim)

    def forward(self, inputs):
        # inputs: [B, token, D]
        B        = inputs.shape[0]
        S        = self.num_slots
        D_slot   = self.slot_dim
        epsilon  = 1e-8

        if self.query_opt:
            slots = self.slots.expand(B, S, D_slot)
        else:
            mu    = self.slots_mu.expand(B, S, D_slot)
            sigma = self.slots_logsigma.exp().expand(B, S, D_slot)
            slots = mu + sigma * torch.randn(
                mu.shape, device=sigma.device, dtype=sigma.dtype)

        slots_init = slots
        inputs     = self.initial_mlp(inputs)          # [B, token, D_slot]
        keys       = self.K(inputs)                    # [B, token, D_slot]
        values     = self.V(inputs)                    # [B, token, D_slot]

        for t in range(self.iters):
            if t == self.iters - 1 and self.query_opt:
                slots = slots.detach() + slots_init - slots_init.detach()

            slots_prev = slots
            slots      = self.norm(slots)
            queries    = self.Q(slots)                 # [B, S, D_slot]

            dots = torch.einsum('bsd,btd->bst', queries, keys)   # [B, S, token]
            dots *= self.scale
            attn = dots.softmax(dim=1) + epsilon
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bst,btd->bsd', attn, values)  # [B, S, D_slot]

            slots = self.gru(
                updates.reshape(-1, D_slot),
                slots_prev.reshape(-1, D_slot),
            ).reshape(B, S, D_slot)
            slots = self.mlp(slots)

        return self.final_layer(slots)   # [B, S, D_slot]


class _Decoder(nn.Module):
    """DINOSAUR MLP decoder — from DINOSAUR/models/model.py (class Decoder)."""

    def __init__(self, slot_dim, hidden_dim=2048):
        super().__init__()
        self.layer1 = nn.Linear(slot_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, 768 + 1)   # 768 recon + 1 mask
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, slot_maps):
        # slot_maps: [B*K, token, D_slot]
        x = self.relu(self.layer1(slot_maps))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return self.layer4(x)              # [B*K, token, 769]



# DINO Visual Encoder

class DinoVisualEncoder(nn.Module):
    """
    Frozen DINO/DINOv2 backbone that returns patch tokens.
    CLS token is stripped.  All backbone parameters are frozen.

    Multi-GPU note: a dist.barrier() ensures only rank-0 downloads the hub
    weights and all other ranks wait before loading from cache.
    """

    def __init__(self, backbone='dinov2_vitb14'):
        super().__init__()
        self.backbone = backbone
        # Let rank-0 download first; other ranks wait at barrier then load from cache.
        import torch.distributed as dist
        is_dist = dist.is_available() and dist.is_initialized()

        if is_dist and dist.get_rank() != 0:
            dist.barrier()   # ranks 1-3 wait here

        if 'dinov2' in backbone:
            repo = 'facebookresearch/dinov2:main'
        else:
            repo = 'facebookresearch/dino:main'

        self.model = torch.hub.load(
            repo,
            backbone,
        )

        if is_dist and dist.get_rank() == 0:
            dist.barrier()   # rank-0 done downloading; release others

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()    # set once at init; re-enforced in forward

    @torch.no_grad()
    def forward(self, x):
        # x: [B, 3, 224, 224]
        self.model.eval()
        if 'dinov2' in self.backbone:
            out = self.model.forward_features(x)
            return out['x_norm_patchtokens']  # e.g., [B, 256, 768] for vitb14
        else:
            tokens = self.model.prepare_tokens(x)    # [B, 197, 768]
            for blk in self.model.blocks:
                tokens = blk(tokens)
            return tokens[:, 1:]                      # drop CLS -> [B, 196, 768]



# Slot Classifier

class SlotClassifier(nn.Module):
    """Shared linear head applied to each slot; global = sum over slots."""

    def __init__(self, slot_dim: int, num_classes: int, num_slots: int):
        super().__init__()
        self.head = nn.Linear(slot_dim, num_classes)
        # Add learnable weights for each slot to break permutation invariance (Slot style)
        self.weights = nn.Parameter(torch.ones(num_slots))

    def forward(self, slots):
        # slots: [B, K, slot_dim]
        slot_logits   = self.head(slots)           # [B, K, C]

        global_logits = 0
        for i in range(slot_logits.size(1)):
            global_logits += self.weights[i] * slot_logits[:, i, :]
        return slot_logits, global_logits



# OCONet

class OCONet(nn.Module):
    """
    Object-Centric OOD network.

    Architecture:
        image → DinoVisualEncoder → feat_tokens [B, 256, 768]
              → token_projector   → proj_tokens  [B, 256, slot_dim]
              → _SA               → slots        [B, K, slot_dim]
              → SlotClassifier    → slot_logits  [B, K, C]
                                  → logits       [B, C]
              → _Decoder          → recon_tokens [B, 256, 768]

    forward() returns a dict with all tensors needed by trainer and postprocessor.
    If return_feature=True (evaluator compatibility), returns (logits, slots.mean(1)).
    """

    def __init__(self, config):
        super().__init__()

        num_classes = config.num_classes
        num_slots   = getattr(config, 'num_slots',  6)
        slot_dim    = getattr(config, 'slot_dim',   256)
        sa_iters    = getattr(config, 'sa_iters',   3)
        token_dim   = getattr(config, 'token_dim',  768)  # DINO patch feature dim
        backbone    = getattr(config, 'backbone',   'dinov2_vitb14')
        dinosaur_checkpoint = getattr(config, 'dinosaur_checkpoint', '/dinov2_checkpoint_epoch_199.pt')

        self.num_slots = num_slots
        self.slot_dim  = slot_dim
        # 14 -> 256 tokens, 16 -> 196 tokens
        if '14' in backbone:
            self.token_num = 256
        else:
            self.token_num = 196

        # ---- Backbone (frozen) -------------------------------------------
        self.visual_encoder = DinoVisualEncoder(backbone=backbone)

        # ---- Slot Attention (trainable) ----------------------------------
        self.slot_attention = _SA(
            num_slots=num_slots,
            slot_dim=slot_dim,
            slot_att_iter=sa_iters,
            query_opt=False,
            input_dim=token_dim,
        )

        # ---- Classifier (trainable) -------------------------------------
        self.slot_classifier = SlotClassifier(slot_dim, num_classes, num_slots)

        # ---- Decoder (trainable) ----------------------------------------
        self.slot_decoder = _Decoder(slot_dim)

        # Positional broadcast embedding for decoder (DINOSAURpp.pos_dec)
        self.pos_dec = nn.Parameter(torch.empty(1, self.token_num, slot_dim))
        init.normal_(self.pos_dec, mean=0., std=0.02)

        # ---- Load pre-trained weights -----------------------------------
        self.load_dinosaur_weights(dinosaur_checkpoint)

    def load_dinosaur_weights(self, ckpt_path):
        import os
        if not os.path.exists(ckpt_path):
            print(f"[OCONet] Pre-trained weights not found at {ckpt_path}. Skipping.")
            return

        print(f"[OCONet] Loading pre-trained dinosaur weights from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('model', ckpt)

        # map checkpoint to 'slot_attention.' and 'slot_decoder.' respectively.
        mapped_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.slot_encoder.'):
                # Map slot_encoder to slot_attention
                new_k = k.replace('module.slot_encoder.', 'slot_attention.')
                mapped_dict[new_k] = v
            elif k.startswith('module.slot_decoder.'):
                # Map slot_decoder to slot_decoder
                new_k = k.replace('module.slot_decoder.', 'slot_decoder.')
                mapped_dict[new_k] = v
            elif k == 'module.pos_dec':
                mapped_dict['pos_dec'] = v

        # Load into the model, non-strict because backbone and classifier are not in dinosaur.pt
        missing, unexpected = self.load_state_dict(mapped_dict, strict=False)
        
        # Verify that the keys we actually loaded correspond to the mapped ones
        loaded_keys = set(mapped_dict.keys()) - set(unexpected)
        if len(loaded_keys) > 0:
            print(f"[OCONet] Successfully loaded {len(loaded_keys)} weight tensors from dinosaur.")
        else:
            print("[OCONet] WARNING: No matching keys found in dinosaur for current architecture.")



    # Helpers (mirrors DINOSAURpp internals)

    def _sbd_slots(self, slots):
        """
        Broadcast each slot to all token positions.
        slots: [B, K, D]  →  slot_maps: [B, K, token_num, D]
        """
        B, K, D = slots.shape
        # [B*K, 1, D] → [B*K, token_num, D]
        flat = slots.view(B * K, 1, D).expand(B * K, self.token_num, D)
        pos  = self.pos_dec.expand(B * K, self.token_num, D)
        flat = flat + pos
        return flat.view(B, K, self.token_num, D)

    def _reconstruct(self, slot_maps_out):
        """
        slot_maps_out: [B, K, token_num, 769]
        Returns:
            recon_tokens: [B, token_num, 768]
            masks:        [B, K, token_num]
        """
        channels, masks = slot_maps_out.split([768, 1], dim=-1)
        masks   = masks.softmax(dim=1)                  # [B, K, token_num, 1]
        recon   = (channels * masks).sum(dim=1)         # [B, token_num, 768]
        return recon, masks.squeeze(-1)


    # Forward

    def forward(self, x, return_feature=False, return_oco_dict=False):
        B = x.shape[0]

        # 1. Frozen backbone → patch tokens
        feat_tokens = self.visual_encoder(x)             # [B, 256, 768]

        # 2. Slot Attention (receives 768-D features directly)
        slots = self.slot_attention(feat_tokens)          # [B, K, slot_dim]

        # 3. Shared classifier
        slot_logits, global_logits = self.slot_classifier(slots)
        # slot_logits [B,K,C], global_logits [B,C]

        # Fast path: evaluator just needs logits tensor
        if not return_feature and not return_oco_dict:
            return global_logits

        if return_feature:
            cls_feature = slots.mean(dim=1)              # [B, slot_dim]
            return global_logits, cls_feature

        # return_oco_dict=True: trainer and postprocessor path
        # 5. Decode for reconstruction
        slot_maps = self._sbd_slots(slots)               # [B, K, token_num, D]
        flat      = slot_maps.view(B * self.num_slots, self.token_num, self.slot_dim)
        flat_out  = self.slot_decoder(flat)              # [B*K, token_num, 769]
        slot_maps_out = flat_out.view(B, self.num_slots, self.token_num, 769)
        recon_tokens, _ = self._reconstruct(slot_maps_out)  # [B, token_num, 768]

        # 6. Derived tensors for the postprocessor
        slot_prob = torch.softmax(slot_logits, dim=-1)   # [B, K, C]
        slot_pred = slot_logits.argmax(dim=-1)           # [B, K]

        return {
            'logits':       global_logits,   # [B, C]
            'slot_logits':  slot_logits,      # [B, K, C]
            'slot_prob':    slot_prob,        # [B, K, C]
            'slot_pred':    slot_pred,        # [B, K]
            'slots':        slots,            # [B, K, slot_dim]
            'feat_tokens':  feat_tokens,      # [B, 256, 768]
            'recon_tokens': recon_tokens,     # [B, 256, 768]
        }

    # Evaluator helpers
    def get_fc(self):
        w = self.slot_classifier.head.weight.cpu().detach().numpy()
        b = self.slot_classifier.head.bias.cpu().detach().numpy()
        return w, b

    def get_fc_layer(self):
        return self.slot_classifier.head
