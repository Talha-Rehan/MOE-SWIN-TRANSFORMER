# mmdet/models/backbones/swin_moe.py
from __future__ import annotations
import torch.nn as nn
from mmengine.registry import MODELS
from .swin import SwinTransformer
from .bricks.moe_mlp import MoEMlp

@MODELS.register_module()
class SwinTransformerMoE(SwinTransformer):
    """
    Wrapper that loads a normal Swin checkpoint, then replaces every MLP-like
    block (has fc1 & fc2 Linear layers) with a Partially-Shared MoE-MLP:
      - shared expand (fc1_shared) + act + drop
      - router on hidden
      - expert-specific contract (per-expert fc2)
    Copies fc1/fc2 weights for a warm start.
    """

    def __init__(
        self,
        num_experts: int = 4,
        topk: int = 1,
        moe_drop: float = 0.0,
        gate_temperature: float = 1.0,
        use_prob_scale: bool = True,
        balance_loss_weight: float = 0.01,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._moe_applied = False
        self.moe_cfg = dict(
            num_experts=num_experts,
            topk=topk,
            drop=moe_drop,
            gate_temperature=gate_temperature,
            use_prob_scale=use_prob_scale,
            balance_loss_weight=balance_loss_weight,   # NEW

        )

    def init_weights(self):
        # 1) load standard Swin weights (includes original MLPs)
        super().init_weights()
        # 2) swap MLP-like modules to MoE and copy weights
        if not self._moe_applied:
            self._swap_all_mlps_to_moe(copy_weights=True)
            self._moe_applied = True

    def _swap_all_mlps_to_moe(self, copy_weights: bool = True):
        def recursive_swap(parent: nn.Module):
            for name, child in list(parent.named_children()):
                # Detect an MLP-like block by fc1/fc2 Linear layers (works across versions)
                if hasattr(child, 'fc1') and hasattr(child, 'fc2') \
                   and isinstance(child.fc1, nn.Linear) and isinstance(child.fc2, nn.Linear):
                    in_dim = child.fc1.in_features
                    hid_dim = child.fc1.out_features
                    # Try to keep the same dropout prob if present
                    drop_p = 0.0
                    if hasattr(child, 'drop') and isinstance(child.drop, nn.Dropout):
                        drop_p = float(child.drop.p)

                    moe = MoEMlp(
                        in_features=in_dim,
                        hidden_features=hid_dim,
                        num_experts=self.moe_cfg['num_experts'],
                        topk=self.moe_cfg['topk'],
                        drop=self.moe_cfg['drop'] if self.moe_cfg['drop'] is not None else drop_p,
                        gate_temperature=self.moe_cfg['gate_temperature'],
                        use_prob_scale=self.moe_cfg['use_prob_scale'],
                    )
                    moe.balance_loss_weight = self.moe_cfg['balance_loss_weight']  # NEW

                    if copy_weights:
                        moe.init_from_mlp(child)  # copy fc1->shared, fc2->each expert
                    parent._modules[name] = moe  # swap in-place
                else:
                    recursive_swap(child)
        recursive_swap(self)
