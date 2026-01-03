# mmdet/models/backbones/bricks/moe_mlp.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS

@MODELS.register_module()
class MoEMlp(nn.Module):
    """
    Partially-shared Swin-FFN with MoE only on the contract (down-projection) layer.

    Structure (per token):
      H = act( x @ W_up_shared + b_up ) ;  # shared expand to hidden_features
      p = softmax( gate(H) / temperature );  # gating on shared features
      y = sum_k p_k * ( H @ W_down^{expert_k} + b_down^{expert_k} ) ;  # expert-specific contract
      return y

    Shapes match the stock Swin MLP: input (..., C) -> output (..., C).
    Notes:
      * topk=1 recommended for single-GPU stability.
      * 'use_prob_scale=True' multiplies expert outputs by gate probs.
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_experts: int = 4,
        topk: int = 1,
        drop: float = 0.0,
        gate_temperature: float = 1.0,
        use_prob_scale: bool = True,
    ):
        super().__init__()
        assert 1 <= topk <= num_experts
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_experts = num_experts
        self.topk = topk
        self.temperature = gate_temperature
        self.use_prob_scale = use_prob_scale

        # --- Shared expand (fc1) + activation + (optional) dropout ---
        self.fc1_shared = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop) if drop > 0 else nn.Identity()

        # --- Expert-specific contract (fc2) (+ optional dropout) ---
        experts = []
        for _ in range(num_experts):
            experts.append(nn.Sequential(
                nn.Linear(hidden_features, in_features),            # contract only
                nn.Dropout(drop) if drop > 0 else nn.Identity(),
            ))
        self.experts = nn.ModuleList(experts)

        # --- Gating operates on the shared hidden H ---
        self.gate = nn.Linear(hidden_features, num_experts)
        # init gate near-uniform
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        #------------------------------------------------#
        self.balance_loss_weight = 0.01  # make this configurable later if you like
        self.register_buffer('_last_balance_loss', torch.zeros(()), persistent=False)

    @torch.no_grad()
    def init_from_mlp(self, mlp_like) -> None:
        """
        Warm-start from the original Swin MLP (with fc1/fc2):
          - copy fc1 -> fc1_shared
          - copy fc2 -> each expert's contract layer
        """
        assert hasattr(mlp_like, 'fc1') and hasattr(mlp_like, 'fc2'), \
            "Expected original MLP with fc1/fc2 to warm start."

        # copy shared expand
        self.fc1_shared.weight.copy_(mlp_like.fc1.weight)
        self.fc1_shared.bias.copy_(mlp_like.fc1.bias)

        # copy contract into every expert
        for exp in self.experts:
            # exp[0] is Linear(hidden->in)
            exp[0].weight.copy_(mlp_like.fc2.weight)
            exp[0].bias.copy_(mlp_like.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., C)
        orig_shape = x.shape
        C = orig_shape[-1]
        x_flat = x.reshape(-1, C)  # [T, C]  (T = total tokens)

        # Shared expand -> hidden
        H = self.fc1_shared(x_flat)             # [T, H]
        H = self.act(H)
        H = self.drop1(H)

        # Gating on hidden
        logits = self.gate(H) / self.temperature         # [T, E]
        probs = F.softmax(logits, dim=-1)                # [T, E]

        # ---- Switch-style load-balancing aux loss ----
        T, E = probs.shape
        if self.topk == 1:
            assignments = probs.argmax(dim=-1)                     # [T]
            load = torch.bincount(assignments, minlength=E).float().to(probs.device) / max(T, 1)
        else:
            # soft approx for top-k: total prob mass per expert
            load = probs.sum(dim=0) / max(T, 1)

        imp = probs.sum(dim=0) / max(T, 1)                         # average prob mass
        aux = E * (imp * load).sum()                               # scalar
        self._last_balance_loss = aux * self.balance_loss_weight
        # ------------------------------------------------

        if self.topk == 1:
            # route each token to exactly one expert
            top1 = probs.argmax(dim=-1)  # [T]
            out = x_flat.new_zeros(x_flat.shape)
            for e_idx, expert in enumerate(self.experts):
                mask = (top1 == e_idx)
                if mask.any():
                    y = expert(H[mask])                 # contract only
                    if self.use_prob_scale:
                        y = probs[mask, e_idx].unsqueeze(-1) * y
                    out[mask] = y
        else:
            # top-k routing: sum expert outputs weighted by probs for those k
            topk_vals, topk_idx = torch.topk(probs, k=self.topk, dim=-1)  # [T, k] each
            out = x_flat.new_zeros(x_flat.shape)
            for k in range(self.topk):
                idxk = topk_idx[:, k]      # [T]
                wk   = topk_vals[:, k]     # [T]
                for e_idx, expert in enumerate(self.experts):
                    mask = (idxk == e_idx)
                    if mask.any():
                        y = expert(H[mask])
                        # always scale by wk for k>1 (mixture sum)
                        y = wk[mask].unsqueeze(-1) * y
                        out[mask] += y

        return out.reshape(orig_shape)
    
    def get_aux_loss(self, reset: bool = True):
        v = self._last_balance_loss
        if reset:
            self._last_balance_loss = torch.zeros_like(v)
        return v
