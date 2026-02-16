# src/models/tcf.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TCFOutput:
    scores: torch.Tensor            # [B] sigmoid
    user_rep: torch.Tensor          # [B, K3+K1]  [U,E]
    item_rep: torch.Tensor          # [B, K3+K1]  [V,F]
    logits: torch.Tensor            # [B]


def build_mlp(dims: List[int], dropout: float = 0.0) -> nn.Sequential:
    """
    dims: [in, h1, h2, ..., out]
    """
    layers: List[nn.Module] = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(a, b))
        if b != dims[-1]:
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
    return nn.Sequential(*layers)


class TCF(nn.Module):
    """
    Text Collaborative Filtering (TCF) in TDAR.

    Paper definition:
      user rep  = [U, E]   (concat CF user embedding and fixed textual feature)
      item rep  = [V, F]
      R_hat_ui  = f([U,E]_u, [V,F]_i, Θ)

    where U,V,Θ trainable; E,F fixed (from TMN). :contentReference[oaicite:2]{index=2}

    This module supports:
      - interaction="mlp": f = MLP(concat(user_rep, item_rep)) -> sigmoid
      - interaction="dot": f = dot(Wu*user_rep, Wi*item_rep) -> sigmoid
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        k3_embed_dim: int,
        k1_text_dim: int,
        interaction: str = "mlp",              # "mlp" or "dot"
        mlp_hidden: Optional[List[int]] = None, # e.g. [256, 128]
        dropout: float = 0.0,
        reg_only_embeddings: bool = True,
    ):
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.k3 = int(k3_embed_dim)
        self.k1 = int(k1_text_dim)
        self.interaction = str(interaction).lower()
        self.reg_only_embeddings = bool(reg_only_embeddings)

        # CF embeddings U, V
        self.user_emb = nn.Embedding(self.num_users, self.k3)
        self.item_emb = nn.Embedding(self.num_items, self.k3)

        rep_dim = self.k3 + self.k1  # [U,E] or [V,F]

        # Interaction function f(., Θ)
        if self.interaction == "mlp":
            if mlp_hidden is None:
                mlp_hidden = [256, 128]
            dims = [2 * rep_dim] + list(map(int, mlp_hidden)) + [1]
            self.f = build_mlp(dims, dropout=dropout)
        elif self.interaction == "dot":
            self.user_proj = nn.Linear(rep_dim, rep_dim, bias=False)
            self.item_proj = nn.Linear(rep_dim, rep_dim, bias=False)
        else:
            raise ValueError(f"Unknown interaction: {interaction}")

        # Optional: store fixed textual features as buffers (precomputed by TMN)
        # E_all: [num_users, k1], F_all: [num_items, k1]
        self.register_buffer("E_all", None, persistent=False)
        self.register_buffer("F_all", None, persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        if self.interaction == "mlp":
            for m in self.f.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        elif self.interaction == "dot":
            nn.init.xavier_uniform_(self.user_proj.weight)
            nn.init.xavier_uniform_(self.item_proj.weight)

    @torch.no_grad()
    def set_text_features(self, E_all: torch.Tensor, F_all: torch.Tensor):
        """
        Cache fixed textual features (from TMN) inside the model.
        E_all: [num_users, k1]
        F_all: [num_items, k1]
        """
        if E_all.dim() != 2 or E_all.size(0) != self.num_users or E_all.size(1) != self.k1:
            raise ValueError(f"E_all must be [num_users, k1]=[{self.num_users},{self.k1}], got {tuple(E_all.shape)}")
        if F_all.dim() != 2 or F_all.size(0) != self.num_items or F_all.size(1) != self.k1:
            raise ValueError(f"F_all must be [num_items, k1]=[{self.num_items},{self.k1}], got {tuple(F_all.shape)}")

        self.E_all = E_all.detach()
        self.F_all = F_all.detach()

    def _get_text_batch(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        E_batch: Optional[torch.Tensor] = None,
        F_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prefer cached buffers; otherwise require E_batch/F_batch.
        """
        if self.E_all is not None and self.F_all is not None:
            E = self.E_all[user_idx]  # [B,k1]
            F_ = self.F_all[item_idx] # [B,k1]
            return E, F_

        if E_batch is None or F_batch is None:
            raise RuntimeError("TCF requires fixed textual features. Either call set_text_features(E_all,F_all) "
                               "or pass E_batch/F_batch to forward().")
        return E_batch, F_batch

    def forward(
        self,
        user_idx: torch.Tensor,                 # [B]
        item_idx: torch.Tensor,                 # [B]
        E_batch: Optional[torch.Tensor] = None, # [B,k1] optional if cached
        F_batch: Optional[torch.Tensor] = None, # [B,k1] optional if cached
        return_reps: bool = True,
    ) -> TCFOutput:
        """
        Returns:
          scores = sigmoid(logits)
        """
        U = self.user_emb(user_idx)  # [B,k3]
        V = self.item_emb(item_idx)  # [B,k3]

        E, Ftxt = self._get_text_batch(user_idx, item_idx, E_batch, F_batch)  # [B,k1]
        # Make sure on same device/dtype
        E = E.to(U.device, dtype=U.dtype)
        Ftxt = Ftxt.to(V.device, dtype=V.dtype)

        user_rep = torch.cat([U, E], dim=-1)      # [B,k3+k1]  [U,E] :contentReference[oaicite:3]{index=3}
        item_rep = torch.cat([V, Ftxt], dim=-1)   # [B,k3+k1]  [V,F] :contentReference[oaicite:4]{index=4}

        if self.interaction == "mlp":
            x = torch.cat([user_rep, item_rep], dim=-1)  # [B,2*(k3+k1)]
            logits = self.f(x).squeeze(-1)               # [B]
        else:  # dot
            pu = self.user_proj(user_rep)
            qi = self.item_proj(item_rep)
            logits = (pu * qi).sum(dim=-1)               # [B]

        scores = torch.sigmoid(logits)

        return TCFOutput(
            scores=scores,
            user_rep=user_rep if return_reps else user_rep.detach(),
            item_rep=item_rep if return_reps else item_rep.detach(),
            logits=logits,
        )

    def reg_embeddings(self) -> torch.Tensor:
        """
        Paper's reg term is Frobenius norm of U and V. :contentReference[oaicite:5]{index=5}
        """
        return (self.user_emb.weight ** 2).sum() + (self.item_emb.weight ** 2).sum()

    def reg_all(self) -> torch.Tensor:
        """
        Optional: include interaction parameters in regularization.
        Default training can still use reg_embeddings() only.
        """
        reg = self.reg_embeddings()
        if self.reg_only_embeddings:
            return reg

        if self.interaction == "mlp":
            for m in self.f.modules():
                if isinstance(m, nn.Linear):
                    reg = reg + (m.weight ** 2).sum()
        else:
            reg = reg + (self.user_proj.weight ** 2).sum() + (self.item_proj.weight ** 2).sum()
        return reg
