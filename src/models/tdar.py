# src/models/tdar.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn

from src.models.tcf import TCF, TCFOutput
from src.models.domain_classifier import DomainClassifier, DomainClassifierOutput


@dataclass
class TDARForwardOutput:
    # prediction outputs
    source: Optional[TCFOutput]                 # source prediction results (flattened)
    target: Optional[TCFOutput]                 # target prediction results (flattened)

    # domain classifier logits (user/item)
    dom_user_logits: Optional[torch.Tensor]     # [Ns+Nt, 2]
    dom_item_logits: Optional[torch.Tensor]     # [Ns+Nt, 2]
    dom_user_labels: Optional[torch.Tensor]     # [Ns+Nt]
    dom_item_labels: Optional[torch.Tensor]     # [Ns+Nt]

    # for debugging / analysis
    meta: Dict[str, Any]


class TDAR(nn.Module):
    """
    TDAR core model:
      - source TCF with its own CF embeddings (U^s, V^s) + fixed text features (E^s, F^s)
      - target TCF with its own CF embeddings (U^t, V^t) + fixed text features (E^t, F^t)
      - adversarial alignment via domain classifiers on representations:
          user_rep = [U, E], item_rep = [V, F]
        GRL inside DomainClassifier reverses gradients to embeddings/features.

    Notes:
      - Text features E/F are assumed fixed from TMN and should be cached via set_text_features().
      - Domain labels convention: source=0, target=1.
    """

    def __init__(
        self,
        # sizes
        num_users_s: int,
        num_items_s: int,
        num_users_t: int,
        num_items_t: int,
        # dims
        k1_text_dim: int,
        k3_embed_dim: int,
        # TCF config
        interaction: str = "mlp",
        mlp_hidden: Optional[List[int]] = None,
        dropout: float = 0.0,
        reg_only_embeddings: bool = True,
        # domain classifier config
        cls_hidden: Optional[List[int]] = None,
        cls_dropout: float = 0.0,
        grl_lambda: float = 1.0,
        num_domains: int = 2,
    ):
        super().__init__()

        self.k1 = int(k1_text_dim)
        self.k3 = int(k3_embed_dim)
        self.rep_dim = self.k1 + self.k3
        self.num_domains = int(num_domains)

        # Two domain-specific TCF models
        self.tcf_s = TCF(
            num_users=num_users_s,
            num_items=num_items_s,
            k3_embed_dim=self.k3,
            k1_text_dim=self.k1,
            interaction=interaction,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
            reg_only_embeddings=reg_only_embeddings,
        )
        self.tcf_t = TCF(
            num_users=num_users_t,
            num_items=num_items_t,
            k3_embed_dim=self.k3,
            k1_text_dim=self.k1,
            interaction=interaction,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
            reg_only_embeddings=reg_only_embeddings,
        )

        # Domain classifiers (user/item), shared across domains
        self.cls_user = DomainClassifier(
            input_dim=self.rep_dim,
            num_domains=self.num_domains,
            hidden=cls_hidden,
            dropout=cls_dropout,
            use_grl=True,
            grl_lambda=grl_lambda,
        )
        self.cls_item = DomainClassifier(
            input_dim=self.rep_dim,
            num_domains=self.num_domains,
            hidden=cls_hidden,
            dropout=cls_dropout,
            use_grl=True,
            grl_lambda=grl_lambda,
        )

    # -------------------------
    # utilities
    # -------------------------

    @torch.no_grad()
    def set_text_features_source(self, E_all_s: torch.Tensor, F_all_s: torch.Tensor):
        """
        Cache fixed textual features for source TCF.
        E_all_s: [num_users_s, k1]
        F_all_s: [num_items_s, k1]
        """
        self.tcf_s.set_text_features(E_all_s, F_all_s)

    @torch.no_grad()
    def set_text_features_target(self, E_all_t: torch.Tensor, F_all_t: torch.Tensor):
        """
        Cache fixed textual features for target TCF.
        """
        self.tcf_t.set_text_features(E_all_t, F_all_t)

    def set_grl_lambda(self, lambd: float):
        """
        Set GRL strength for both user/item domain classifiers.
        """
        self.cls_user.set_lambda(lambd)
        self.cls_item.set_lambda(lambd)

    def reg_source(self) -> torch.Tensor:
        return self.tcf_s.reg_embeddings()

    def reg_target(self) -> torch.Tensor:
        return self.tcf_t.reg_embeddings()

    # -------------------------
    # forward helpers
    # -------------------------

    def _forward_source_flat(
        self,
        user_idx_s: torch.Tensor,  # [Ns]
        item_idx_s: torch.Tensor,  # [Ns]
    ) -> TCFOutput:
        return self.tcf_s(user_idx=user_idx_s, item_idx=item_idx_s, return_reps=True)

    def _forward_target_flat(
        self,
        user_idx_t: torch.Tensor,  # [Nt]
        item_idx_t: torch.Tensor,  # [Nt]
    ) -> TCFOutput:
        return self.tcf_t(user_idx=user_idx_t, item_idx=item_idx_t, return_reps=True)

    def _domain_logits(
        self,
        user_rep_s: Optional[torch.Tensor],  # [Ns, rep_dim]
        user_rep_t: Optional[torch.Tensor],  # [Nt, rep_dim]
        item_rep_s: Optional[torch.Tensor],  # [Ns, rep_dim]
        item_rep_t: Optional[torch.Tensor],  # [Nt, rep_dim]
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Build domain classifier logits for users/items.
        Returns (dom_user_logits, dom_item_logits, dom_user_labels, dom_item_labels)
        """
        if user_rep_s is None or user_rep_t is None or item_rep_s is None or item_rep_t is None:
            return None, None, None, None

        # concat
        hu = torch.cat([user_rep_s, user_rep_t], dim=0)  # [Ns+Nt, rep_dim]
        hi = torch.cat([item_rep_s, item_rep_t], dim=0)  # [Ns+Nt, rep_dim]

        Ns = user_rep_s.size(0)
        Nt = user_rep_t.size(0)

        # domain labels: source=0, target=1
        y_u = torch.cat([
            torch.zeros(Ns, dtype=torch.long, device=device),
            torch.ones(Nt, dtype=torch.long, device=device),
        ], dim=0)
        y_i = torch.cat([
            torch.zeros(Ns, dtype=torch.long, device=device),
            torch.ones(Nt, dtype=torch.long, device=device),
        ], dim=0)

        dom_user_logits = self.cls_user(hu, return_output=False)  # [Ns+Nt, 2]
        dom_item_logits = self.cls_item(hi, return_output=False)  # [Ns+Nt, 2]

        return dom_user_logits, dom_item_logits, y_u, y_i

    # -------------------------
    # main forward
    # -------------------------

    def forward(
        self,
        # source batch (flattened)
        user_idx_s: Optional[torch.Tensor] = None,  # [Ns]
        item_idx_s: Optional[torch.Tensor] = None,  # [Ns]
        # target batch (flattened)
        user_idx_t: Optional[torch.Tensor] = None,  # [Nt]
        item_idx_t: Optional[torch.Tensor] = None,  # [Nt]
        # whether compute domain loss terms
        compute_domain: bool = True,
    ) -> TDARForwardOutput:
        """
        You can pass:
          - source only
          - target only
          - both (typical TDAR step)
        Domain logits are produced only when both source and target are provided.

        Returns TDARForwardOutput including:
          - source / target prediction outputs (scores, reps)
          - dom_user_logits / dom_item_logits and their labels
        """
        device = next(self.parameters()).device

        out_s: Optional[TCFOutput] = None
        out_t: Optional[TCFOutput] = None

        if user_idx_s is not None and item_idx_s is not None:
            out_s = self._forward_source_flat(user_idx_s.to(device), item_idx_s.to(device))

        if user_idx_t is not None and item_idx_t is not None:
            out_t = self._forward_target_flat(user_idx_t.to(device), item_idx_t.to(device))

        dom_user_logits = dom_item_logits = dom_user_labels = dom_item_labels = None

        if compute_domain and (out_s is not None) and (out_t is not None):
            dom_user_logits, dom_item_logits, dom_user_labels, dom_item_labels = self._domain_logits(
                user_rep_s=out_s.user_rep,
                user_rep_t=out_t.user_rep,
                item_rep_s=out_s.item_rep,
                item_rep_t=out_t.item_rep,
                device=device,
            )

        meta = {
            "Ns": int(out_s.scores.numel()) if out_s is not None else 0,
            "Nt": int(out_t.scores.numel()) if out_t is not None else 0,
            "rep_dim": self.rep_dim,
        }

        return TDARForwardOutput(
            source=out_s,
            target=out_t,
            dom_user_logits=dom_user_logits,
            dom_item_logits=dom_item_logits,
            dom_user_labels=dom_user_labels,
            dom_item_labels=dom_item_labels,
            meta=meta,
        )