# src/models/tmn.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """
    logits: [..., L]
    mask:   [..., L] bool, True means valid token
    return: same shape, softmax over valid positions only; invalid positions -> 0
    """
    # set invalid positions to a very negative number
    neg_inf = torch.finfo(logits.dtype).min
    logits_masked = logits.masked_fill(~mask, neg_inf)

    # if a row is all invalid, softmax would be NaN; handle by replacing with zeros later
    attn = F.softmax(logits_masked, dim=dim)

    # zero out invalids
    attn = attn * mask.to(attn.dtype)

    # renormalize for safety (in case of all-masked rows)
    denom = attn.sum(dim=dim, keepdim=True).clamp_min(eps)
    attn = attn / denom
    return attn


@dataclass
class TMNOutput:
    scores: torch.Tensor              # [B]  sigmoid(E_u · F_i)
    user_text: torch.Tensor           # [B, K1]  E_u
    item_text: torch.Tensor           # [B, K1]  F_i
    user_attn: Optional[torch.Tensor] # [B, Lu]  a_uw
    item_attn: Optional[torch.Tensor] # [B, Li]  a_iv


class TMN(nn.Module):
    """
    Text Memory Network (TMN) per TDAR paper.

    We have:
      - S: word semantic embedding matrix (pretrained, fixed)  shape [V, K1]
      - T: word latent embedding matrix (trainable)           shape [V, K2]
      - P: user latent embedding matrix (trainable)           shape [num_users, K2]
      - Q: item latent embedding matrix (trainable)           shape [num_items, K2]

    For a user u with word ids w in R_u:
      e_uw = <P_u, T_w>                 (dot product)
      a_uw = softmax_{w in R_u}(e_uw)
      E_u  = sum_{w in R_u} a_uw * S_w  (word semantic weighted sum)

    Similarly for item i:
      e_iv, a_iv, F_i

    Predict (text-only model):
      R_hat_ui = sigmoid( E_u · F_i )

    Notes:
      - Input expects padded word ids + masks (from dataset.collate_pad_text).
      - Handles empty docs: if mask all False, produces zero text vector.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        vocab_size: int,
        k1_word_dim: int = 300,
        k2_latent_dim: int = 64,
        pad_id: int = 0,
        freeze_semantic: bool = True,
        semantic_init: Optional[torch.Tensor] = None,  # [V, K1] pretrained word2vec vectors if provided
    ):
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.vocab_size = int(vocab_size)
        self.k1 = int(k1_word_dim)
        self.k2 = int(k2_latent_dim)
        self.pad_id = int(pad_id)

        # Word semantic matrix S (pretrained word2vec) — fixed by default
        self.word_semantic = nn.Embedding(self.vocab_size, self.k1, padding_idx=self.pad_id)
        if semantic_init is not None:
            if semantic_init.shape != (self.vocab_size, self.k1):
                raise ValueError(f"semantic_init shape must be {(self.vocab_size, self.k1)}, got {tuple(semantic_init.shape)}")
            with torch.no_grad():
                self.word_semantic.weight.copy_(semantic_init)
        if freeze_semantic:
            self.word_semantic.weight.requires_grad_(False)

        # Word latent matrix T (trainable)
        self.word_latent = nn.Embedding(self.vocab_size, self.k2, padding_idx=self.pad_id)

        # User / Item latent matrices P, Q (trainable)
        self.user_latent = nn.Embedding(self.num_users, self.k2)
        self.item_latent = nn.Embedding(self.num_items, self.k2)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize trainable embeddings
        nn.init.normal_(self.word_latent.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.user_latent.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.item_latent.weight, mean=0.0, std=0.02)

        # padding rows to zero
        with torch.no_grad():
            if self.word_latent.padding_idx is not None:
                self.word_latent.weight[self.word_latent.padding_idx].fill_(0.0)
            if self.word_semantic.padding_idx is not None:
                self.word_semantic.weight[self.word_semantic.padding_idx].fill_(0.0)

    def encode_user_text(
        self,
        user_idx: torch.Tensor,         # [B]
        user_word_ids: torch.Tensor,    # [B, Lu]
        user_mask: torch.Tensor,        # [B, Lu] bool
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
          E_u: [B, K1]
          user_attn: [B, Lu] or None
        """
        B, Lu = user_word_ids.shape

        P_u = self.user_latent(user_idx)                     # [B, K2]
        T_w = self.word_latent(user_word_ids)                # [B, Lu, K2]

        # e_uw = dot(P_u, T_w)
        # einsum: (B,K2) x (B,Lu,K2) -> (B,Lu)
        e = torch.einsum("bk,blk->bl", P_u, T_w)             # [B, Lu]

        a = masked_softmax(e, user_mask, dim=-1)             # [B, Lu]

        S_w = self.word_semantic(user_word_ids)              # [B, Lu, K1]
        # E_u = sum a_uw * S_w
        E_u = torch.einsum("bl,blk->bk", a, S_w)             # [B, K1]

        # handle empty docs: if mask sum == 0, force E_u = 0
        empty = (user_mask.sum(dim=-1) == 0)                 # [B]
        if empty.any():
            E_u = E_u.masked_fill(empty.unsqueeze(-1), 0.0)
            a = a.masked_fill(empty.unsqueeze(-1), 0.0)

        return E_u, (a if return_attn else None)

    def encode_item_text(
        self,
        item_idx: torch.Tensor,         # [B]
        item_word_ids: torch.Tensor,    # [B, Li]
        item_mask: torch.Tensor,        # [B, Li] bool
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
          F_i: [B, K1]
          item_attn: [B, Li] or None
        """
        B, Li = item_word_ids.shape

        Q_i = self.item_latent(item_idx)                     # [B, K2]
        T_v = self.word_latent(item_word_ids)                # [B, Li, K2]

        e = torch.einsum("bk,blk->bl", Q_i, T_v)             # [B, Li]
        a = masked_softmax(e, item_mask, dim=-1)             # [B, Li]

        S_v = self.word_semantic(item_word_ids)              # [B, Li, K1]
        F_i = torch.einsum("bl,blk->bk", a, S_v)             # [B, K1]

        empty = (item_mask.sum(dim=-1) == 0)
        if empty.any():
            F_i = F_i.masked_fill(empty.unsqueeze(-1), 0.0)
            a = a.masked_fill(empty.unsqueeze(-1), 0.0)

        return F_i, (a if return_attn else None)

    def forward(
        self,
        user_idx: torch.Tensor,      # [B]
        item_idx: torch.Tensor,      # [B]
        user_word_ids: torch.Tensor, # [B, Lu]
        user_mask: torch.Tensor,     # [B, Lu] bool
        item_word_ids: torch.Tensor, # [B, Li]
        item_mask: torch.Tensor,     # [B, Li] bool
        return_attn: bool = False,
    ) -> TMNOutput:
        """
        Returns TMNOutput with text-only score:
          score = sigmoid( dot(E_u, F_i) )
        """
        E_u, a_u = self.encode_user_text(user_idx, user_word_ids, user_mask, return_attn=return_attn)
        F_i, a_i = self.encode_item_text(item_idx, item_word_ids, item_mask, return_attn=return_attn)

        logits = (E_u * F_i).sum(dim=-1)        # [B]
        scores = torch.sigmoid(logits)          # [B]

        return TMNOutput(
            scores=scores,
            user_text=E_u,
            item_text=F_i,
            user_attn=a_u,
            item_attn=a_i,
        )

    def l2_regularization(self) -> torch.Tensor:
        """
        Frobenius/L2 norm regularizer used in the paper (for P, Q, T).
        Return scalar tensor.
        """
        reg = 0.0
        reg = reg + (self.user_latent.weight ** 2).sum()
        reg = reg + (self.item_latent.weight ** 2).sum()
        reg = reg + (self.word_latent.weight ** 2).sum()
        return reg

    @torch.no_grad()
    def export_all_text_features(
        self,
        user_docs: torch.Tensor,  # [num_users, Lu] padded
        user_mask: torch.Tensor,  # [num_users, Lu]
        item_docs: torch.Tensor,  # [num_items, Li] padded
        item_mask: torch.Tensor,  # [num_items, Li]
        batch_size: int = 4096,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Utility: precompute all user/item textual features E/F for freezing in TCF/TDAR.

        Returns:
          {
            "E": [num_users, K1],
            "F": [num_items, K1],
          }
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        # users
        E_all = []
        num_users = user_docs.size(0)
        for s in range(0, num_users, batch_size):
            e = min(s + batch_size, num_users)
            u_idx = torch.arange(s, e, device=device, dtype=torch.long)
            uw = user_docs[s:e].to(device)
            um = user_mask[s:e].to(device)
            E_u, _ = self.encode_user_text(u_idx, uw, um, return_attn=False)
            E_all.append(E_u.cpu())
        E = torch.cat(E_all, dim=0)

        # items
        F_all = []
        num_items = item_docs.size(0)
        for s in range(0, num_items, batch_size):
            e = min(s + batch_size, num_items)
            i_idx = torch.arange(s, e, device=device, dtype=torch.long)
            iw = item_docs[s:e].to(device)
            im = item_mask[s:e].to(device)
            F_i, _ = self.encode_item_text(i_idx, iw, im, return_attn=False)
            F_all.append(F_i.cpu())
        F_feat = torch.cat(F_all, dim=0)

        return {"E": E, "F": F_feat}
