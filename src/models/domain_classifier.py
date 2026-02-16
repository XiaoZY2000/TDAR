# src/models/domain_classifier.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Gradient Reversal Layer
# =========================

class _GRLFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float):
        ctx.lambd = float(lambd)
        # forward: identity
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # backward: reverse gradient
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    """
    Gradient Reversal Layer (GRL):
      y = x
      dL/dx = -lambda * dL/dy
    """

    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = float(lambd)

    def set_lambda(self, lambd: float):
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GRLFn.apply(x, self.lambd)


# =========================
# MLP builder
# =========================

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


# =========================
# Domain Classifier
# =========================

@dataclass
class DomainClassifierOutput:
    logits: torch.Tensor   # [B, num_domains]
    probs: torch.Tensor    # [B, num_domains]
    pred: torch.Tensor     # [B] predicted domain id


class DomainClassifier(nn.Module):
    """
    Domain classifier g(., Φ) used in TDAR:
      - Input: representation h (e.g., [U,E] or [V,F])
      - GRL: apply GradientReversal to enforce domain-invariant embeddings
      - Output: domain logits (cross-entropy)

    Typical usage in TDAR:
      logits = cls(grl(h))
      loss_cls = CE(logits, domain_label)
    """

    def __init__(
        self,
        input_dim: int,
        num_domains: int = 2,           # source vs target
        hidden: Optional[List[int]] = None,
        dropout: float = 0.0,
        use_grl: bool = True,
        grl_lambda: float = 1.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_domains = int(num_domains)

        if hidden is None:
            hidden = [256, 128]

        dims = [self.input_dim] + list(map(int, hidden)) + [self.num_domains]
        self.net = build_mlp(dims, dropout=dropout)

        self.use_grl = bool(use_grl)
        self.grl = GradientReversal(grl_lambda) if self.use_grl else None

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_lambda(self, lambd: float):
        """
        Set GRL lambda dynamically during training.
        """
        if self.grl is not None:
            self.grl.set_lambda(lambd)

    def forward(self, h: torch.Tensor, return_output: bool = False):
        """
        h: [B, input_dim]
        return:
          if return_output=False: logits [B, num_domains]
          else: DomainClassifierOutput
        """
        if self.grl is not None:
            h = self.grl(h)

        logits = self.net(h)  # [B, num_domains]
        if not return_output:
            return logits

        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1)
        return DomainClassifierOutput(logits=logits, probs=probs, pred=pred)