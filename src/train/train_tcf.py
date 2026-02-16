# src/train/train_tcf.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import (
    DomainData,
    NegativeSamplingDataset,
    collate_neg_sampling,
)
from src.models.tmn import TMN
from src.models.tcf import TCF
from src.utils.seeds import set_global_seed, seed_worker


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _get_device(cfg: Dict[str, Any]) -> torch.device:
    dev = str(cfg.get("device", "cpu")).lower()
    if dev.startswith("cuda") and torch.cuda.is_available():
        return torch.device(dev)
    return torch.device("cpu")


def _pad_docs_all(docs: list[list[int]], pad_id: int = 0, max_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    docs: List[List[int]] length N
    return:
      ids:  LongTensor [N, L]
      mask: BoolTensor [N, L]
    """
    if max_len is None:
        max_len = max((len(x) for x in docs), default=0)
    N = len(docs)
    ids = torch.full((N, max_len), fill_value=pad_id, dtype=torch.long)
    mask = torch.zeros((N, max_len), dtype=torch.bool)
    for n, seq in enumerate(docs):
        L = min(len(seq), max_len)
        if L > 0:
            ids[n, :L] = torch.tensor(seq[:L], dtype=torch.long)
            mask[n, :L] = True
    return ids, mask


@torch.no_grad()
def _precompute_text_features_from_tmn(
    tmn: TMN,
    data: DomainData,
    device: torch.device,
    max_user_len: Optional[int],
    max_item_len: Optional[int],
    batch_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      E_all: [num_users, k1]
      F_all: [num_items, k1]
    """
    pad_id = int(data.word2id.get("<PAD>", 0))
    user_docs, user_mask = _pad_docs_all(data.user_docs, pad_id=pad_id, max_len=max_user_len)
    item_docs, item_mask = _pad_docs_all(data.item_docs, pad_id=pad_id, max_len=max_item_len)

    feats = tmn.export_all_text_features(
        user_docs=user_docs,
        user_mask=user_mask,
        item_docs=item_docs,
        item_mask=item_mask,
        batch_size=batch_size,
        device=device,
    )
    E_all = feats["E"]  # CPU tensor
    F_all = feats["F"]  # CPU tensor
    return E_all, F_all


@torch.no_grad()
def _evaluate_tcf_bce(
    model: TCF,
    loader: DataLoader,
    device: torch.device,
    bce: nn.BCELoss,
) -> float:
    model.eval()
    total_loss = 0.0
    total_n = 0

    for u, items, labels in loader:
        u = u.to(device)                      # [B]
        items = items.to(device)              # [B, 1+neg]
        labels = labels.to(device)            # [B, 1+neg]

        B, M = items.shape

        user_idx = u.unsqueeze(1).expand(B, M).reshape(-1)     # [B*M]
        item_idx = items.reshape(-1)                           # [B*M]
        y = labels.reshape(-1)                                 # [B*M]

        out = model(user_idx=user_idx, item_idx=item_idx)
        loss = bce(out.scores, y)

        total_loss += float(loss.item()) * y.numel()
        total_n += y.numel()

    return total_loss / max(total_n, 1)


def run_train_tcf(cfg: Dict[str, Any], domain: str):
    """
    Train TCF for one domain and save best checkpoint:
      checkpoints/tcf/<Domain>.pt

    Dependencies:
      - processed data exists at data/processed/<Domain>/
      - TMN checkpoint exists at checkpoints/tmn/<Domain>.pt
    """
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed, deterministic=True)

    device = _get_device(cfg)

    # ---- load processed data ----
    processed_root = cfg["processed_root"]
    data = DomainData.load(processed_root, domain)

    pad_id = int(data.word2id.get("<PAD>", 0))

    # ---- load TMN checkpoint ----
    ckpt_root = Path(cfg["checkpoint_root"])  # you create checkpoints/ top-level manually
    tmn_ckpt_path = ckpt_root / "tmn" / f"{domain}.pt"
    if not tmn_ckpt_path.exists():
        raise FileNotFoundError(f"TMN checkpoint not found: {tmn_ckpt_path}")

    tmn_state = torch.load(tmn_ckpt_path, map_location="cpu")
    k1 = int(tmn_state["k1_word_dim"])
    k2 = int(tmn_state["k2_latent_dim"])
    freeze_semantic = bool(tmn_state.get("freeze_semantic", False))
    tmn_cfg = tmn_state.get("cfg_tmn", {})

    # 对齐预计算截断长度：优先用 TCF cfg，其次 TMN cfg
    tcf_cfg = cfg.get("tcf", {})
    max_user_len = tcf_cfg.get("max_user_len", tmn_cfg.get("max_user_len", None))
    max_item_len = tcf_cfg.get("max_item_len", tmn_cfg.get("max_item_len", None))
    if max_user_len is not None:
        max_user_len = int(max_user_len)
    if max_item_len is not None:
        max_item_len = int(max_item_len)

    # ---- build TMN model to export E/F ----
    tmn = TMN(
        num_users=data.num_users,
        num_items=data.num_items,
        vocab_size=data.vocab_size,
        k1_word_dim=k1,
        k2_latent_dim=k2,
        pad_id=pad_id,
        freeze_semantic=freeze_semantic,
        semantic_init=None,  # 若你后续接 word2vec，可在这里加载并传入
    )
    tmn.load_state_dict(tmn_state["state_dict"], strict=True)
    tmn.to(device)
    tmn.eval()

    # ---- precompute textual features (fixed) ----
    export_bs = int(tcf_cfg.get("export_batch_size", 4096))
    E_all, F_all = _precompute_text_features_from_tmn(
        tmn=tmn,
        data=data,
        device=device,
        max_user_len=max_user_len,
        max_item_len=max_item_len,
        batch_size=export_bs,
    )
    # keep on CPU; TCF.forward will move the selected rows to GPU
    # (and keeps memory low)

    # ---- build TCF model ----
    k3 = int(tcf_cfg.get("k3_embed_dim", 64))
    lr = float(tcf_cfg.get("lr", 1e-3))
    reg = float(tcf_cfg.get("reg", 1e-2))
    epochs = int(tcf_cfg.get("epochs", 10))
    batch_size = int(tcf_cfg.get("batch_size", 512))
    num_workers = int(tcf_cfg.get("num_workers", 0))

    neg_ratio = int(tcf_cfg.get("neg_ratio", cfg.get("tdar", {}).get("source_neg_ratio", 4)))

    interaction = str(tcf_cfg.get("interaction", "mlp")).lower()
    mlp_hidden = tcf_cfg.get("mlp_hidden", [256, 128])
    dropout = float(tcf_cfg.get("dropout", 0.0))
    reg_only_embeddings = bool(tcf_cfg.get("reg_only_embeddings", True))

    model = TCF(
        num_users=data.num_users,
        num_items=data.num_items,
        k3_embed_dim=k3,
        k1_text_dim=k1,
        interaction=interaction,
        mlp_hidden=mlp_hidden,
        dropout=dropout,
        reg_only_embeddings=reg_only_embeddings,
    ).to(device)

    model.set_text_features(E_all, F_all)  # cache fixed E/F

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()

    # ---- datasets: negative sampling for train/val ----
    # negatives should avoid seen positives; for training, avoid train positives;
    # for validation, avoid all positives (train+val+test) to be safer.
    u2pos_train = data.build_user_pos("train")
    u2pos_all = data.build_all_pos()

    train_ds = NegativeSamplingDataset(
        positives=data.train,
        num_items=data.num_items,
        user_pos_dict=u2pos_train,
        neg_ratio=neg_ratio,
        seed=seed,
        with_text=False,
    )
    val_ds = NegativeSamplingDataset(
        positives=data.val,
        num_items=data.num_items,
        user_pos_dict=u2pos_all,
        neg_ratio=neg_ratio,
        seed=seed + 1,
        with_text=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        collate_fn=collate_neg_sampling,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        collate_fn=collate_neg_sampling,
        drop_last=False,
    )

    # ---- checkpoint path ----
    tcf_dir = ckpt_root / "tcf"
    _ensure_dir(tcf_dir)  # only create subdir
    ckpt_path = tcf_dir / f"{domain}.pt"

    # ---- train loop ----
    best_val = float("inf")
    best_state: Optional[Dict[str, Any]] = None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        for u, items, labels in train_loader:
            u = u.to(device)          # [B]
            items = items.to(device)  # [B, M]
            labels = labels.to(device)

            B, M = items.shape
            user_idx = u.unsqueeze(1).expand(B, M).reshape(-1)  # [B*M]
            item_idx = items.reshape(-1)                        # [B*M]
            y = labels.reshape(-1)                              # [B*M]

            out = model(user_idx=user_idx, item_idx=item_idx)
            loss_pred = bce(out.scores, y)
            loss_reg = reg * model.reg_embeddings()
            loss = loss_pred + loss_reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss_pred.item()) * y.numel()
            total_n += y.numel()

        train_bce = total_loss / max(total_n, 1)
        val_bce = _evaluate_tcf_bce(model, val_loader, device, bce)

        print(f"[TCF][{domain}] epoch {ep}/{epochs}  train_bce={train_bce:.6f}  val_bce={val_bce:.6f}")

        if val_bce < best_val:
            best_val = val_bce
            # 是否把 E_all/F_all 存进 checkpoint（可选，默认 False，避免文件过大）
            store_text = bool(tcf_cfg.get("store_text_features", False))

            best_state = {
                "domain": domain,
                "seed": seed,
                "processed_root": str(processed_root),
                "tmn_checkpoint": str(tmn_ckpt_path),
                "num_users": data.num_users,
                "num_items": data.num_items,
                "k1_text_dim": k1,
                "k3_embed_dim": k3,
                "interaction": interaction,
                "mlp_hidden": mlp_hidden,
                "dropout": dropout,
                "neg_ratio": neg_ratio,
                "best_val_bce": best_val,
                "state_dict": model.state_dict(),
                "cfg_tcf": tcf_cfg,
            }
            if store_text:
                best_state["E_all"] = E_all  # CPU tensor
                best_state["F_all"] = F_all

    if best_state is None:
        raise RuntimeError("TCF training did not produce any checkpoint (unexpected).")

    torch.save(best_state, ckpt_path)
    print(f"[TCF][{domain}] saved best checkpoint: {ckpt_path} (best_val_bce={best_val:.6f})")

    summary_path = tcf_dir / f"{domain}.summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "domain": domain,
                "best_val_bce": best_val,
                "checkpoint": str(ckpt_path),
                "seed": seed,
                "k1_text_dim": k1,
                "k3_embed_dim": k3,
                "interaction": interaction,
                "neg_ratio": neg_ratio,
                "max_user_len": max_user_len,
                "max_item_len": max_item_len,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return ckpt_path


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--domain", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    run_train_tcf(cfg, domain=args.domain)
