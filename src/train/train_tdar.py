# src/train/train_tdar.py

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
    InteractionDataset,
    collate_neg_sampling,
)
from src.models.tmn import TMN
from src.models.tdar import TDAR
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
    return feats["E"], feats["F"]


@torch.no_grad()
def _evaluate_target_bce(
    model: TDAR,
    loader: DataLoader,
    device: torch.device,
    bce: nn.BCELoss,
) -> float:
    model.eval()
    total_loss = 0.0
    total_n = 0

    for u, i, y in loader:
        u = u.to(device)
        i = i.to(device)
        y = y.to(device)

        out = model(user_idx_t=u, item_idx_t=i, compute_domain=False)
        scores = out.target.scores
        loss = bce(scores, y)

        total_loss += float(loss.item()) * y.numel()
        total_n += y.numel()

    return total_loss / max(total_n, 1)


def run_train_tdar(cfg: Dict[str, Any]):
    """
    Train TDAR for one source->target experiment and save best checkpoint:
      checkpoints/tdar/<Source>_to_<Target>.pt
    """
    # ---- seeds & device ----
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed, deterministic=True)
    device = _get_device(cfg)

    # ---- domains ----
    source_domain = cfg["source_domain"]
    target_domain = cfg["target_domain"]

    # ---- load processed data ----
    processed_root = cfg["processed_root"]
    src = DomainData.load(processed_root, source_domain)
    tgt = DomainData.load(processed_root, target_domain)

    pad_id_s = int(src.word2id.get("<PAD>", 0))
    pad_id_t = int(tgt.word2id.get("<PAD>", 0))

    # ---- load TMN checkpoints (both domains) ----
    ckpt_root = Path(cfg["checkpoint_root"])  # you create top-level
    tmn_ckpt_s = ckpt_root / "tmn" / f"{source_domain}.pt"
    tmn_ckpt_t = ckpt_root / "tmn" / f"{target_domain}.pt"
    if not tmn_ckpt_s.exists():
        raise FileNotFoundError(f"TMN checkpoint not found: {tmn_ckpt_s}")
    if not tmn_ckpt_t.exists():
        raise FileNotFoundError(f"TMN checkpoint not found: {tmn_ckpt_t}")

    st_s = torch.load(tmn_ckpt_s, map_location="cpu")
    st_t = torch.load(tmn_ckpt_t, map_location="cpu")

    k1 = int(st_s["k1_word_dim"])
    k2 = int(st_s["k2_latent_dim"])
    freeze_semantic_s = bool(st_s.get("freeze_semantic", False))
    freeze_semantic_t = bool(st_t.get("freeze_semantic", False))

    # text truncation lengths: prefer tdar cfg, else tcf/tmn cfg
    tdar_cfg = cfg.get("tdar", {})
    max_user_len = tdar_cfg.get("max_user_len", None)
    max_item_len = tdar_cfg.get("max_item_len", None)
    if max_user_len is not None:
        max_user_len = int(max_user_len)
    if max_item_len is not None:
        max_item_len = int(max_item_len)

    # ---- build TMN models to export E/F ----
    tmn_s = TMN(
        num_users=src.num_users,
        num_items=src.num_items,
        vocab_size=src.vocab_size,
        k1_word_dim=k1,
        k2_latent_dim=k2,
        pad_id=pad_id_s,
        freeze_semantic=freeze_semantic_s,
        semantic_init=None,
    )
    tmn_s.load_state_dict(st_s["state_dict"], strict=True)
    tmn_s.to(device).eval()

    tmn_t = TMN(
        num_users=tgt.num_users,
        num_items=tgt.num_items,
        vocab_size=tgt.vocab_size,
        k1_word_dim=k1,
        k2_latent_dim=k2,
        pad_id=pad_id_t,
        freeze_semantic=freeze_semantic_t,
        semantic_init=None,
    )
    tmn_t.load_state_dict(st_t["state_dict"], strict=True)
    tmn_t.to(device).eval()

    export_bs = int(tdar_cfg.get("export_batch_size", 4096))
    E_s, F_s = _precompute_text_features_from_tmn(tmn_s, src, device, max_user_len, max_item_len, batch_size=export_bs)
    E_t, F_t = _precompute_text_features_from_tmn(tmn_t, tgt, device, max_user_len, max_item_len, batch_size=export_bs)

    # ---- build TDAR model ----
    k3 = int(tdar_cfg.get("k3_embed_dim", 64))
    interaction = str(tdar_cfg.get("interaction", "mlp")).lower()
    mlp_hidden = tdar_cfg.get("mlp_hidden", [256, 128])
    dropout = float(tdar_cfg.get("dropout", 0.0))
    reg_only_embeddings = bool(tdar_cfg.get("reg_only_embeddings", True))

    cls_hidden = tdar_cfg.get("cls_hidden", [256, 128])
    cls_dropout = float(tdar_cfg.get("cls_dropout", 0.0))
    grl_lambda = float(tdar_cfg.get("grl_lambda", 1.0))

    model = TDAR(
        num_users_s=src.num_users,
        num_items_s=src.num_items,
        num_users_t=tgt.num_users,
        num_items_t=tgt.num_items,
        k1_text_dim=k1,
        k3_embed_dim=k3,
        interaction=interaction,
        mlp_hidden=mlp_hidden,
        dropout=dropout,
        reg_only_embeddings=reg_only_embeddings,
        cls_hidden=cls_hidden,
        cls_dropout=cls_dropout,
        grl_lambda=grl_lambda,
        num_domains=2,
    ).to(device)

    model.set_text_features_source(E_s, F_s)
    model.set_text_features_target(E_t, F_t)

    # ---- (optional) init from TCF checkpoints ----
    if bool(tdar_cfg.get("init_from_tcf", True)):
        tcf_s_ckpt = ckpt_root / "tcf" / f"{source_domain}.pt"
        tcf_t_ckpt = ckpt_root / "tcf" / f"{target_domain}.pt"
        if tcf_s_ckpt.exists():
            st = torch.load(tcf_s_ckpt, map_location="cpu")
            model.tcf_s.load_state_dict(st["state_dict"], strict=False)
            print(f"[TDAR] initialized source TCF from {tcf_s_ckpt}")
        if tcf_t_ckpt.exists():
            st = torch.load(tcf_t_ckpt, map_location="cpu")
            model.tcf_t.load_state_dict(st["state_dict"], strict=False)
            print(f"[TDAR] initialized target TCF from {tcf_t_ckpt}")

    # ---- optimizers ----
    lr_s = float(tdar_cfg.get("lr_source", 1e-3))
    lr_t = float(tdar_cfg.get("lr_target", 1e-3))
    lr_cls = float(tdar_cfg.get("lr_cls", 1e-3))

    # separate parameter groups
    params = [
        {"params": model.tcf_s.parameters(), "lr": lr_s},
        {"params": model.tcf_t.parameters(), "lr": lr_t},
        {"params": model.cls_user.parameters(), "lr": lr_cls},
        {"params": model.cls_item.parameters(), "lr": lr_cls},
    ]
    optimizer = torch.optim.Adam(params)

    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()

    # ---- datasets & loaders ----
    neg_ratio = int(tdar_cfg.get("source_neg_ratio", 4))
    batch_size = int(tdar_cfg.get("batch_size", 512))
    num_workers = int(tdar_cfg.get("num_workers", 0))
    epochs = int(tdar_cfg.get("epochs", 10))

    # source: negative sampling (avoid train positives)
    u2pos_s_train = src.build_user_pos("train")
    src_train_ds = NegativeSamplingDataset(
        positives=src.train,
        num_items=src.num_items,
        user_pos_dict=u2pos_s_train,
        neg_ratio=neg_ratio,
        seed=seed,
        with_text=False,
    )
    src_loader = DataLoader(
        src_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        collate_fn=collate_neg_sampling,
        drop_last=False,
    )

    # target: only positives
    tgt_train_ds = InteractionDataset(tgt.train)
    tgt_val_ds = InteractionDataset(tgt.val)

    tgt_train_loader = DataLoader(
        tgt_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        collate_fn=None,
        drop_last=False,
    )
    tgt_val_loader = DataLoader(
        tgt_val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        collate_fn=None,
        drop_last=False,
    )

    # ---- training loop ----
    reg_s = float(tdar_cfg.get("reg_source", 1e-2))
    reg_t = float(tdar_cfg.get("reg_target", 1e-2))
    alpha_t = float(tdar_cfg.get("alpha_target", 0.0))  # weight for target pred loss (often 0 or small)
    beta_adv = float(tdar_cfg.get("beta_adv", 1.0))      # weight for domain losses

    ckpt_dir = ckpt_root / "tdar"
    _ensure_dir(ckpt_dir)
    ckpt_path = ckpt_dir / f"{source_domain}_to_{target_domain}.pt"

    best_val = float("inf")
    best_state: Optional[Dict[str, Any]] = None

    # zip loaders: iterate min length each epoch
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        it_tgt = iter(tgt_train_loader)

        for u_s, items_s, labels_s in src_loader:
            # get a batch from target (cycle if shorter)
            try:
                u_t, i_t, y_t = next(it_tgt)
            except StopIteration:
                it_tgt = iter(tgt_train_loader)
                u_t, i_t, y_t = next(it_tgt)

            # ---- prepare source flattened ----
            u_s = u_s.to(device)                 # [B]
            items_s = items_s.to(device)         # [B, M]
            labels_s = labels_s.to(device)       # [B, M]
            B, M = items_s.shape

            user_idx_s = u_s.unsqueeze(1).expand(B, M).reshape(-1)
            item_idx_s = items_s.reshape(-1)
            y_s = labels_s.reshape(-1)

            # ---- prepare target ----
            u_t = u_t.to(device)
            i_t = i_t.to(device)
            y_t = y_t.to(device)

            # ---- forward ----
            out = model(
                user_idx_s=user_idx_s,
                item_idx_s=item_idx_s,
                user_idx_t=u_t,
                item_idx_t=i_t,
                compute_domain=True,
            )

            # ---- losses ----
            loss_s = bce(out.source.scores, y_s)
            loss_t = bce(out.target.scores, y_t) if alpha_t != 0.0 else torch.zeros((), device=device)

            loss_du = ce(out.dom_user_logits, out.dom_user_labels)
            loss_di = ce(out.dom_item_logits, out.dom_item_labels)

            loss_reg = reg_s * model.reg_source() + reg_t * model.reg_target()

            loss = loss_s + alpha_t * loss_t + beta_adv * (loss_du + loss_di) + loss_reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss_s.item()) * y_s.numel()
            total_n += y_s.numel()

        train_bce_s = total_loss / max(total_n, 1)
        val_bce_t = _evaluate_target_bce(model, tgt_val_loader, device, bce)

        print(f"[TDAR][{source_domain}->{target_domain}] epoch {ep}/{epochs}  train_src_bce={train_bce_s:.6f}  val_tgt_bce={val_bce_t:.6f}")

        if val_bce_t < best_val:
            best_val = val_bce_t
            best_state = {
                "source_domain": source_domain,
                "target_domain": target_domain,
                "seed": seed,
                "processed_root": str(processed_root),
                "tmn_ckpt_s": str(tmn_ckpt_s),
                "tmn_ckpt_t": str(tmn_ckpt_t),
                "num_users_s": src.num_users,
                "num_items_s": src.num_items,
                "num_users_t": tgt.num_users,
                "num_items_t": tgt.num_items,
                "k1_text_dim": k1,
                "k3_embed_dim": k3,
                "interaction": interaction,
                "best_val_tgt_bce": best_val,
                "state_dict": model.state_dict(),
                "cfg_tdar": tdar_cfg,
            }

    if best_state is None:
        raise RuntimeError("TDAR training did not produce any checkpoint (unexpected).")

    torch.save(best_state, ckpt_path)
    print(f"[TDAR] saved best checkpoint: {ckpt_path} (best_val_tgt_bce={best_val:.6f})")

    summary_path = ckpt_dir / f"{source_domain}_to_{target_domain}.summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_domain": source_domain,
                "target_domain": target_domain,
                "best_val_tgt_bce": best_val,
                "checkpoint": str(ckpt_path),
                "seed": seed,
                "k1_text_dim": k1,
                "k3_embed_dim": k3,
                "neg_ratio": neg_ratio,
                "alpha_target": alpha_t,
                "beta_adv": beta_adv,
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
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    run_train_tdar(cfg)
