# src/train/train_tmn.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import DomainData, InteractionWithTextDataset, collate_pad_text
from src.models.tmn import TMN
from src.utils.seeds import set_global_seed, seed_worker


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _get_device(cfg: Dict[str, Any]) -> torch.device:
    dev = str(cfg.get("device", "cpu")).lower()
    if dev.startswith("cuda") and torch.cuda.is_available():
        return torch.device(dev)
    return torch.device("cpu")


@torch.no_grad()
def _evaluate_bce(
    model: TMN,
    loader: DataLoader,
    device: torch.device,
    bce: nn.BCELoss,
) -> float:
    model.eval()
    total_loss = 0.0
    total_n = 0

    for u, i, y, u_words, u_mask, i_words, i_mask in loader:
        u = u.to(device)
        i = i.to(device)
        y = y.to(device)

        u_words = u_words.to(device)
        u_mask = u_mask.to(device)
        i_words = i_words.to(device)
        i_mask = i_mask.to(device)

        out = model(
            user_idx=u,
            item_idx=i,
            user_word_ids=u_words,
            user_mask=u_mask,
            item_word_ids=i_words,
            item_mask=i_mask,
            return_attn=False,
        )
        loss = bce(out.scores, y)
        bs = y.numel()
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(total_n, 1)


def run_train_tmn(cfg: Dict[str, Any], domain: str):
    """
    Train TMN for one domain and save best checkpoint:
      checkpoints/tmn/<Domain>.pt

    Required cfg fields (see your yaml):
      processed_root, checkpoint_root, seed, device, tmn{...}
    """
    # ---- seeds ----
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed, deterministic=True)

    # ---- device ----
    device = _get_device(cfg)

    # ---- load processed data ----
    processed_root = cfg["processed_root"]
    data = DomainData.load(processed_root, domain)

    tmn_cfg = cfg.get("tmn", {})
    k1 = int(tmn_cfg.get("k1_word2vec_dim", 300))
    k2 = int(tmn_cfg.get("k2_latent_dim", 64))
    lr = float(tmn_cfg.get("lr", 1e-3))
    reg = float(tmn_cfg.get("reg", 1e-2))
    epochs = int(tmn_cfg.get("epochs", 10))
    batch_size = int(tmn_cfg.get("batch_size", 256))
    num_workers = int(tmn_cfg.get("num_workers", 0))

    # 文本截断（很重要，Amazon 用户/物品拼起来会很长）
    max_user_len = tmn_cfg.get("max_user_len", None)
    max_item_len = tmn_cfg.get("max_item_len", None)
    if max_user_len is not None:
        max_user_len = int(max_user_len)
    if max_item_len is not None:
        max_item_len = int(max_item_len)

    pad_id = int(data.word2id.get("<PAD>", 0))

    # ---- build datasets ----
    train_ds = InteractionWithTextDataset(data.train, data.user_docs, data.item_docs)
    val_ds = InteractionWithTextDataset(data.val, data.user_docs, data.item_docs)

    collate_fn = lambda batch: collate_pad_text(
        batch,
        pad_id=pad_id,
        max_user_len=max_user_len,
        max_item_len=max_item_len,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # ---- model ----
    # 若你后续要接 word2vec：可以在 cfg 里加 semantic_init_path，然后这里读入 semantic_init
    semantic_init = None
    freeze_semantic = bool(tmn_cfg.get("freeze_semantic", True))
    # 如果你没提供 pretrained vectors，但 freeze_semantic=True，会导致 S 随机且不训练；
    # 所以我们做个更合理的默认：若 semantic_init=None，则默认不冻结（除非你显式指定）
    if semantic_init is None and "freeze_semantic" not in tmn_cfg:
        freeze_semantic = False

    model = TMN(
        num_users=data.num_users,
        num_items=data.num_items,
        vocab_size=data.vocab_size,
        k1_word_dim=k1,
        k2_latent_dim=k2,
        pad_id=pad_id,
        freeze_semantic=freeze_semantic,
        semantic_init=semantic_init,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()

    # ---- checkpoint path ----
    ckpt_root = Path(cfg["checkpoint_root"])  # 你手动建 checkpoints/ 顶层
    tmn_dir = ckpt_root / "tmn"
    _ensure_dir(tmn_dir)  # 只创建子目录
    ckpt_path = tmn_dir / f"{domain}.pt"

    # ---- train loop ----
    best_val = float("inf")
    best_state: Optional[Dict[str, Any]] = None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        for u, i, y, u_words, u_mask, i_words, i_mask in train_loader:
            u = u.to(device)
            i = i.to(device)
            y = y.to(device)

            u_words = u_words.to(device)
            u_mask = u_mask.to(device)
            i_words = i_words.to(device)
            i_mask = i_mask.to(device)

            out = model(
                user_idx=u,
                item_idx=i,
                user_word_ids=u_words,
                user_mask=u_mask,
                item_word_ids=i_words,
                item_mask=i_mask,
                return_attn=False,
            )

            loss_pred = bce(out.scores, y)
            loss_reg = reg * model.l2_regularization()
            loss = loss_pred + loss_reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = y.numel()
            total_loss += float(loss_pred.item()) * bs
            total_n += bs

        train_bce = total_loss / max(total_n, 1)
        val_bce = _evaluate_bce(model, val_loader, device, bce)

        print(f"[TMN][{domain}] epoch {ep}/{epochs}  train_bce={train_bce:.6f}  val_bce={val_bce:.6f}")

        if val_bce < best_val:
            best_val = val_bce
            best_state = {
                "domain": domain,
                "seed": seed,
                "processed_root": str(processed_root),
                "num_users": data.num_users,
                "num_items": data.num_items,
                "vocab_size": data.vocab_size,
                "pad_id": pad_id,
                "k1_word_dim": k1,
                "k2_latent_dim": k2,
                "freeze_semantic": freeze_semantic,
                "state_dict": model.state_dict(),
                "best_val_bce": best_val,
                "cfg_tmn": tmn_cfg,
            }

    # ---- save best ----
    if best_state is None:
        raise RuntimeError("TMN training did not produce any checkpoint (unexpected).")

    torch.save(best_state, ckpt_path)
    print(f"[TMN][{domain}] saved best checkpoint: {ckpt_path} (best_val_bce={best_val:.6f})")

    # 可选：保存一份训练摘要 json（不强制）
    summary_path = tmn_dir / f"{domain}.summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "domain": domain,
                "best_val_bce": best_val,
                "checkpoint": str(ckpt_path),
                "seed": seed,
                "k1_word_dim": k1,
                "k2_latent_dim": k2,
                "freeze_semantic": freeze_semantic,
                "max_user_len": max_user_len,
                "max_item_len": max_item_len,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return ckpt_path


# 允许你单独跑这个脚本（可选）
if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--domain", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    run_train_tmn(cfg, domain=args.domain)
