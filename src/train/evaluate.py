# src/train/evaluate.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional

import torch

from src.data.dataset import DomainData, build_rank_eval_data
from src.models.tdar import TDAR


def _get_device(cfg: Dict[str, Any]) -> torch.device:
    dev = str(cfg.get("device", "cpu")).lower()
    if dev.startswith("cuda") and torch.cuda.is_available():
        return torch.device(dev)
    return torch.device("cpu")


def _precision_recall_f1_at_k(recommended: List[int], gt_pos: Set[int], k: int) -> Tuple[float, float, float]:
    if k <= 0:
        return 0.0, 0.0, 0.0
    topk = recommended[:k]
    if not topk:
        return 0.0, 0.0, 0.0
    hit = sum((i in gt_pos) for i in topk)
    prec = hit / float(k)
    rec = hit / float(len(gt_pos)) if len(gt_pos) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _ndcg_at_k(recommended: List[int], gt_pos: Set[int], k: int) -> float:
    """
    Binary relevance NDCG@k:
      DCG = sum_{r=1..k} rel_r / log2(r+1)
      IDCG = best possible with |gt_pos| positives
    """
    if k <= 0:
        return 0.0
    topk = recommended[:k]
    if not topk:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(topk, start=1):
        if item in gt_pos:
            dcg += 1.0 / torch.log2(torch.tensor(rank + 1.0)).item()

    ideal_hits = min(len(gt_pos), k)
    if ideal_hits == 0:
        return 0.0

    idcg = 0.0
    for rank in range(1, ideal_hits + 1):
        idcg += 1.0 / torch.log2(torch.tensor(rank + 1.0)).item()

    return dcg / idcg


@torch.no_grad()
def _score_all_items_for_users(
    model: TDAR,
    user_ids: List[int],
    num_items: int,
    device: torch.device,
    item_chunk_size: int = 4096,
) -> torch.Tensor:
    """
    Score all items for each user in user_ids on TARGET model (tcf_t).
    Returns:
      scores: FloatTensor [U, I]
    """
    model.eval()
    U = len(user_ids)
    I = int(num_items)

    # output on CPU to avoid huge GPU memory
    all_scores = torch.empty((U, I), dtype=torch.float32)

    # users on device
    user_tensor = torch.tensor(user_ids, dtype=torch.long, device=device)  # [U]

    for start in range(0, I, item_chunk_size):
        end = min(start + item_chunk_size, I)
        items = torch.arange(start, end, dtype=torch.long, device=device)  # [C]
        C = end - start

        # create pairs (u, i) for all u in batch and items in chunk
        # user_idx_t: [U*C], item_idx_t: [U*C]
        user_idx_t = user_tensor.unsqueeze(1).expand(U, C).reshape(-1)
        item_idx_t = items.unsqueeze(0).expand(U, C).reshape(-1)

        out = model(user_idx_t=user_idx_t, item_idx_t=item_idx_t, compute_domain=False)
        scores = out.target.scores.reshape(U, C).detach().float().cpu()  # [U,C]

        all_scores[:, start:end] = scores

    return all_scores


def _recommend_topk(
    scores_row: torch.Tensor,                 # [I]
    seen_pos: Optional[Set[int]],
    k: int,
) -> List[int]:
    """
    Return top-k item indices after filtering seen_pos.
    Implementation: mask seen_pos to -inf then topk.
    """
    s = scores_row.clone()
    if seen_pos:
        idx = torch.tensor(list(seen_pos), dtype=torch.long)
        # guard: some domains might have empty pos sets
        idx = idx[(idx >= 0) & (idx < s.numel())]
        if idx.numel() > 0:
            s[idx] = -1e9

    # torch.topk requires k<=len; we keep safe
    kk = min(k, s.numel())
    topv, topi = torch.topk(s, kk)
    return topi.tolist()


def run_evaluate(cfg: Dict[str, Any]):
    """
    Evaluate TDAR on target domain.
    By default evaluate split="test", filter_seen="train".
    Output metrics: Precision/Recall/F1/NDCG at K.
    """
    device = _get_device(cfg)
    processed_root = cfg["processed_root"]
    ckpt_root = Path(cfg["checkpoint_root"])

    eval_cfg = cfg.get("eval", {})
    split = str(eval_cfg.get("split", "test")).lower()              # val or test
    filter_seen = str(eval_cfg.get("filter_seen", "train")).lower() # train/all/none
    topk_list = eval_cfg.get("topk", [2, 5, 10, 20, 50, 100])
    topk_list = [int(k) for k in topk_list]

    item_chunk_size = int(eval_cfg.get("item_chunk_size", 4096))
    max_eval_users = eval_cfg.get("max_eval_users", None)  # for quick debug
    if max_eval_users is not None:
        max_eval_users = int(max_eval_users)

    # checkpoint path inferred from config
    source_domain = cfg["source_domain"]
    target_domain = cfg["target_domain"]
    tdar_ckpt_path = ckpt_root / "tdar" / f"{source_domain}_to_{target_domain}.pt"
    if not tdar_ckpt_path.exists():
        raise FileNotFoundError(f"TDAR checkpoint not found: {tdar_ckpt_path}")

    state = torch.load(tdar_ckpt_path, map_location="cpu")
    # build model skeleton
    k1 = int(state["k1_text_dim"])
    k3 = int(state["k3_embed_dim"])
    interaction = str(state.get("interaction", "mlp")).lower()
    tdar_cfg = state.get("cfg_tdar", cfg.get("tdar", {}))

    mlp_hidden = tdar_cfg.get("mlp_hidden", [256, 128])
    dropout = float(tdar_cfg.get("dropout", 0.0))
    reg_only_embeddings = bool(tdar_cfg.get("reg_only_embeddings", True))
    cls_hidden = tdar_cfg.get("cls_hidden", [256, 128])
    cls_dropout = float(tdar_cfg.get("cls_dropout", 0.0))
    grl_lambda = float(tdar_cfg.get("grl_lambda", 1.0))

    # load processed domains
    src = DomainData.load(processed_root, source_domain)
    tgt = DomainData.load(processed_root, target_domain)

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
    model.load_state_dict(state["state_dict"], strict=True)

    # IMPORTANT: TDAR forward requires cached E/F in each TCF model.
    # In our training script, we did NOT store E/F in checkpoint (to avoid size).
    # So for evaluation, we must reconstruct E/F by loading TMN checkpoints and exporting again.
    # (This matches our training-time pipeline.)
    tmn_ckpt_s = Path(state["tmn_ckpt_s"])
    tmn_ckpt_t = Path(state["tmn_ckpt_t"])
    if not tmn_ckpt_s.exists() or not tmn_ckpt_t.exists():
        raise FileNotFoundError("TMN checkpoints referenced in TDAR ckpt are missing.")

    # Recompute E/F for both domains (only need target for scoring, but keep consistent)
    from src.models.tmn import TMN as TMNModel

    st_s = torch.load(tmn_ckpt_s, map_location="cpu")
    st_t = torch.load(tmn_ckpt_t, map_location="cpu")

    max_user_len = tdar_cfg.get("max_user_len", None)
    max_item_len = tdar_cfg.get("max_item_len", None)
    if max_user_len is not None:
        max_user_len = int(max_user_len)
    if max_item_len is not None:
        max_item_len = int(max_item_len)

    export_bs = int(tdar_cfg.get("export_batch_size", 4096))

    tmn_s = TMNModel(
        num_users=src.num_users,
        num_items=src.num_items,
        vocab_size=src.vocab_size,
        k1_word_dim=int(st_s["k1_word_dim"]),
        k2_latent_dim=int(st_s["k2_latent_dim"]),
        pad_id=int(src.word2id.get("<PAD>", 0)),
        freeze_semantic=bool(st_s.get("freeze_semantic", False)),
        semantic_init=None,
    ).to(device)
    tmn_s.load_state_dict(st_s["state_dict"], strict=True)
    tmn_s.eval()

    tmn_t = TMNModel(
        num_users=tgt.num_users,
        num_items=tgt.num_items,
        vocab_size=tgt.vocab_size,
        k1_word_dim=int(st_t["k1_word_dim"]),
        k2_latent_dim=int(st_t["k2_latent_dim"]),
        pad_id=int(tgt.word2id.get("<PAD>", 0)),
        freeze_semantic=bool(st_t.get("freeze_semantic", False)),
        semantic_init=None,
    ).to(device)
    tmn_t.load_state_dict(st_t["state_dict"], strict=True)
    tmn_t.eval()

    # helper from train_tdar.py (inline to keep evaluate.py standalone)
    def _pad_docs_all(docs: list[list[int]], pad_id: int = 0, max_len: Optional[int] = None):
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

    def _export_EF(tmn, data):
        pad_id = int(data.word2id.get("<PAD>", 0))
        u_ids, u_mask = _pad_docs_all(data.user_docs, pad_id=pad_id, max_len=max_user_len)
        i_ids, i_mask = _pad_docs_all(data.item_docs, pad_id=pad_id, max_len=max_item_len)
        feats = tmn.export_all_text_features(
            user_docs=u_ids,
            user_mask=u_mask,
            item_docs=i_ids,
            item_mask=i_mask,
            batch_size=export_bs,
            device=device,
        )
        return feats["E"], feats["F"]

    E_s, F_s = _export_EF(tmn_s, src)
    E_t, F_t = _export_EF(tmn_t, tgt)
    model.set_text_features_source(E_s, F_s)
    model.set_text_features_target(E_t, F_t)

    # build eval sets
    eval_data = build_rank_eval_data(tgt, split=split, filter_seen=filter_seen)
    eval_users = eval_data.eval_users
    if max_eval_users is not None:
        eval_users = eval_users[:max_eval_users]

    # score matrix [U, I]
    scores = _score_all_items_for_users(
        model=model,
        user_ids=eval_users,
        num_items=tgt.num_items,
        device=device,
        item_chunk_size=item_chunk_size,
    )

    # compute metrics
    result = {}
    for k in topk_list:
        precs, recs, f1s, ndcgs = [], [], [], []
        for row_idx, u in enumerate(eval_users):
            gt_pos = eval_data.u2eval_pos.get(u, set())
            seen = eval_data.u2seen_pos.get(u, set()) if filter_seen != "none" else None

            rec_list = _recommend_topk(scores[row_idx], seen, k)
            p, r, f1 = _precision_recall_f1_at_k(rec_list, gt_pos, k)
            ndcg = _ndcg_at_k(rec_list, gt_pos, k)

            precs.append(p); recs.append(r); f1s.append(f1); ndcgs.append(ndcg)

        result[f"Precision@{k}"] = float(sum(precs) / max(len(precs), 1))
        result[f"Recall@{k}"] = float(sum(recs) / max(len(recs), 1))
        result[f"F1@{k}"] = float(sum(f1s) / max(len(f1s), 1))
        result[f"NDCG@{k}"] = float(sum(ndcgs) / max(len(ndcgs), 1))

    # print nicely
    print(f"[EVAL][{source_domain}->{target_domain}] split={split} filter_seen={filter_seen} users={len(eval_users)} items={tgt.num_items}")
    for k in topk_list:
        print(
            f"  K={k:>3d}  "
            f"F1={result[f'F1@{k}']:.6f}  "
            f"NDCG={result[f'NDCG@{k}']:.6f}  "
            f"P={result[f'Precision@{k}']:.6f}  "
            f"R={result[f'Recall@{k}']:.6f}"
        )

    # optionally save metrics json next to tdar checkpoint
    save_metrics = bool(eval_cfg.get("save_metrics", True))
    if save_metrics:
        out_path = tdar_ckpt_path.with_suffix(f".{split}.metrics.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "split": split,
                    "filter_seen": filter_seen,
                    "num_eval_users": len(eval_users),
                    "num_items": tgt.num_items,
                    "checkpoint": str(tdar_ckpt_path),
                    "metrics": result,
                    "eval_cfg": eval_cfg,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[EVAL] saved metrics: {out_path}")

    return result


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    run_evaluate(cfg)
