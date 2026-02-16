import os
import re
import json
import gzip
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# -------------------------
# IO helpers
# -------------------------

def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def iter_jsonl(path: str) -> Iterable[dict]:
    with open_maybe_gzip(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# -------------------------
# Text processing
# -------------------------

_token_re = re.compile(r"[A-Za-z0-9]+")

def tokenize(text: str) -> List[str]:
    text = text.lower()
    return _token_re.findall(text)


# -------------------------
# Core steps
# -------------------------

def dedup_keep_latest(reviews: Iterable[dict]) -> List[dict]:
    """
    对同一 (reviewerID, asin) 的多条记录，仅保留 unixReviewTime 最大的一条
    """
    best = {}
    for r in reviews:
        u = r.get("reviewerID")
        i = r.get("asin")
        if not u or not i:
            continue
        t = int(r.get("unixReviewTime", 0) or 0)
        key = (u, i)
        if key not in best or t > best[key][0]:
            best[key] = (t, r)
    return [v[1] for v in best.values()]

def k_core_filter(pairs: List[Tuple[str, str, dict]], k: int) -> List[Tuple[str, str, dict]]:
    """
    标准 k-core：反复删除交互数 < k 的用户和物品
    pairs: (user_id, item_id, payload_dict)
    """
    changed = True
    cur = pairs
    while changed:
        changed = False
        ucnt = Counter(u for u, _, _ in cur)
        icnt = Counter(i for _, i, _ in cur)
        new_cur = []
        for u, i, p in cur:
            if ucnt[u] >= k and icnt[i] >= k:
                new_cur.append((u, i, p))
        if len(new_cur) != len(cur):
            changed = True
            cur = new_cur
    return cur

def build_mappings(pairs: List[Tuple[str, str, dict]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    users = sorted({u for u, _, _ in pairs})
    items = sorted({i for _, i, _ in pairs})
    user2idx = {u: idx for idx, u in enumerate(users)}
    item2idx = {i: idx for idx, i in enumerate(items)}
    return user2idx, item2idx

def split_pairs(
    ui_pairs: List[Tuple[int, int]],
    seed: int = 42,
    ratios=(0.8, 0.1, 0.1),
) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]], List[Tuple[int,int]]]:
    assert abs(sum(ratios) - 1.0) < 1e-9
    rng = random.Random(seed)
    ui_pairs = ui_pairs[:]
    rng.shuffle(ui_pairs)
    n = len(ui_pairs)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = ui_pairs[:n_train]
    val = ui_pairs[n_train:n_train + n_val]
    test = ui_pairs[n_train + n_val:]
    return train, val, test

def build_vocab_and_docs(
    pairs: List[Tuple[str, str, dict]],
    user2idx: Dict[str, int],
    item2idx: Dict[str, int],
    vocab_size: int = 50000,
    min_freq: int = 5,
    use_summary: bool = True,
) -> Tuple[Dict[str,int], List[List[int]], List[List[int]]]:
    """
    输出：
    - word2id
    - user_docs: List[List[word_id]] per user_idx
    - item_docs: List[List[word_id]] per item_idx
    """
    cnt = Counter()
    tmp_user_tokens = defaultdict(list)  # user_idx -> tokens
    tmp_item_tokens = defaultdict(list)  # item_idx -> tokens

    for u, i, p in pairs:
        uid = user2idx[u]
        iid = item2idx[i]

        text = (p.get("reviewText") or "")
        if use_summary:
            s = p.get("summary") or ""
            if s:
                text = s + " " + text

        toks = tokenize(text)
        if not toks:
            continue

        cnt.update(toks)
        tmp_user_tokens[uid].extend(toks)
        tmp_item_tokens[iid].extend(toks)

    # 构词表：>= min_freq 的 top vocab_size
    vocab = [w for w, c in cnt.items() if c >= min_freq]
    vocab.sort(key=lambda w: cnt[w], reverse=True)
    vocab = vocab[:vocab_size]

    # 预留 PAD / UNK
    word2id = {"<PAD>": 0, "<UNK>": 1}
    for w in vocab:
        if w not in word2id:
            word2id[w] = len(word2id)

    def to_ids(tokens: List[str]) -> List[int]:
        unk = word2id["<UNK>"]
        return [word2id.get(t, unk) for t in tokens]

    num_users = len(user2idx)
    num_items = len(item2idx)

    user_docs = [[] for _ in range(num_users)]
    item_docs = [[] for _ in range(num_items)]

    for uid, toks in tmp_user_tokens.items():
        user_docs[uid] = to_ids(toks)
    for iid, toks in tmp_item_tokens.items():
        item_docs[iid] = to_ids(toks)

    return word2id, user_docs, item_docs


# -------------------------
# Main pipeline
# -------------------------

def preprocess_amazon_reviews(
    in_path: str,
    out_dir: str,
    k_core: int = 5,
    seed: int = 42,
    vocab_size: int = 50000,
    min_freq: int = 5,
):
    """
    in_path: raw Amazon review json/json.gz (1 line 1 json)
    out_dir: data/processed/<Domain>
             注意：data/ 和 data/processed/ 需你提前创建；这里只会创建子目录
    """

    out_dir = Path(out_dir)
    # 只创建子目录
    ensure_dir(out_dir)
    ensure_dir(out_dir / "splits")
    ensure_dir(out_dir / "text")

    # A) 读原始 reviews，抽取必要字段
    raw = []
    for r in iter_jsonl(in_path):
        u = r.get("reviewerID")
        i = r.get("asin")
        if not u or not i:
            continue
        raw.append({
            "reviewerID": u,
            "asin": i,
            "unixReviewTime": int(r.get("unixReviewTime", 0) or 0),
            "reviewText": r.get("reviewText") or "",
            "summary": r.get("summary") or "",
            "overall": r.get("overall", None),
        })

    # B) 去重：同一 (u,i) 只保留最新
    raw = dedup_keep_latest(raw)

    # C) 组 pair 并做 k-core
    pairs = [(r["reviewerID"], r["asin"], r) for r in raw]
    pairs = k_core_filter(pairs, k=k_core)

    # D) 建映射
    user2idx, item2idx = build_mappings(pairs)

    # E) 写全量交互（implicit 正样本）
    interactions_path = out_dir / "interactions.tsv"
    with open(interactions_path, "w", encoding="utf-8") as f:
        f.write("user_idx\titem_idx\ty\tunixReviewTime\n")
        for u, i, p in pairs:
            f.write(f"{user2idx[u]}\t{item2idx[i]}\t1\t{int(p.get('unixReviewTime',0))}\n")

    # F) 切分（80/10/10）
    ui_pairs = [(user2idx[u], item2idx[i]) for u, i, _ in pairs]
    train, val, test = split_pairs(ui_pairs, seed=seed)

    def dump_split(name, data):
        path = out_dir / "splits" / f"{name}.tsv"
        with open(path, "w", encoding="utf-8") as f:
            f.write("user_idx\titem_idx\ty\n")
            for u, i in data:
                f.write(f"{u}\t{i}\t1\n")
        return path

    train_path = dump_split("train", train)
    val_path = dump_split("val", val)
    test_path = dump_split("test", test)

    # G) 构建 vocab + user/item 文档
    word2id, user_docs, item_docs = build_vocab_and_docs(
        pairs, user2idx, item2idx,
        vocab_size=vocab_size,
        min_freq=min_freq,
        use_summary=True,
    )

    # H) 保存 mappings / vocab
    with open(out_dir / "mappings.json", "w", encoding="utf-8") as f:
        json.dump({"user2idx": user2idx, "item2idx": item2idx}, f, ensure_ascii=False)

    id2word = {str(v): k for k, v in word2id.items()}
    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump({"word2id": word2id, "id2word": id2word}, f, ensure_ascii=False)

    # I) 保存 user_docs / item_docs（jsonl）
    user_words_path = out_dir / "text" / "user_words.jsonl"
    item_words_path = out_dir / "text" / "item_words.jsonl"

    with open(user_words_path, "w", encoding="utf-8") as f:
        for uid, wids in enumerate(user_docs):
            f.write(json.dumps({"user_idx": uid, "word_ids": wids}) + "\n")

    with open(item_words_path, "w", encoding="utf-8") as f:
        for iid, wids in enumerate(item_docs):
            f.write(json.dumps({"item_idx": iid, "word_ids": wids}) + "\n")

    # J) stats
    stats = {
        "num_users": len(user2idx),
        "num_items": len(item2idx),
        "num_interactions": len(pairs),
        "vocab_size": len(word2id),
        "paths": {
            "interactions": str(interactions_path),
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
            "user_words": str(user_words_path),
            "item_words": str(item_words_path),
            "mappings": str(out_dir / "mappings.json"),
            "vocab": str(out_dir / "vocab.json"),
        }
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


# -------------------------
# CLI (可选)
# -------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--k_core", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument("--min_freq", type=int, default=5)
    args = parser.parse_args()

    stats = preprocess_amazon_reviews(
        in_path=args.in_path,
        out_dir=args.out_dir,
        k_core=args.k_core,
        seed=args.seed,
        vocab_size=args.vocab_size,
        min_freq=args.min_freq,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))