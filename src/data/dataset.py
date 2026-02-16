from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Set

import torch
from torch.utils.data import Dataset


# =========================
# Low-level readers
# =========================

def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def _read_tsv_interactions(path: Path, has_time: bool = False) -> List[Tuple[int, int, int]]:
    """
    Read:
      interactions.tsv: user_idx, item_idx, y, unixReviewTime
      ...
      splits/*.tsv: user_idx, item_idx, y
    Return list of (u, i, y)
    """
    data: List[Tuple[int, int, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            u = int(parts[0]); i = int(parts[1]); y = int(parts[2])
            data.append((u, i, y))
    return data

def _read_words_jsonl(path: Path, key: str) -> List[List[int]]:
    """
    key = "user_idx" or "item_idx"
    Each line: {"user_idx": uid, "word_ids": [...]}
    Return: docs[idx] = word_ids
    """
    docs: List[List[int]] = []
    # first pass: get max idx
    max_idx = -1
    lines: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            lines.append(obj)
            max_idx = max(max_idx, int(obj[key]))
    docs = [[] for _ in range(max_idx + 1)]
    for obj in lines:
        idx = int(obj[key])
        docs[idx] = list(map(int, obj.get("word_ids", [])))
    return docs


# =========================
# Domain container
# =========================

@dataclass
class DomainData:
    """
    Holds all processed data for one domain.
    Directory layout (from preprocess_amazon.py):
      processed/<Domain>/
        mappings.json
        vocab.json
        splits/train.tsv, val.tsv, test.tsv
        text/user_words.jsonl, item_words.jsonl
    """
    domain: str
    root: Path

    user2idx: Dict[str, int]
    item2idx: Dict[str, int]
    word2id: Dict[str, int]

    num_users: int
    num_items: int
    vocab_size: int

    train: List[Tuple[int, int, int]]
    val: List[Tuple[int, int, int]]
    test: List[Tuple[int, int, int]]

    user_docs: List[List[int]]  # user_docs[u] -> list of word_ids
    item_docs: List[List[int]]  # item_docs[i] -> list of word_ids

    @staticmethod
    def load(processed_root: str | Path, domain: str) -> "DomainData":
        processed_root = Path(processed_root)
        dom_root = processed_root / domain
        if not dom_root.exists():
            raise FileNotFoundError(f"Domain processed folder not found: {dom_root}")

        mappings = _read_json(dom_root / "mappings.json")
        vocab = _read_json(dom_root / "vocab.json")

        user2idx = mappings["user2idx"]
        item2idx = mappings["item2idx"]
        word2id = vocab["word2id"]

        train = _read_tsv_interactions(dom_root / "splits" / "train.tsv")
        val = _read_tsv_interactions(dom_root / "splits" / "val.tsv")
        test = _read_tsv_interactions(dom_root / "splits" / "test.tsv")

        user_docs = _read_words_jsonl(dom_root / "text" / "user_words.jsonl", key="user_idx")
        item_docs = _read_words_jsonl(dom_root / "text" / "item_words.jsonl", key="item_idx")

        num_users = len(user2idx)
        num_items = len(item2idx)

        # 有时 jsonl 的 max_idx 可能 < num_users（极少见），这里做防御性补齐
        if len(user_docs) < num_users:
            user_docs.extend([[] for _ in range(num_users - len(user_docs))])
        if len(item_docs) < num_items:
            item_docs.extend([[] for _ in range(num_items - len(item_docs))])

        return DomainData(
            domain=domain,
            root=dom_root,
            user2idx=user2idx,
            item2idx=item2idx,
            word2id=word2id,
            num_users=num_users,
            num_items=num_items,
            vocab_size=len(word2id),
            train=train,
            val=val,
            test=test,
            user_docs=user_docs,
            item_docs=item_docs,
        )

    def build_user_pos(self, split: str = "train") -> Dict[int, Set[int]]:
        """
        Build u -> set(pos_items) from a split (train/val/test).
        """
        if split == "train":
            data = self.train
        elif split == "val":
            data = self.val
        elif split == "test":
            data = self.test
        else:
            raise ValueError(f"Unknown split: {split}")

        u2pos: Dict[int, Set[int]] = {}
        for u, i, y in data:
            if y != 1:
                continue
            u2pos.setdefault(u, set()).add(i)
        return u2pos

    def build_all_pos(self) -> Dict[int, Set[int]]:
        """
        Build u -> set(pos_items) from train+val+test (useful for eval filtering).
        """
        u2pos: Dict[int, Set[int]] = {}
        for data in (self.train, self.val, self.test):
            for u, i, y in data:
                if y != 1:
                    continue
                u2pos.setdefault(u, set()).add(i)
        return u2pos


# =========================
# Datasets
# =========================

class InteractionDataset(Dataset):
    """
    Basic dataset returning (u, i, y).
    """
    def __init__(self, interactions: List[Tuple[int, int, int]]):
        self.data = interactions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        u, i, y = self.data[idx]
        return u, i, y


class InteractionWithTextDataset(Dataset):
    """
    Returns (u, i, y, user_word_ids, item_word_ids)
    Suitable for TMN/TCF training where you want textual docs in batch.
    """
    def __init__(
        self,
        interactions: List[Tuple[int, int, int]],
        user_docs: List[List[int]],
        item_docs: List[List[int]],
    ):
        self.data = interactions
        self.user_docs = user_docs
        self.item_docs = item_docs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        u, i, y = self.data[idx]
        return u, i, y, self.user_docs[u], self.item_docs[i]


class NegativeSamplingDataset(Dataset):
    """
    Online negative sampling for implicit feedback.

    Each __getitem__ returns one positive + (neg_ratio) negatives:
      - positives: (u, pos_i, 1)
      - negatives: (u, neg_i, 0)

    Output format:
      u: int
      items: List[int]  length = 1 + neg_ratio
      labels: List[int] length = 1 + neg_ratio

    Typically used for SOURCE domain in TDAR (provide negative supervision).
    """
    def __init__(
        self,
        positives: List[Tuple[int, int, int]],
        num_items: int,
        user_pos_dict: Dict[int, Set[int]],
        neg_ratio: int = 4,
        seed: int = 42,
        with_text: bool = False,
        user_docs: Optional[List[List[int]]] = None,
        item_docs: Optional[List[List[int]]] = None,
    ):
        self.pos = [(u, i) for (u, i, y) in positives if y == 1]
        self.num_items = int(num_items)
        self.user_pos = user_pos_dict
        self.neg_ratio = int(neg_ratio)
        self.rng = torch.Generator()
        self.rng.manual_seed(int(seed))

        self.with_text = bool(with_text)
        self.user_docs = user_docs
        self.item_docs = item_docs
        if self.with_text:
            assert self.user_docs is not None and self.item_docs is not None

    def __len__(self):
        return len(self.pos)

    def _sample_negative(self, u: int) -> int:
        # rejection sampling
        pos_set = self.user_pos.get(u, set())
        while True:
            neg_i = int(torch.randint(low=0, high=self.num_items, size=(1,), generator=self.rng).item())
            if neg_i not in pos_set:
                return neg_i

    def __getitem__(self, idx: int):
        u, pos_i = self.pos[idx]

        items = [pos_i]
        labels = [1]

        for _ in range(self.neg_ratio):
            items.append(self._sample_negative(u))
            labels.append(0)

        if not self.with_text:
            return u, items, labels

        # with text
        u_words = self.user_docs[u]
        i_words = [self.item_docs[it] for it in items]
        return u, items, labels, u_words, i_words


# =========================
# Collate functions
# =========================

def _pad_1d(seqs: List[List[int]], pad_id: int = 0, max_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of 1D int sequences to [B, L].
    Return:
      padded: LongTensor [B, L]
      mask:   BoolTensor [B, L]  True where real token
    """
    if max_len is None:
        max_len = max((len(s) for s in seqs), default=0)
    B = len(seqs)
    padded = torch.full((B, max_len), fill_value=pad_id, dtype=torch.long)
    mask = torch.zeros((B, max_len), dtype=torch.bool)
    for b, s in enumerate(seqs):
        L = min(len(s), max_len)
        if L > 0:
            padded[b, :L] = torch.tensor(s[:L], dtype=torch.long)
            mask[b, :L] = True
    return padded, mask

def collate_pad_text(batch, pad_id: int = 0, max_user_len: Optional[int] = None, max_item_len: Optional[int] = None):
    """
    Collate for InteractionWithTextDataset:
      input item: (u, i, y, u_words, i_words)
    Output:
      u: LongTensor [B]
      i: LongTensor [B]
      y: FloatTensor [B]
      u_words: LongTensor [B, Lu]
      u_mask:  BoolTensor [B, Lu]
      i_words: LongTensor [B, Li]
      i_mask:  BoolTensor [B, Li]
    """
    u_list, i_list, y_list, u_words_list, i_words_list = [], [], [], [], []
    for u, i, y, uw, iw in batch:
        u_list.append(u)
        i_list.append(i)
        y_list.append(y)
        u_words_list.append(uw)
        i_words_list.append(iw)

    u = torch.tensor(u_list, dtype=torch.long)
    i = torch.tensor(i_list, dtype=torch.long)
    y = torch.tensor(y_list, dtype=torch.float32)

    u_words, u_mask = _pad_1d(u_words_list, pad_id=pad_id, max_len=max_user_len)
    i_words, i_mask = _pad_1d(i_words_list, pad_id=pad_id, max_len=max_item_len)
    return u, i, y, u_words, u_mask, i_words, i_mask


def collate_neg_sampling(batch, pad_id: int = 0, max_user_len: Optional[int] = None, max_item_len: Optional[int] = None):
    """
    Collate for NegativeSamplingDataset (with_text=False):
      input item: (u, items, labels)
    Output:
      u: LongTensor [B]
      items: LongTensor [B, 1+neg_ratio]
      labels: FloatTensor [B, 1+neg_ratio]
    """
    u_list, items_list, labels_list = [], [], []
    for u, items, labels in batch:
        u_list.append(u)
        items_list.append(items)
        labels_list.append(labels)

    u = torch.tensor(u_list, dtype=torch.long)
    items = torch.tensor(items_list, dtype=torch.long)
    labels = torch.tensor(labels_list, dtype=torch.float32)
    return u, items, labels


def collate_neg_sampling_with_text(batch, pad_id: int = 0, max_user_len: Optional[int] = None, max_item_len: Optional[int] = None):
    """
    Collate for NegativeSamplingDataset (with_text=True):
      input item: (u, items, labels, u_words, i_words_list)
    Output:
      u: LongTensor [B]
      items: LongTensor [B, 1+neg_ratio]
      labels: FloatTensor [B, 1+neg_ratio]
      u_words: LongTensor [B, Lu]
      u_mask: BoolTensor [B, Lu]
      i_words: LongTensor [B*(1+neg_ratio), Li]  (flatten)
      i_mask:  BoolTensor [B*(1+neg_ratio), Li]
    """
    u_list, items_list, labels_list, u_words_list = [], [], [], []
    all_i_words: List[List[int]] = []

    for u, items, labels, u_words, i_words_list in batch:
        u_list.append(u)
        items_list.append(items)
        labels_list.append(labels)
        u_words_list.append(u_words)
        all_i_words.extend(i_words_list)

    u = torch.tensor(u_list, dtype=torch.long)
    items = torch.tensor(items_list, dtype=torch.long)
    labels = torch.tensor(labels_list, dtype=torch.float32)

    u_words, u_mask = _pad_1d(u_words_list, pad_id=pad_id, max_len=max_user_len)
    i_words, i_mask = _pad_1d(all_i_words, pad_id=pad_id, max_len=max_item_len)
    return u, items, labels, u_words, u_mask, i_words, i_mask


# =========================
# Eval helper container
# =========================

@dataclass
class RankEvalData:
    """
    Minimal structure for ranking eval:
      - eval_users: users appearing in val/test
      - u2eval_pos: positives in this split (ground truth)
      - u2seen_pos: positives to filter out (e.g., train positives, or all positives)
    """
    eval_users: List[int]
    u2eval_pos: Dict[int, Set[int]]
    u2seen_pos: Dict[int, Set[int]]


def build_rank_eval_data(domain_data: DomainData, split: str = "test", filter_seen: str = "train") -> RankEvalData:
    """
    split: "val" or "test"
    filter_seen:
      - "train": filter out training positives only (common)
      - "all": filter out train+val+test positives (strict)
      - "none": no filtering
    """
    u2eval = domain_data.build_user_pos(split=split)

    if filter_seen == "train":
        u2seen = domain_data.build_user_pos(split="train")
    elif filter_seen == "all":
        u2seen = domain_data.build_all_pos()
    elif filter_seen == "none":
        u2seen = {}
    else:
        raise ValueError(f"Unknown filter_seen: {filter_seen}")

    eval_users = sorted(u2eval.keys())
    return RankEvalData(eval_users=eval_users, u2eval_pos=u2eval, u2seen_pos=u2seen)
