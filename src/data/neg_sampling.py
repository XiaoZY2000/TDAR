# src/data/neg_sampling.py

from __future__ import annotations

import random
from typing import Dict, Set, List, Iterable, Optional


class NegativeSampler:
    """
    Negative sampler for implicit feedback.

    Given:
      - num_items
      - user_pos_dict: {u: set(pos_item_ids)}

    It can sample k negatives for a user u, avoiding positives.

    Features:
      - reproducible with seed
      - supports sampling with or without replacement
      - handles edge cases (when almost all items are positives)
    """

    def __init__(
        self,
        num_items: int,
        user_pos_dict: Dict[int, Set[int]],
        seed: int = 42,
        with_replacement: bool = False,
        max_tries: int = 100,
    ):
        """
        Args:
          num_items: total number of items
          user_pos_dict: dict u -> set(pos_items)
          seed: random seed
          with_replacement: whether negatives can repeat
          max_tries: max trials to avoid infinite loop when positives are many
        """
        self.num_items = int(num_items)
        self.user_pos_dict = user_pos_dict
        self.with_replacement = bool(with_replacement)
        self.max_tries = int(max_tries)

        self.rng = random.Random(seed)

        # Precompute universe for fallback sampling
        self._all_items = list(range(self.num_items))

    def sample_one(self, u: int) -> int:
        """
        Sample one negative item for user u.
        """
        pos = self.user_pos_dict.get(u, set())

        # If user has no positives, just sample any item
        if not pos:
            return self.rng.randrange(self.num_items)

        # Try random sampling with rejection
        for _ in range(self.max_tries):
            j = self.rng.randrange(self.num_items)
            if j not in pos:
                return j

        # Fallback: sample from complement set (slower but safe)
        # Build candidate list once
        candidates = [i for i in self._all_items if i not in pos]
        if not candidates:
            # Degenerate case: user interacted with all items
            # Just return a random one (will be a "false negative")
            return self.rng.randrange(self.num_items)
        return self.rng.choice(candidates)

    def sample_k(self, u: int, k: int) -> List[int]:
        """
        Sample k negatives for user u.
        """
        k = int(k)
        if k <= 0:
            return []

        pos = self.user_pos_dict.get(u, set())

        # If with replacement: just sample independently
        if self.with_replacement:
            return [self.sample_one(u) for _ in range(k)]

        # Without replacement
        negs: List[int] = []
        used: Set[int] = set()

        # Fast path: if available negatives are small, just enumerate
        if len(pos) < self.num_items:
            # candidates = complement set
            candidates = [i for i in self._all_items if i not in pos]
            if not candidates:
                # Degenerate: no negatives
                return []
            if k >= len(candidates):
                # Return all candidates (shuffled)
                self.rng.shuffle(candidates)
                return candidates[:]
            # Sample without replacement
            return self.rng.sample(candidates, k)

        # General path (rarely used)
        tries = 0
        while len(negs) < k and tries < self.max_tries * k:
            j = self.rng.randrange(self.num_items)
            tries += 1
            if j in pos:
                continue
            if j in used:
                continue
            used.add(j)
            negs.append(j)

        if len(negs) < k:
            # Fallback to complement enumeration
            candidates = [i for i in self._all_items if i not in pos and i not in used]
            self.rng.shuffle(candidates)
            for j in candidates:
                if len(negs) >= k:
                    break
                negs.append(j)

        return negs

    def reseed(self, seed: int):
        """
        Reset RNG seed (useful per-epoch if you want).
        """
        self.rng.seed(int(seed))


def build_user_pos_dict(
    interactions: Iterable[tuple[int, int, float]],
) -> Dict[int, Set[int]]:
    """
    Build user->set(items) from interactions.

    interactions: iterable of (u, i, y) or (u, i, *)
    Only u and i are used.
    """
    u2pos: Dict[int, Set[int]] = {}
    for x in interactions:
        u = int(x[0])
        i = int(x[1])
        if u not in u2pos:
            u2pos[u] = set()
        u2pos[u].add(i)
    return u2pos
