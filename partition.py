import os
import pickle
from functools import lru_cache

CACHE_FILE = "partitions_cache.pkl"


# ---------------------------------------------------------------------------
# Core framework
# ---------------------------------------------------------------------------

def count_with_parts(n, parts):
    """Count partitions of n using only parts in `parts` (unlimited repetition)."""
    dp = [0] * (n + 1)
    dp[0] = 1
    for p in sorted(parts):
        for i in range(p, n + 1):
            dp[i] += dp[i - p]
    return dp[n]


def compute_N(lhs_counter, max_n=100, verbose=True, parts_by_n=None):
    """
    Build set N and verify p(n|condition) == p(n|parts in N) for n=1..max_n.

    At each n:
      - lhs - rhs == 1  =>  add n to N
      - lhs - rhs == 0  =>  skip
      - anything else   =>  failure

    parts_by_n: optional dict {n: [list of partitions]} to print alongside counts.
    Returns (N, success: bool).
    """
    N = []
    for n in range(1, max_n + 1):
        lhs = lhs_counter(n)
        rhs = count_with_parts(n, N)
        diff = lhs - rhs
        if diff == 1:
            N.append(n)
            if verbose:
                print(f"n={n:3d}  lhs={lhs}  rhs={rhs}  -> added {n} to N")
                if parts_by_n:
                    print(f"         partitions: {parts_by_n[n]}")
        elif diff == 0:
            if verbose:
                print(f"n={n:3d}  lhs={lhs}  rhs={rhs}  -> ok")
                if parts_by_n:
                    print(f"         partitions: {parts_by_n[n]}")
        else:
            print(f"FAILURE at n={n}: lhs={lhs}, rhs={rhs}, diff={diff}")
            return N, False
    return N, True


def precompute(conditions, max_n=100, force=False):
    """
    Generate and cache partition lists for each condition.
    Loads from CACHE_FILE if it exists (unless force=True).
    Returns dict: {label: {n: [partitions]}}
    """
    if not force and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        if cache.get("__max_n__") == max_n and all(
            label in cache for label, *_ in conditions
        ):
            print(f"Loaded cache from {CACHE_FILE}")
            return cache

    print(f"Precomputing partitions for n=1..{max_n} ...")
    cache = {"__max_n__": max_n}
    for label, _counter, generator in conditions:
        print(f"  {label} ...", end=" ", flush=True)
        cache[label] = {n: list(generator(n)) for n in range(1, max_n + 1)}
        total = sum(len(v) for v in cache[label].values())
        print(f"{total} partitions total")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    print(f"Saved to {CACHE_FILE}\n")
    return cache


# ---------------------------------------------------------------------------
# Condition 50: Second Rogers-Ramanujan
# 2-distinct parts >= 2: all parts >= 2, consecutive parts differ by >= 2
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _rr2(remaining, min_part):
    if remaining == 0:
        return 1
    return sum(_rr2(remaining - p, p + 2) for p in range(min_part, remaining + 1))

def lhs_rr2(n):
    return _rr2(n, 2)

def gen_rr2(n):
    def helper(remaining, min_part, current):
        if remaining == 0:
            yield tuple(current)
            return
        for p in range(min_part, remaining + 1):
            current.append(p)
            yield from helper(remaining - p, p + 2, current)
            current.pop()
    yield from helper(n, 2, [])


# ---------------------------------------------------------------------------
# Conditions 51 & 52: Göllnitz-Gordon
# 2-distinct (gap >= 2), no consecutive even multiples (if 2k present, 2k+2 not).
# Condition 51: parts >= 1.  Condition 52: parts >= 3.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _gg(remaining, min_part):
    if remaining == 0:
        return 1
    total = 0
    for p in range(min_part, remaining + 1):
        next_min = p + 3 if p % 2 == 0 else p + 2
        total += _gg(remaining - p, next_min)
    return total

def lhs_gg1(n): return _gg(n, 1)
def lhs_gg2(n): return _gg(n, 3)

def _gen_gg(n, start):
    def helper(remaining, min_part, current):
        if remaining == 0:
            yield tuple(current)
            return
        for p in range(min_part, remaining + 1):
            next_min = p + 3 if p % 2 == 0 else p + 2
            current.append(p)
            yield from helper(remaining - p, next_min, current)
            current.pop()
    yield from helper(n, start, [])

def gen_gg1(n): yield from _gen_gg(n, 1)
def gen_gg2(n): yield from _gen_gg(n, 3)


# ---------------------------------------------------------------------------
# Condition 53: Schur's identity
# 3-distinct (gap >= 3), no consecutive multiples of 3 (if 3k present, 3k+3 not)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _schur(remaining, min_part):
    if remaining == 0:
        return 1
    total = 0
    for p in range(min_part, remaining + 1):
        next_min = p + 4 if p % 3 == 0 else p + 3
        total += _schur(remaining - p, next_min)
    return total

def lhs_schur(n):
    return _schur(n, 1)

def gen_schur(n):
    def helper(remaining, min_part, current):
        if remaining == 0:
            yield tuple(current)
            return
        for p in range(min_part, remaining + 1):
            next_min = p + 4 if p % 3 == 0 else p + 3
            current.append(p)
            yield from helper(remaining - p, next_min, current)
            current.pop()
    yield from helper(n, 1, [])


# ---------------------------------------------------------------------------
# Condition 54: Gordon's identity
# Parts >= 2, each appears at most twice; if a part appears twice,
# neither adjacent value (p-1 or p+1) is a part.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _gordon(remaining, last_part, last_count):
    if remaining == 0:
        return 1
    min_part = 2 if last_count == 0 else (last_part + 1 if last_count == 1 else last_part + 2)
    total = 0
    for p in range(min_part, remaining + 1):
        total += _gordon(remaining - p, p, 1)
        if remaining >= 2 * p and last_part != p - 1:
            total += _gordon(remaining - 2 * p, p, 2)
    return total

def lhs_gordon(n):
    return _gordon(n, 0, 0)

def gen_gordon(n):
    def helper(remaining, last_part, last_count, current):
        if remaining == 0:
            yield tuple(current)
            return
        min_part = 2 if last_count == 0 else (last_part + 1 if last_count == 1 else last_part + 2)
        for p in range(min_part, remaining + 1):
            current.append(p)
            yield from helper(remaining - p, p, 1, current)
            current.pop()
            if remaining >= 2 * p and last_part != p - 1:
                current.extend([p, p])
                yield from helper(remaining - 2 * p, p, 2, current)
                current.pop(); current.pop()
    yield from helper(n, 0, 0, [])


# ---------------------------------------------------------------------------
# Condition 55: Andrews's identity
# Parts >= 2; odd parts are distinct and each is >= 3 more than the
# next smaller part (the part immediately below it when sorted).
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _andrews(remaining, last_part, last_odd):
    if remaining == 0:
        return 1
    total = 0
    min_even = (last_part + 1) if last_odd else max(2, last_part)
    if min_even % 2 != 0:
        min_even += 1
    min_even = max(min_even, 2)
    for p in range(min_even, remaining + 1, 2):
        total += _andrews(remaining - p, p, False)
    min_odd = max(3, last_part + 3)
    if min_odd % 2 == 0:
        min_odd += 1
    for p in range(min_odd, remaining + 1, 2):
        total += _andrews(remaining - p, p, True)
    return total

def lhs_andrews(n):
    return _andrews(n, 0, False)

def gen_andrews(n):
    def helper(remaining, last_part, last_odd, current):
        if remaining == 0:
            yield tuple(current)
            return
        min_even = (last_part + 1) if last_odd else max(2, last_part)
        if min_even % 2 != 0:
            min_even += 1
        min_even = max(min_even, 2)
        for p in range(min_even, remaining + 1, 2):
            current.append(p)
            yield from helper(remaining - p, p, False, current)
            current.pop()
        min_odd = max(3, last_part + 3)
        if min_odd % 2 == 0:
            min_odd += 1
        for p in range(min_odd, remaining + 1, 2):
            current.append(p)
            yield from helper(remaining - p, p, True, current)
            current.pop()
    yield from helper(n, 0, False, [])


# ---------------------------------------------------------------------------
# Conditions table: (label, lhs_counter, generator)
# ---------------------------------------------------------------------------

CONDITIONS = [
    ("50: Second Rogers-Ramanujan (2-distinct parts >= 2)",             lhs_rr2,    gen_rr2),
    ("51: First Göllnitz-Gordon (2-distinct, no consec even multiples)", lhs_gg1,   gen_gg1),
    ("52: Second Göllnitz-Gordon (2-distinct >= 3, no consec even)",     lhs_gg2,   gen_gg2),
    ("53: Schur (3-distinct, no consec multiples of 3)",                 lhs_schur, gen_schur),
    ("54: Gordon (parts >= 2, at most twice, if twice no adjacent)",     lhs_gordon,gen_gordon),
    ("55: Andrews (parts >= 2, odd distinct, >= 3 more than next smaller)", lhs_andrews, gen_andrews),
]


if __name__ == "__main__":
    MAX_N = 100
    cache = precompute(CONDITIONS, max_n=MAX_N)

    for label, counter, _gen in CONDITIONS:
        print(f"\n{'='*60}")
        print(f"Condition {label}")
        print("="*60)
        parts_by_n = cache[label]
        N, success = compute_N(counter, max_n=MAX_N, verbose=True, parts_by_n=parts_by_n)
        print()
        print("SUCCESS" if success else "FAILURE")
        print(f"N = {N}")
