def count_with_parts(n, parts):
    """Count partitions of n using only parts in `parts` (unlimited repetition)."""
    dp = [0] * (n + 1)
    dp[0] = 1
    for p in sorted(parts):
        for i in range(p, n + 1):
            dp[i] += dp[i - p]
    return dp[n]


def compute_N(lhs_counter, max_n=100, verbose=True):
    """
    Build set N and verify p(n|condition) == p(n|parts in N) for n=1..max_n.

    At each n:
      - lhs - rhs == 1  =>  add n to N
      - lhs - rhs == 0  =>  skip
      - anything else   =>  failure

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
        elif diff == 0:
            if verbose:
                print(f"n={n:3d}  lhs={lhs}  rhs={rhs}  -> ok")
        else:
            print(f"FAILURE at n={n}: lhs={lhs}, rhs={rhs}, diff={diff}")
            return N, False
    return N, True


# ---------------------------------------------------------------------------
# Plug your condition in here
# ---------------------------------------------------------------------------

def lhs_counter(n):
    """Replace this body with p(n | your condition)."""
    raise NotImplementedError("lhs_counter not yet defined")


if __name__ == "__main__":
    N, success = compute_N(lhs_counter)
    print()
    print("SUCCESS" if success else "FAILURE")
    print(f"N = {N}")
