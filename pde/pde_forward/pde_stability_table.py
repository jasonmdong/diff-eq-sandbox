import math
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# u_x + u_y + u = e^{x+2y}, u(x,0) = 0
# Backward scheme: u[i][j+1] = (2-h)*u[i][j] - u[i+1][j] + h*exp(x + 2y)
#
# The scheme is always unstable — corner value grows ~2^N, overflowing float64
# at N ~ 1023 (= log2 of max double ~1.8e308).
#
# For each mmax, find the maximum N (= minimum h = mmax/N) before the corner
# value overflows to inf. That h is the "barely touches infinity" threshold.


def solve_corner(mmax, N):
    h = mmax / N
    u = np.zeros((N + 2, N + 2), dtype=np.float64)
    x_vals = np.arange(N) * h
    for j in range(N):
        y_val = j * h
        u[:N, j + 1] = (2 - h) * u[:N, j] - u[1:N + 1, j] + h * np.exp(x_vals + 2 * y_val)
    return u[N - 1, N - 1]


def find_max_finite_N(mmax, N_hi=2000):
    """
    Binary search for the largest N where the corner is still a finite float.
    Below this N: finite (but huge). Above: inf.
    """
    if math.isfinite(solve_corner(mmax, N_hi)):
        return None  # need a larger N_hi

    N_lo = 1
    while N_hi - N_lo > 1:
        mid = (N_lo + N_hi) // 2
        if math.isfinite(solve_corner(mmax, mid)):
            N_lo = mid
        else:
            N_hi = mid
    return N_hi


def exact_value(mmax):
    return math.exp(3 * mmax) / 4 - math.exp(-mmax) / 4


def mmax_label(mmax):
    if mmax >= 1:
        return str(int(mmax))
    return f"1/{round(1/mmax)}"


mmax_values = [2 / 2**i for i in range(7)]  # 2, 1, 1/2, 1/4, 1/8, 1/16, 1/32

print(f"{'mmax':>6} | {'N_crit':>7} | {'h_crit':>13} | {'corner (num)':>14} | {'exact':>12} | {'ratio num/exact':>16}")
print("-" * 82)

for mmax in mmax_values:
    N_crit = find_max_finite_N(mmax)
    label = mmax_label(mmax)

    if N_crit is None:
        print(f"{label:>6} | {'N/A':>7} | {'N/A':>13} | {'N/A':>14} | {'N/A':>12} | {'N/A':>16}")
        continue

    h_crit = mmax / N_crit
    corner = solve_corner(mmax, N_crit)
    ex = exact_value(mmax)
    ratio = corner / ex

    print(f"{label:>6} | {N_crit:>7d} | {h_crit:>13.8f} | {corner:>14.4e} | {ex:>12.6f} | {ratio:>16.4e}")

print()
print("N_crit ~ 1033 for all mmax because the scheme grows as (2-h)^N ~ 2^N,")
print("overflowing float64 (~1.8e308 = 2^1023) at N ~ 1023-1040 regardless of mmax.")
print("h_crit = mmax / N_crit scales linearly with mmax.")
