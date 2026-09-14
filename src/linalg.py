"""Dense linear algebra in pure Python.

numpy is not available in this environment (see README implementation note), so
the reconciliation solve is written out by hand.  Matrices are lists of rows;
every routine below is O(n^3) at worst and the hierarchy is deliberately kept
small (17 nodes, 12 of them at the bottom) so that is irrelevant.
"""

from __future__ import annotations

Matrix = list[list[float]]
Vector = list[float]


def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0] * cols for _ in range(rows)]


def identity(n: int) -> Matrix:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def shape(a: Matrix) -> tuple[int, int]:
    return len(a), (len(a[0]) if a else 0)


def transpose(a: Matrix) -> Matrix:
    return [list(col) for col in zip(*a)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    ar, ac = shape(a)
    br, bc = shape(b)
    if ac != br:
        raise ValueError("matmul dimension mismatch: %dx%d times %dx%d" % (ar, ac, br, bc))
    # Transpose b once so the inner loop walks contiguous lists; naive triple
    # loops over b[k][j] are ~3x slower here and this runs inside a backtest.
    bt = transpose(b)
    out = zeros(ar, bc)
    for i in range(ar):
        ai = a[i]
        oi = out[i]
        for j in range(bc):
            btj = bt[j]
            s = 0.0
            for k in range(ac):
                s += ai[k] * btj[k]
            oi[j] = s
    return out


def matvec(a: Matrix, x: Vector) -> Vector:
    if shape(a)[1] != len(x):
        raise ValueError("matvec dimension mismatch")
    return [sum(row[k] * x[k] for k in range(len(x))) for row in a]


def add(a: Matrix, b: Matrix) -> Matrix:
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def scale(a: Matrix, c: float) -> Matrix:
    return [[c * x for x in row] for row in a]


def cholesky(a: Matrix) -> Matrix:
    """Lower-triangular L with L L' = A, for symmetric positive definite A.

    Raises ValueError if a pivot is non-positive: that is the signal that the
    covariance estimate has gone singular (more nodes than residual samples is
    the usual cause), and silently patching it would hide a real modelling
    failure behind a plausible-looking number.
    """
    n = len(a)
    L = zeros(n, n)
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                d = a[i][i] - s
                if d <= 0.0:
                    raise ValueError("matrix is not positive definite at pivot %d" % i)
                L[i][j] = d ** 0.5
            else:
                L[i][j] = (a[i][j] - s) / L[j][j]
    return L


def cho_solve(L: Matrix, b: Matrix) -> Matrix:
    """Solve (L L') X = B by forward then back substitution."""
    n = len(L)
    rows_b, cols_b = shape(b)
    if rows_b != n:
        raise ValueError("cho_solve dimension mismatch")
    y = zeros(n, cols_b)
    for i in range(n):
        for j in range(cols_b):
            s = sum(L[i][k] * y[k][j] for k in range(i))
            y[i][j] = (b[i][j] - s) / L[i][i]
    x = zeros(n, cols_b)
    for i in range(n - 1, -1, -1):
        for j in range(cols_b):
            s = sum(L[k][i] * x[k][j] for k in range(i + 1, n))
            x[i][j] = (y[i][j] - s) / L[i][i]
    return x


def solve(a: Matrix, b: Matrix) -> Matrix:
    """General solve A X = B by Gaussian elimination with partial pivoting.

    Used for the OLS/WLS normal equations, which are SPD in theory but can be
    fed a user-supplied weight matrix; partial pivoting keeps it honest.
    """
    n = len(a)
    if shape(a)[1] != n:
        raise ValueError("solve requires a square matrix")
    rows_b, cols_b = shape(b)
    if rows_b != n:
        raise ValueError("solve dimension mismatch")
    aug = [list(a[i]) + list(b[i]) for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-300:
            raise ValueError("matrix is singular at column %d" % col)
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        for r in range(n):
            if r == col:
                continue
            f = aug[r][col] / pv
            if f == 0.0:
                continue
            arr = aug[r]
            acr = aug[col]
            for c in range(col, n + cols_b):
                arr[c] -= f * acr[c]
    return [[aug[i][n + j] / aug[i][i] for j in range(cols_b)] for i in range(n)]


def inv(a: Matrix) -> Matrix:
    return solve(a, identity(len(a)))
