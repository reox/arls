"""
Copyright (c) 2020 Rondall E. Jones, Ph.D.
Copyright (c) 2021 The Python Packaging Authority
Copyright (c) 2023 Sebastian Bachmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

(MIT License)


This module contains utility functions used by the arls functions.
"""
import numpy as np
from scipy._lib._util import _asarray_validated
from scipy.linalg.misc import LinAlgError


def checkAb(A, b):
    """Check shapes of A and b and convert to correct matrix/vector"""
    if A.ndim != 2:
        raise LinAlgError("Input array should be 2-D.")
    m, n = A.shape
    if m == 0 or n == 0:
        raise LinAlgError("Matrix is empty.")
    if len(b) != m:
        raise LinAlgError("Matrix and RHS do not have the same number of rows.")
    A = np.atleast_2d(_asarray_validated(A, check_finite=True))
    b = np.atleast_1d(_asarray_validated(b, check_finite=True))
    return A, b


def decide_width(mg):
    widths = ((2, 1),
              (8, 2),
              (12, 3),
              (20, 4),
              (28, 5),
              (36, 6),
              (50, 7),
              (64, 8),
              (80, 9),
              (200, 10),
              (300, 12),
              (400, 14),
              (1000, 16))
    for lim_mg, width in widths:
        if mg <= lim_mg:
            return width
    # Larger than the maximum
    return 20    


def decide_multiple(width):
    if width < 3:
        return 30.0
    elif width <= 10:
        return 20.0
    elif width <= 20:
        return 15.0
    else:
        return 7.0


def compute_mov_sums(g, mg, w):
    numsums = mg - w + 1
    sums = np.zeros(numsums)
    for i in range(0, numsums):
        sums[i] = np.sum(g[i:i+w])
    return sums


def splita(g, mg):
    """ Determines a usable rank based on large rise in Picard Vector"""
    # initialize
    if mg < 2:
        return mg
    w = decide_width(mg)
    sensitivity = g[0]
    small = sensitivity
    local = sensitivity
    urank = 1
    for i in range(1, mg):
        sensitivity = g[i]
        if i >= w and sensitivity > 25.0 * small and sensitivity > local:
            break
        if sensitivity < small:
            small = small + 0.40 * (sensitivity - small)
        else:
            small = small + 0.10 * (sensitivity - small)
        local = local + 0.40 * (sensitivity - local)
        urank = i + 1
    return urank



def splitb(g, mg):
    """Determines a usable rank based on modest rise in Picard Vector
    after the low point in the PCV."""
    w = decide_width(mg)
    if w < 2:
        return mg  # splitb needs w>=2 to be reliable

    # magnify any divergence by squaring
    gg = g[:mg]**2

    # ignore dropouts
    for i in range(1, mg - 1):
        if gg[i] < 0.2 * gg[i - 1] and gg[i] < 0.2 * gg[i + 1]:
            gg[i] = 0.5 * min(gg[i - 1], gg[i + 1])

    # choose breakpoint as multiple of lowest moving average
    sums = compute_mov_sums(gg, mg, w)
    ilow = np.argmin(sums)
    # bad = 20.0 * sums[ilow]
    multi = decide_multiple(w)
    bad = multi * sums[ilow]

    # look for unexpected rise
    ibad = 0
    for i in range(ilow + 1, mg - w + 1):
        if sums[i] > bad:
            ibad = i
            break
    if ibad <= 0:
        urank = mg  # leave urank alone
    else:
        urank = ibad + w - 1

    return urank


def rmslambdah(A, b, U, S, Vt, ur, lamb):
    """Computes a regularized solution to Ax=b, given the usable rank
    and the Tikhonov lambda value."""
    mn = S.shape[0]
    ps = np.zeros(mn)
    for i in range(0, ur):
        ps[i] = 1.0 / (S[i] + lamb ** 2 / S[i]) if S[i] > 0.0 else 0.0

    # best to do multiplies from right end....
    xa = np.transpose(Vt) @ (np.diag(ps) @ (np.transpose(U) @ b))
    res = b - A @ xa
    r = np.sqrt(np.mean(res**2))
    return xa, r


def discrep(A, b, U, S, Vt, ur, mysigma):
    """ Computes Tikhonov's lambda using b's estimated RMS error, mysigma"""
    lo = 0.0  # for minimum achievable residual
    hi = 0.33 * float(S[0])  # for ridiculously large residual
    lamb = 0.0
    # bisect until we get the residual we want...but quit eventually
    for k in range(0, 50):
        lamb = (lo + hi) * 0.5
        xa, check = rmslambdah(A, b, U, S, Vt, ur, lamb)
        if abs(check - mysigma) < 0.0000001 * mysigma:
            break  # close enough!
        if check > mysigma:
            hi = lamb
        else:
            lo = lamb
    return lamb


def find_max_sense(E, f):
    """find the row of Ex=f which his the highest ratio of f[i]
    to the norm of the row."""
    snmax = -1.0
    ibest = 0  # default
    m = E.shape[0]
    for i in range(0, m):
        rn = np.linalg.norm(E[i, :])
        if rn > 0.0:
            s = abs(f[i]) / rn
            if s > snmax:
                snmax = s
                ibest = i
    return ibest


def prepeq(E, f, neglect):
    """a utility routine for arlseq() below that prepares the equality
    constraints for use"""
    E = np.atleast_2d(_asarray_validated(E, check_finite=True))
    f = np.atleast_1d(_asarray_validated(f, check_finite=True))
    EE = E.copy()
    ff = f.copy()
    m, n = EE.shape
    for i in range(0, m):
        # determine new best row and put it next
        if i == 0:
            imax = find_max_sense(EE, ff)
        else:
            rnmax = -1.0
            imax = -1
            for k in range(i, m):
                rn = np.linalg.norm(EE[k, :])
                if rn > rnmax:
                    rnmax = rn
                    imax = k
        EE[[i, imax], :] = EE[[imax, i], :]
        ff[[i, imax]] = ff[[imax, i]]

        # normalize
        rin = np.linalg.norm(EE[i, :])
        if rin > 0.0:
            EE[i, :] /= rin
            ff[i] /= rin
        else:
            ff[i] = 0.0

        # subtract projections onto EE[i,:]
        for k in range(i + 1, m):
            d = np.dot(EE[k, :], EE[i, :])
            EE[k, :] -= d * EE[i, :]
            ff[k] -= d * ff[i]

    # reject ill-conditioned rows
    if m > 2:
        g = np.zeros(m)
        for k in range(0, m):
            g[k] = abs(ff[k])
        m1 = splita(g, m)
        mm = splitb(g, m1)
        if mm < m:
            EE = np.resize(EE, (mm, n))
            ff = np.resize(ff, mm)
    return EE, ff


def get_worst(GG, hh, x):
    # assess state of inequalities
    p = -1
    mg = GG.shape[0]
    rhs = GG @ x
    worst = 0.0
    for i in range(0, mg):
        if rhs[i] < hh[i]:
            diff = hh[i] - rhs[i]
            if p < 0 or diff > worst:
                p = i
                worst = diff
    return p

