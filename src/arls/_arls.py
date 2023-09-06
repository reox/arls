"""
Copyright (c) 2020 Rondall E. Jones, Ph.D.
Copyright (c) 2021 The Python Packaging Authority

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
"""
import numpy as np
from scipy.linalg.misc import LinAlgError

from ._utils import (
        checkAb,
        splita,
        splitb,
        rmslambdah,
        discrep,
        prepeq,
        get_worst,
        arlspj,
        )


def arlsusv(A, b, U, S, Vt):
    """
    Solves the linear system of equation, Ax = b, for any shape matrix.
    The system can be underdetermined, square, or over-determined.
    That is, A(m,n) can be such that m < n, m = n, or m > n.
    Argument b is a vector of size n.
    This solver automatically detects if the system is ill-conditioned or not
    and in most cases automatically regularizes the problem.

    U, S and Vt constitute the Singular Value Decomposition of A
    which can be computed like this:
        U, S, Vt = np.linalg.svd(A, full_matrices=False)

    The purpose of this routine is to allow you to quickly solve many
    systems of equations which use the SAME MATRIX A.
    Just compute the (expensive) SVD once, and use it as many times as
    you like with different b vectors.

    Please see arls() below for complete details.

    Parameters
    ----------
    A : (m, n) array_like
        Coefficient matrix
    b : (m) array_like
        Column of dependent variables.

    Returns
    -------
    x : (n) array_like column vector, type float.
    nr : int
        The Numerical Rank of A.
    ur : int
        The Minimum Usable Rank.
        Note that "numerical rank" is an attribute of a matrix
        but the "usable rank" that arls() computes is an attribute
        of the problem, Ax=b.
    sigma : float
        The estimated right-hand-side root-mean-square error.
    lambda : float
        The estimated Tikhonov regularization parameter.

    Raises
    ------
    LinAlgError
        If A is not 2-D.
        If A is empty.
        If A and b do not have the same row size.
        If SCIPY's SVD() does not converge.
    """
    A, b = checkAb(A, b)
    m, n = A.shape
    mn = min(m, n)
    if np.count_nonzero(A) == 0 or np.count_nonzero(b) == 0:
        return np.zeros(n), 0, 0, 0.0, 0.0

    # compute contributions to norm of solution
    beta = np.transpose(U) @ b
    k = 0
    g = np.zeros(mn)
    sense = 0.0
    si = 0.0
    cond = max(A.shape) * np.spacing(A.real.dtype.type(1))
    eps = S[0] * cond
    for i in range(0, mn):
        si = S[i]
        if si <= eps:
            break
        sense = beta[i] / si
        if sense < 0.0:
            sense = -sense
        g[i] = sense
        k = i + 1
    nr = k  # traditional numeric rank
    if k <= 0:
        return np.zeros(n), 0, 0, 0.0, 0.0  # failsave check

    # two-stage search for divergence in Picard Condition Vector
    ura = splita(g, k)
    urb = splitb(g, ura)
    ur = min(ura, urb)
    if ur >= mn:
        # problem is not ill-conditioned
        x, check = rmslambdah(A, b, U, S, Vt, ur, 0.0)
        sigma = 0.0
        lambdah = 0.0
    else:
        # from ur, determine sigma
        Utb = np.transpose(U) @ b
        sigma = np.sqrt(np.mean(Utb[ur:mn]**2))
        # from sigma, determine lambda
        lambdah = discrep(A, b, U, S, Vt, ur, sigma)
        # from lambda, determine solution
        x, check = rmslambdah(A, b, U, S, Vt, ur, lambdah)
    return x, nr, ur, sigma, lambdah


def arls(A, b):
    """
    Solves the linear system of equation, Ax = b, for any shape matrix.
    The system can be underdetermined, square, or over-determined.
    That is, A(m,n) can be such that m < n, m = n, or m > n.
    Argument b is a vector of size m.
    This solver automatically detects if the system is ill-conditioned or not,
    then:
     -- If the equations are consistent then the solution will usually be
        exact within round-off error.
     -- If the equations are inconsistent then the the solution will be
        by least-squares. That is, it solves ``min ||b - Ax||_2``.
     -- If the equations are inconsistent and diagnosable as ill-conditioned
        using the principles of the first reference below, the system will be
        automatically regularized and the residual will be larger than minimum.
     -- If either A or b is all zeros then the solution will be all zeros.

    Parameters
    ----------
    A : (m, n) array_like
        Coefficient matrix
    b : (m) array_like
        Column of dependent variables.

    Returns
    -------
    x : (n) array_like column vector, type float.

    Raises
    ------
    LinAlgError
        If A is not 2-D.
        If A is empty.
        If A and b do not have the same row size.
        If SCIPY's SVD() does not converge.

    Examples
    --------
    arls() will behave like any good least-squares solver when the system
    is well conditioned.
    Here is a tiny example of an ill-conditioned system as handled by arls(),

       x + y = 2
       x + 1.01 y =3

    Then A = [[ 1., 1.],
              [ 1., 1.01.]]
    and  b = [2.0, 3.0]

    Then standard solvers will return:
       x = [-98. , 100.]

    But arls() will see the violation of the Picard Condition and return
       x = [1.12216 , 1.12779]

    Notes:
    -----
    1. When the system is ill-conditioned, the process works best when the rows
       of A are scaled so that the elements of b have similar estimated errors.
    2. arls() occasionally may produce a smoother (i.e., more regularized)
       solution than desired. In this case please try scipy routine lsmr.
    3. With any linear equation solver, check that the solution is reasonable.
       In particular, you should check the residual vector, Ax - b.
    4. arls() neither needs nor accepts optional parameters such as iteration
       limits, error estimates, variable bounds, condition number limits, etc.
       It also does not return any error flags as there are no error states.
       As long as the SVD converges (and SVD failure is remarkably rare)
       then arls() and other routines in this package will complete normally.
    5. arls()'s intent (and the intent of all routines in this module)
       is to find a reasonable solution even in the midst of excessive
       inaccuracy, ill-conditioning, singularities, duplicated data, etc.
    6. In view of note 5, arls() is not appropriate for situations
       where the requirements are more for high accuracy rather than
       robustness. So, we assume, in the coding, where needed, that no data
       needs to be considered more accurate than 8 significant figures.

    References
    ----------
    The auto-regularization algorithm in this software arose from the research
    for my dissertation, "Solving Linear Algebraic Systems Arising in the
    Solution of Integral Equations of the First Kind", University of
    New Mexico, Albuquerque, NM, 1985.

    Many thanks to Cleve B. Moler, MatLab creater and co-founder of MathWorks
    for his kindness, energy and insights in guiding my dissertation research.
    My thanks also to Richard Hanson (deceased), co-author of the classic
    "Solving Least Squares Problems", co-creater of BLAS, co-worker, and
    co-advisor for the last year of my dissertation work.

    For for a short presentation on the Picard Condition which is at the heart
    of this package's algorithms, please see http://www.rejtrix.net/
    For a complete description, see "The Discrete Picard Condition for Discrete
    Ill-posed Problems", Per Christian Hansen, 1990.
    See link.springer.com/article/10.1007/BF01933214

    For discussion of incorporating equality and inequality constraints
    (including nonnegativity) in solving linear algebraic problems, see
    "Solving Least Squares Problems", by Charles L. Lawson and
    Richard J. Hanson, Prentice-Hall 1974.
    My implementation of these features has evolved somewhat
    from that fine book, but is based on those algorithms.

    Rondall E. Jones, Ph.D.
    rejones7@msn.com
    """
    A, b = checkAb(A, b)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return arlsusv(A, b, U, S, Vt)[0]


def arlseq(A, b, E, f):
    """Solves the double linear system of equations

       Ax = b  (least squares)
       Ex == f  (exact)

    Both Ax=b and Ex=f can be underdetermined, square,
    or over-determined.

    Ex=f is treated as a set of equality constraints.
    These constraints are usually few in number and well behaved.
    But clearly the caller can easily provide equations in Ex=f that
    are impossible to satisfy as a group. For example, there could be
    one equation requiring x[0]=0, and another requiring x[0]=1.
    And, the solver must deal with there being redundant or other
    pathological situations within the E matrix.
    So the solution process will either solve each equation in Ex=f exactly
    (within roundoff) or if that is impossible, arlseq() will discard
    one or more equations until the remaining equations are solvable
    exactly (within roundoff).
    We will refer below to the solution of this reduced system as "xe".

    After Ex=f is processed as above, the rows of Ax=b will have their
    projections onto every row of Ex=f subtracted from them.
    We will call this reduced set of equations A'x = b'.
    (Thus, the rows of A' will all be orthogonal to the rows of E.)
    This reduced problem A'x = b', will then be solved with arls().
    We will refer to the solution of this system as "xt".

    The final solution will be x = xe + xt.

    Parameters
    ----------
    A : (m, n)  array_like "Coefficient" matrix, type float.
    b : (m)     array_like column of dependent variables, type float.
    E : (me, n) array_like "Coefficient" matrix, type float.
    f : (me)    array_like column of dependent variables, type float.

    Returns
    -------
    x : (n) array_like column, type float.

    Raises
    ------
    LinAlgError
        If A is not 2-D.
        If A is empty.
        If A and b do not have the same row size.
        If E is not 2-D.
        If E is empty.
        If E and f do not have the same row size.
        If A and E do not have the same number of columns.
        If SCIPY's SVD() does not converge.

    Examples
    --------
    Here is a tiny example of a problem which has an "unknown" amount
    of error in the right hand side, but for which the user knows that the
    correct SUM of the unknowns must be 3:

         x + 2 y = 5.3   (Least Squares)
       2 x + 3 y = 7.8
           x + y = 3     ( Exact )

    Then the arrays for arlseq are:

       A = [[ 1.,  2.0],
            [ 2.,  3.0]]
       b = [5.3, 7.8]

       E = [[ 1.0,  1.0]]
       f = [3.0]

    Without using the equality constraint we are given here,
    standard solvers will return [x,y] = [-.3 , 2.8].
    Even arls() will return the same [x,y] = [-.3 , 2.8].
    The residual for this solution is [0.0 , 0.0] (within roundoff).
    But of course x + y = 2.5, not the 3.0 we really want.

    Arlsnn() could help here by disallowing presumably unacceptable
    negative values, producing [x,y] = [0. , 2.6].
    The residual for this solution is [-0.1 , 0.] which is of course
    an increase from zero, but this is natural since we have forced
    the solution away from being the "exact" result, for good reason.
    Note that x + y = 2.6, which is a little better.

    If we solve with arlseq(A,b,E,f) then we get [x,y] = [1.004, 1.996]
    which satisfies x + y = 3 exactly.
    This answer is close to the "correct" answer of [x,y] = [1.0 , 2.0]
    if the right hand side had been the correct [5.,8.] instead of [5.3,7.8].
    The residual for this solution is [-0.3 , 0.2] which is yet larger.
    Again, when adding constraints to the problem the residual
    typically increases, but the solution becomes more acceptable.

    Notes:
    -----
    See arls() above for notes and references.
    """
    A, b = checkAb(A, b)
    m, n = A.shape
    if np.count_nonzero(A) == 0 or np.count_nonzero(b) == 0:
        return np.zeros(n)
    AA = A.copy()
    bb = b.copy()
    rnmax = np.max(np.linalg.norm(AA, axis=1))
    neglect = rnmax * 0.000000001  # see Note 6. for arls()

    E, f = checkAb(E, f)
    EE = E.copy()
    ff = f.copy()
    me, ne = EE.shape
    if n != ne:
        raise LinAlgError(
            "The two matrices do not have the same number of unknowns.")
    EE, ff = prepeq(EE, ff, neglect)

    # decouple AAx=bb from EEx=ff
    AA, bb = arlspj(AA, bb, EE, ff, neglect)

    # final solution
    xe = np.transpose(EE) @ ff
    if AA.shape[0] > 0:
        xt = arls(AA, bb)
        return xt + xe
    else:
        return xe


def arlsgt(A, b, G, h):
    """Solves the double linear system of equations

       Ax = b  (least squares)
       Gx >= h ("greater than" inequality constraints)

    Both Ax=b and Gx>=h can be underdetermined, square, or over-determined.
    Arguments b and h must be single columns.
    Arlsgt() uses arls(), above, as the core solver, and iteratively selects
    rows of Gx>=h to move to a growing list of equality constraints, choosing
    first whatever equation in Gx>=h most violates its requirement.

    Note that "less than" equations can be included by negating
    both sides of the equation, thus turning it into a "greater than".

    If either A or b is all zeros then the solution will be all zeros.

    Parameters
    ----------
    A : (m, n)  array_like "Coefficient" matrix, type float.
    b : (m)     array_like column of dependent variables, type float.
    G : (mg, n) array_like "Coefficient" matrix, type float.
    b : (mg)    array_like column of dependent variables, type float.

    Returns
    -------
    x : (n) array_like column, type float.

    Raises
    ------
    LinAlgError
        If A is not 2-D.
        If A is empty.
        If A and b do not have the same row size.
        If G is not 2-D.
        If G is empty.
        If G and h do not have the same row size.
        If A and G do not have the same number of columns.
        If SCIPY's SVD() does not converge.

    Example
    -------
    Let A = [[1,1,1],
             [0,1,1],
             [1,0,1]]
    and b = [5.9, 5.0, 3.9]

    Then any least-squares solver would produce x = [0.9, 2., 3.]
    The residual for this solution is zero within roundoff.

    But if we happen to know that all the answers should be at least 1.0
    then we can add inequalites to insure that:
        x[0] >= 1
        x[1] >= 1
        x[2] >= 1

    This can be expressed in the matrix equation Gx>=h where
        G = [[1,0,0],
             [0,1,0],
             [0,0,1]]
        h = [1,1,1]

    Then arlsgt(A,b,G,h) produces x = [1., 2.013, 2.872].
    The residual vector and its norm are then:
       res = [-0.015, -0.115, 0.028]
       norm(res) = 0.119

    If the user had just adjusted the least-squares answer of [0.9, 2., 3.]
    to [1., 2., 3.] without re-solving then the residual vector
    and its norm would be
       res = [0.1, 0, 0.1]
       norm(res) = 0.141
    which is significantly larger.
    """
    A, b = checkAb(A, b)
    m, n = A.shape
    if np.count_nonzero(A) == 0 or np.count_nonzero(b) == 0:
        return np.zeros(n)

    G, h = checkAb(G, h)
    mg, ng = G.shape
    if n != ng:
        raise LinAlgError(
            "The two matrices do not have the same number of unknowns.")
    GG = G.copy()
    hh = h.copy()

    EE = []
    ff = []
    me = 0
    ne = 0

    # get initial solution... it might actually be right
    x = arls(A, b)
    nx = np.linalg.norm(x)
    if nx <= 0.0:
        return np.zeros(n)

    # while constraints are not fully satisfied:
    while True:
        p = get_worst(GG, hh, x)
        if p < 0:
            break

        # delete row from GGx=hh
        row = GG[p, :]
        rhsp = hh[p]
        GG = np.delete(GG, p, 0)
        hh = np.delete(hh, p, 0)

        # add row to Ex>=f
        if me == 0:
            EE = np.zeros((1, ng))
            EE[0, :] = row
            ff = np.zeros(1)
            ff[0] = rhsp
            me = 1
            ne = ng
        else:
            me += 1
            EE = np.resize(EE, (me, ne))
            EE[me - 1, :] = row[:]
            ff = np.resize(ff, me)
            ff[me - 1] = rhsp
        # re-solve modified system
        x = arlseq(A, b, EE, ff)
    return x


def arlsnn(A, b):
    """Computes a nonnegative solution of A*x = b.

    Arlsnn() uses arls(), above, as the core solver, and iteratively removes
    variables that violate the nonnegativity constraint.

    Parameters
    ----------
    A : (m, n) array_like
        Coefficient matrix
    b : (m) array_like
        Column of dependent variables.

    Returns
    -------
    x : (n) array_like column vector, type float.

    Raises
    ------
    LinAlgError
        If A is not 2-D.
        If A is empty.
        If A and b do not have the same row size.
        If SCIPY's SVD() does not converge.

    Example
    -------
    Let A = [[2,2,1],
             [2,1,0],
             [1,2,0]]
    and b = [3.9, 3, 2]

    Most least squares solvers will return
       x = [1, 1, -0.1]
    which solves the system exactly (within roundoff).

    But arlsnn() returns
       x = [1.04, 0.92, 0.]
    As is expected, this result has a non-zero residual,
    which is [0.02, 0., -0.04]. This is to be expected.
    """
    A, b = checkAb(A, b)
    m, n = A.shape
    if np.count_nonzero(A) == 0 or np.count_nonzero(b) == 0:
        return np.zeros(n)

    # get initial solution and Tikhonov parameter
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    xt, nt, ur, sigma, lambdah = arlsusv(A, b, U, S, Vt)
    x1 = xt  # remember this
    # see if unconstrained solution is already non-negative
    if min(xt) >= 0.0:
        return xt

    # the approach here is to actually delete columns,
    # for SVD speed and stability,
    # rather than just zero out columns.
    C = A.copy()
    cols = [0] * n  # list of active column numbers
    for i in range(1, n):
        cols[i] = i
    nn = n
    for i in range(1, nn):
        # choose a column to zero
        p = -1
        worst = 0.0
        for j in range(0, nn):
            if xt[j] < worst:
                p = j
                worst = xt[p]
        if p < 0:
            break
        # remove column p and resolve
        C = np.delete(C, p, 1)
        cols.pop(p)
        nn -= 1
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        ms = len(S)
        ps = np.zeros(ms)
        for i in range(0, ms):
            ps[i] = 1.0 / (S[i] + lambdah ** 2 / S[i]) if S[i] > 0.0 else 0.0
        xt = np.transpose(Vt) @ (np.diag(ps) @ (np.transpose(U) @ b))

    # degenerate case: nn==n
    if nn == n:
        return x1
    # degenerate case: nn==1
    if xt[0] < 0.0:
        xt[0] = 0.0
    # rebuild full solution vector
    x = np.zeros(n)
    for j in range(0, nn):
        x[int(cols[j])] = xt[j]
    # double check to be sure solution is nonnegative
    for j in range(0, nn):
        x[j] = max(x[j], 0.0)
    return x


def arlshape(A, b, nonneg, slope, curve):
    """
    ArlsShape() solves the linear system of equation, Ax = b, in the same
    way as arls(). But with ArlsShape the user can choose three different
    types of constraints on the shape of the solution.
    These constraints can be combined in any of 18 ways.
    The three types are:

    - nonneg = 1 for a nonnegative solution.
      nonneg = 0 to skip this constraint.

    - slope =  1 for a rising (or non-decreasing) solution.
      slope = -1 for a falling (or non-increasing) solution.
      slope =  0 to skip this constraint.

    - curve =  1 for a concave up (or possibly straight) solution
                 (like y = x^2).
      curve = -1 for a concave down (or possibly straight) solution.
      curve =  0 to skip this constraint.

    For example, x = ArlsShape(A,b,0,1,-1) would demand a solution to
    Ax=b which is rising and concave down.

    Note: If all you want is nonnegativity, please use arlsnn(),
    which is more robust and faster.

    Note: We recommend that you use only one constraint initially.
    If that does not achieve the needed shape then add a second.
    Only use all three if absolutely necessary.
    The reason for this is that if the system is highly ill-conditioned
    AND highly constrained then it is possible for the solver to fail to
    satisfy all requirements with good accuracy.

    Parameters
    ----------
    A : (m, n) array_like
        Coefficient matrix
    b : (m) array_like
        Column of dependent variables.
    nonneg: as discussed above.
    slope: as discussed above.
    curve: as discussed above.

    Returns
    -------
    x : (n) array_like column vector, type float.

    Example
    -------
    Let A be a 6x6 Hilbert matrix. (As available in scipy.linalg.)
    let b be a right hand side vector such that the exact solution
    to Ax=b is
        x = [1., 2., 1.9, 3., 2.9, 4.]
    This solution is obviously NOT non-decreasing.

    You can force the solution to be non-decreasing by calling:
        x = arlshape(A,b,0,1,0)
    The result is:
        x = [1.0004, 1.9877, 1.9877, 2.7647, 3.1657, 3.8936]
    which is rising, except for an (acceptable) flat segment between
    the two values of 1.9877.
    The norm of the residual for this solution is, amazingly, only 1.0e-7.
    """
    A, b = checkAb(A, b)
    m, n = A.shape
    if np.count_nonzero(A) == 0 or np.count_nonzero(b) == 0:
        return np.zeros(n)
    if (
        nonneg < 0
        or nonneg > 1
        or slope < (-1)
        or slope > 1
        or curve < (-1)
        or curve > 1
    ):
        raise LinAlgError("Invalid constraint request in ArlsShape().")
    if nonneg == 0 and slope == 0 and curve == 0:
        return arls(A, b)

    # compute the number of rows in G
    m, n = A.shape
    ng = 0
    if nonneg != 0:
        ng += n
    if slope != 0:
        ng += max(0, n - 1)
    if curve != 0:
        ng += max(0, n - 2)

    G = np.zeros((ng, n))
    h = np.zeros(ng)
    ig = 0
    if nonneg > 0:
        for i in range(0, n):
            G[ig, i] = 1.0
            ig = ig + 1
    if slope > 0:
        for i in range(0, n - 1):
            G[ig, i] = -1.0
            G[ig, i + 1] = 1.0
            ig = ig + 1
    if slope < 0:
        for i in range(0, n - 1):
            G[ig, i] = 1.0
            G[ig, i + 1] = -1.0
            ig = ig + 1
    if curve > 0:
        for i in range(0, n - 2):
            G[ig, i] = 1.0
            G[ig, i + 1] = -2.0
            G[ig, i + 2] = 1.0
            ig = ig + 1
    if curve < 0:
        for i in range(0, n - 2):
            G[ig, i] = -1.0
            G[ig, i + 1] = 2.0
            G[ig, i + 2] = -1.0
            ig = ig + 1
    x = arlsgt(A, b, G, h)

    # assure nonnegativity regardless
    if nonneg > 0:
        for j in range(0, n):
            if x[j] < 0.0:
                x[j] = 0.0
    return x


def arlsall(A, b, E, f, G, h):
    """Solves the triple linear system of equations
        Ax = b  (least squares)
        Ex == f  (exact)
        Gx >= h ("greater than" inequality constraints)

    Each of the three systems an be underdetermined, square, or
    over-determined. However, generally E should have very few rows
    compared to A. Arguments b, f, and h must be single columns.

    Arlsgt() uses arlseq(), above, as the core solver, and iteratively selects
    rows of Gx>=h to addto Ex==f, choosing first whatever remaining equation
    in Gx>=h most violates its requirement.

    Note that "less than" equations can be included by negating
    both sides of the equation, thus turning it into a "greater than".

    If either A or b is all zeros then the solution will be all zeros.

    Parameters
    ----------
    A : (m, n)  array_like "Coefficient" matrix, type float.
    b : (m)     array_like column of dependent variables, type float.
    E : (me, n) array_like "Coefficient" matrix, type float.
    f : (me)    array_like column of dependent variables, type float.
    G : (mg, n) array_like "Coefficient" matrix, type float.
    b : (mg)    array_like column of dependent variables, type float.

    Note: None of the six arguments can be nul (that is, zero dimensional).
    If you do not want to use E and F, use arlsgt.
    If you do not want to use G and h, use arlseq.
    If you must use this routine anyway, but don't want E and f,
    then use E=np.zeros((1,n)), and f=np.zeros(1).
    The same applies to G and h.

    Returns
    -------
    x : (n) array_like column, type float.

    Raises
    ------
    LinAlgError
        If A is not 2-D.
        If A is empty.
        If A and b do not have the same row size.
        If E is not 2-D.
        If E is empty.
        If E and f do not have the same row size.
        If G is not 2-D.
        If G is empty.
        If G and h do not have the same row size.
        If A, E, and G do not have the same number of columns.
        If SCIPY's SVD() does not converge.

    Example
    -------
    Let A = [[1,1,1],
             [0,1,1],
             [1,0,1]]
    and b = [5.9, 5.0, 3.9]

    Then any least-squares solver would produce x = [0.9, 2., 3.]
    The residual for this solution is zero within roundoff.

    But if we happen to know that all the answers should be at least 1.0
    then we can add inequalites to insure that:
        x[0] >= 1
        x[1] >= 1
        x[2] >= 1
    This can be expressed in the matrix equation Gx>=h where
        G = [[1,0,0],
             [0,1,0],
             [0,0,1]]
        h =  [1,1,1]

    Then x = arlsgt(A,b,G,h) produces x = [1., 2.013, 2.872].
    The residual vector and its norm are then:
        res = [-0.015, -0.115, 0.028]
        norm(res) = 0.119
    So the price of adding this constraint is that the residual is no
    longer zero. This is normal behavior.

    Let's say that we have discovered that x[2] should be exactly 3.0.
    You could of course easily recast the problem without x[2] as a
    variable. But that is not so easy or desirable an approach for
    much larger systems.

    We can add that constraint that x[2]==3.0 by using the Ex==f system:
        E = [[0,0,1]]
        f = [3.]

    Calling arlsall(A,b,E,f,G,h) produces x = [1., 1.9, 3.0].
    The residual vector and its norm are then:
        res = [0.0, -0.1, 0.1]
        norm(res) = 0.141
    So again, as we add constraints to force the solution to what we know
    it must be, the residual will usually increase steadily from what the
    least-squares equations left alone will produce.

    But it would be a mistake to accept an answer that does not meet
    the facts that we know the solution must satisfy.
    """
    A, b = checkAb(A, b)
    m, n = A.shape
    if np.count_nonzero(A) == 0 or np.count_nonzero(b) == 0:
        return np.zeros(n)
    AA = A.copy()
    bb = b.copy()
    rnmax = np.max(np.linalg.norm(A, axis=1))
    neglect = rnmax * 0.000000001  # see Note 6. for arls()

    E, f = checkAb(E, f)
    me, ne = E.shape
    EE = E.copy()
    ff = f.copy()

    G, h = checkAb(G, h)
    mg, ng = G.shape
    GG = G.copy()
    hh = h.copy()

    if n != ne or n != ng:
        raise LinAlgError(
            "The matrices do not all have the same number of unknowns.")
    if me > m:
        raise LinAlgError(
            "Too many equality constraints for the size of A.")

    # get an initial solution... might be right
    x = arlseq(AA, bb, EE, ff)

    # while inequality constraints are not fully satisfied:
    while True:
        # assess state of inequalities
        p = get_worst(GG, hh, x)
        if p < 0:
            break

        # delete row from GGx=hh
        row = GG[p, :]
        rhsp = hh[p]
        GG = np.delete(GG, p, 0)
        hh = np.delete(hh, p, 0)

        # add row to Ex>=f
        me += 1
        EE = np.resize(EE, (me, ne))
        EE[me - 1, :] = row[:]
        ff = np.resize(ff, me)
        ff[me - 1] = rhsp

        # prep EE*x = ff, then decouple AAx=bb from EEx=ff
        EE, ff = prepeq(EE, ff, neglect)
        AA, bb = arlspj(AA, bb, EE, ff, neglect)

        # re-solve modified system
        x = arlseq(A, b, EE, ff)

    # Done
    return x
