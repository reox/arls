# ARLS: Automatically Regularized Linear System Solver

Arls is intended for use in solving linear systems, `Ax=b`, which other solvers 
may not be able to handle. Ill-conditioning is the usual reason for people to seek
alternative solvers such as `arls()`. In addition, `arls()` has a rich set of 
constraints which can be useful in solving both difficult and easy systems.

All these solvers apply to any shape of the matrix, `A`. That is, the 
system can have more rows than columns (over-determined), or the same number
of rows and columns (square), or fewer rows than columns (under-determined). And the matrix
`A` can be full rank, or have near zero singular values, or exactly zero singular 
values of any number.

Please see the in-code comments for details of calls and returns, including example applications for each routine.

Our primary solver is:

    x = arls(A, b)[0]

If you need to solve many systems with the same matrix, `A`, but different `b` vectors, use:

    x = arlsusv(A, b, U, S, Vt)[0]

See details in the code comments for how to get `U`, `S`, `Vt`.

If you need the solution to be constrained to be non-negative, use:

    x = arlsnn(A, b)
    
If you need to add constraint equations which must be satisfied exactly (such as
"the sum of the solution elements must be 100.0") then put those equation in a 
separate system, `Ex == f` and call:

    x = arlseq(A, b, E, f)

If you need to add inequality constraints, such as `x1 + x2 >= 1.0` then 
put those equations in a separate system, `Gx >= h` and call:

    x = arlsgt(A, b, G, h)
    
If you need both equality and inequality constraints, use

    x =arlsall(A, b, E, f, G, h) 

You can also constrain the shape of the solution with

    x = arlshape(A, b, nonneg, slope, curve)

You can constrain the solution to be 

1. nonnegative; and/or 
2. monotonically non-decreasing (i.e., rising) or monotonically non-increasing (i.e., falling); and/or 
3. concave upward (like `y = x*x`) or concave downward.

Examples: 

    x = arlshape(A, b, 1, 1, 0) will force x to be nonnegative and rising.
    x = arlshape(A, b, 0, -1, 1) will force x to be falling and concave up.

See details in the code comments.
    


