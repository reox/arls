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

(MIT License)


PURPOSE

Arls is a package for automatically solving difficult linear systems of
equations. Arls also contains a rich set of constrained solvers which build
on the automatic solvers.

EXAMPLE

Any linear system solver (a.k.a. matrix solver) can handle easy problems.
For example, let
A = [[1, 2, 3],
     [1, 1, 1],
     [3, 2, 1]]
b = [6, 3, 6]

Then any linear equation solver will return
x = [1, 1, 1]

But what if
A = [[1, 2, 0],
     [1, 2, 0.01],
     [1.01, 2, 1]]
b = [ 3, 3.2, 3.9]

A quick look at these equations might lead one to think that the solution
will also be fairly close to
   x = [1, 1, 1]

However, a naive solver will actually produce a shocking answer:
   y = [-1910.0, 956.5, 20.]

But arls(A,b) will see the instability and automatically regularize the
system to return
   z = [0.624, 1.236, 0.608]    (rounded to three decimal places)

Of course, there is a cost for adjusting the system in this fashion,
as A*z calculates to    [3.095, 3.101, 3.709]
rather then the desired [3.000, 3.200, 3.900].
There will generally be trade-offs like this when "regularizing" such problems,
whether manually or automatically.

USAGE

(1) Suppose you have a system of equations to solve, Ax=b.
To use arls(), A does not have to be square. It can be any shape.
Then you can get our automatically regularized solution with the call
   x = arls(A, b)
Of course, if the system is well behaved, that is fine.
The answer will then be the same as any good solver would produce.

(2) If you want to solve many systems of equations with the same matrix A,
but different b vectors, first compute the Singular Value Decomposition
like this:
   U, S, Vt = np.linalg.svd(A, full_matrices=False)
Then get the solution for Ax=b for a particular b by calling:
   x = arlsusv(A, b, U, S, Vt)[0]

Note: Both arls() and arlsusv() can also return several diagnostic parameters.
Please see the extended comments for each solver in the code below.

(3) A common requirement is for the elements of x to all be non-negative.
To force this requirement on the solution, just call
   x = arlsnn(A, b)
(You still get all the benefits of automatic regularization.)

(4) Suppose you have special constraints you wish to obeyed exactly.
For example, such an equation might ask for the solution to add to 100.
Then form these "equality constraint" equations into a separate
linear system,
   Ex == f
and call
   x = arlseq(A, b, E, f)
There can be any number of such constraint equations.
But be careful that they make sense and do not conflict with each other
or arlseq() will have to delete some of the offending equations.
Please see the extended comments for arlseq() in the code below.

(5) Now, suppose you have "greater than" constraints you need the solution
to obey. For example, perhaps you know that none of the elements of x should
be less than 1.0. Then form these equations into a separate system,
   Gx >= h
and call
   x = arlsgt(A, b, G, h)
Of course, you can also have "less than" constraints, but you will need
to multiply each equation by -1 (on both sides) to convert it to a
"greater than" constraint.
Please see the extended comments for arlsgt() in the code below.

(6) The above two situations can be combined by calling
   x = arlsall(A, b, E, f, G, h)
Please see the extended comments for arlsall() in the code below.

(7) Finally, the following routine allows you to require the solution
to satisfy certain shape characteristics.
Specifically, you can specify whether the solution should be:
    (1) nonnegative; and/or
    (2) monotonically rising (a.k.a. non-decreasing) or
        monotonically falling (a.k.a. non-increasing); and/or
    (3) concave upward (like y = x*x) or concave downward.
For example, if the solution should be nonnegative and falling, but you have
no clear need for any particular concavity, then just call
   x = arlshape(A, b, 1, -1, 0)
Please see the extended comments for arlshape(A, b, nonneg, slope, curve)
below for details.
Note: if you only need the solution to be nonnegative, arlsnn() is
a better choice.
"""
from ._arls import (
        arlsusv,
        arls,
        arlsnn,
        arlseq,
        arlsgt,
        arlsall,
        arlshape,
        )

__version__ = "1.0.2"
