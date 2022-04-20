#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 18:57:40 2020

@author: ckielasjensen
"""

import numpy as np

from polynomial.bernstein import Bernstein


def _min(bp, dim=0, globMin=np.inf, tol=1e-6):
    ub = bp.cpts[dim, (0, -1)].min()
    if ub < globMin:
        globMin = ub

    minIdx = bp.cpts[dim, :].argmin()
    lb = bp.cpts[dim, minIdx]

    # Prune if the global min is less than the lower bound
    if globMin < lb:
        return globMin

    # If we are within the desired tolerance, return
    if ub - lb < tol:
        return globMin

    # Otherwise split and continue
    else:
        tdiv = (minIdx/bp.deg)*(bp.tf - bp.t0) + bp.t0
        c1, c2 = bp.split(tdiv)
        c1min = _min(c1, dim=dim, globMin=globMin, tol=tol)
        c2min = _min(c2, dim=dim, globMin=globMin, tol=tol)

        return min(c1min, c2min)


if __name__ == '__main__':
    cpts1 = np.array([[0, 1, 2, 3, 4, 5],
                      [33, 6, 7, 2, 4, 5]], dtype=float)

    c1 = Bernstein(cpts1)
    c1t = Bernstein(cpts1, t0=10, tf=20)

    print(f'c1 min[0]: {c1.min(0)}')
    print(f'c1 min[1]: {c1.min(1)}')
    print(f'c1t min[0]: {c1.min(0)}')
    print(f'c1t min[1]: {c1.min(1)}')
    print(f'c1 norm sqr min: {c1.normSquare().min()}')
