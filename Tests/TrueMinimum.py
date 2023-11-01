#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 12:58:43 2020

@author: ckielasjensen
"""

import matplotlib.pyplot as plt
import numpy as np

from constants import DEG_ELEV
from polynomial.bernstein import Bernstein
from polynomial.rationalbernstein import RationalBernstein


def trueMin(bp, dim=0, globMin=np.inf, tol=1e-6, count=0, direction='a'):
    """Returns the minimum value of the Bernstein polynomialin a single
    dimension

    Finds the minimum value of the Bernstein polynomial. This is done by
    first checking the first and last control points since the first and
    last point lie on the curve. If the first or last control point is not
    the minimum value, the curve is split at the lowest control point. The
    new minimum value is then defined as the lowest control point of the
    two new curves. This continues until the difference between the new
    minimum and old minimum values is within the desired tolerance.

    :param dim: Which dimension to return the minimum of.
    :type dim: int
    :param tol: Tolerance of the minimum value.
    :type tol: float
    :param maxIter: Maximum number of iterations to search for the minimum.
    :type maxIter: int
    :return: Minimum value of the Bernstein polynomial. None if maximum
        iterations is met.
    :rtype: float or None
    """
    print(f'{count = :2} | {direction = }')
    count += 1
    ub = bp.cpts[dim, (0, -1)].min()
    if ub < globMin:
        globMin = ub

    minIdx = bp.cpts[dim, :].argmin()
    lb = bp.cpts[dim, minIdx]

    print(f'{lb = :4.3f} | {ub = :4.3f} | {minIdx = :3} | {globMin = :.3f}\n')

    # Prune if the global min is less than the lower bound
    if globMin < lb:
        return globMin

    # If we are within the desired tolerance, return
    if ub - lb < tol:
        return globMin

    # Otherwise split and continue
    else:
        tdiv = (minIdx/bp.deg)*(bp.tf - bp.t0) + bp.t0
        # tdiv = 0.5*(bp.tf - bp.t0) + bp.t0
        c1, c2 = bp.split(tdiv)
        c1min = trueMin(c1, dim=dim, globMin=globMin, tol=tol, count=count, direction='a')
        c2min = trueMin(c2, dim=dim, globMin=globMin, tol=tol, count=count, direction='b')

        return min(c1min, c2min)


def _angularRate(bp):
    bpdot = bp.diff()
    bpddot = bpdot.diff()

    xdot = bpdot.x
    ydot = bpdot.y
    xddot = bpddot.x
    yddot = bpddot.y

    num = yddot*xdot - xddot*ydot
    den = xdot*xdot + ydot*ydot

    cpts = num.elev(DEG_ELEV).cpts / den.elev(DEG_ELEV).cpts
    wgts = den.elev(DEG_ELEV).cpts
    # cpts = num.cpts / den.cpts
    # wgts = den.cpts

    return RationalBernstein(cpts, wgts, bp.t0, bp.tf)


if __name__ == '__main__':
    # sample control points
    cpts1 = np.array([[0, 2, 4, 6, 8, 10],
                      [5, 0, 2, 3, 10, 3]], dtype=float)

    # Bernstein polynomials
    c1 = Bernstein(cpts1, t0=10, tf=20)

    angrate1 = _angularRate(c1)

    c1min = trueMin(angrate1)
    print(f'Min Value: {c1min}')
    
    # random end points from 0 to 10 and random time interval
    critical_points1 = np.array([[random.randint(0, 10), random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)],
                      [random.randint(0, 10), random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)]], dtype=float)
    print(critical_points1)

    t_01=random.randint(0, 30)
    t_f1=random.randint(40, 60)
    trajectory1 = Bernstein(critical_points1, t0=t_01, tf=t_f1)
    
    c1min = trueMin(_angularRate(trajectory1))

    c1min = trueMin(angrate1)
    print(f'Min Value: {c1min}')
