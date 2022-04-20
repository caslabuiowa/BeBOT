#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 17:51:00 2020

@author: ckielasjensen
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import timeit

from polynomial.bernstein import Bernstein


SAVE_FIG = False     # Set to True to save figures
FIG_FORMAT = 'svg'  # Used for the output format when saving figures


def convexHull(curve):
    """
    Plot the convex hull of the Bernstein polynomial.

    Parameters
    ----------
    curve : Bernstein
        Bernstein polynomial object whose convex hull shall be plotted.

    Returns
    -------
    None.

    """
    ax = curve.plot()
    ax.set_title('Convex Hull')

    cpts = curve.cpts
    hull = ConvexHull(cpts.T)
    for simplex in hull.simplices:
        ax.plot(cpts[0, simplex], cpts[1, simplex], 'k:')

    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig('Figures/ConvexHull.'+FIG_FORMAT, format=FIG_FORMAT)


def endPoints(curve):
    """
    Compare the end points of a Bernstein polynomial to its control points.

    Parameters
    ----------
    curve : Bernstein
        Bernstein polynomial object whose end points will be compared to its
        initial and final control points.

    Returns
    -------
    None.

    """
    cpts = curve.cpts
    p1 = curve(curve.t0).T
    p2 = curve(curve.tf).T
    print(f'The first control point is: {cpts[:, 0]}')
    print(f'The first value of the polynomial is: {p1}')
    print(f'The second control point is: {cpts[:, -1]}')
    print(f'The second value of the polynomial is: {p2}')


def derivatives(curve):
    """
    Compute the derivative of the Bernstein polynomial passed in.

    Parameters
    ----------
    curve : Bernstein
        Bernstein polynomial object whose derivative will be computed.

    Returns
    -------
    None.

    """
    cdot = curve.diff()
    ax = cdot.plot()
    ax.set_title('Derivative of Curve 1')

    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig('Figures/Derivative.'+FIG_FORMAT, format=FIG_FORMAT)


def integrals(curve):
    """
    Compute the integral of the Bernstein polynomial passed in.

    Parameters
    ----------
    curve : Bernstein
        Bernstein polynomial object whose integral will be computed.

    Returns
    -------
    None.

    """
    val = curve.integrate()
    print(f'Integral of Curve 1: {val}')


# TODO
def deCasteljau(curve, tdiv):
    """
    Split the curve at tdiv using the de Casteljau algorithm.

    Parameters
    ----------
    curve : Bernstein
        Bernstein polynomial to be split.
    tdiv : float
        Point at which to split the Bernstein polynomial.

    Returns
    -------
    None.

    """
    ax1 = curve.plot()
    ax1.set_title('Curve Before Being Split')

    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig('Figures/Curve1.'+FIG_FORMAT, format=FIG_FORMAT)

    c1, c2 = curve.split(tdiv)
    ax2 = c1.plot(color='b', label='Curve 1')
    c2.plot(ax2, color='r', label='Curve 2')
    ax2.set_title('Split Curve')
    plt.legend()

    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig('Figures/Curve1Split.'+FIG_FORMAT, format=FIG_FORMAT)


def degreeElevation(curve, elev):
    """
    Elevate the Bernstein polynomial and plot the original and elevated curve.

    Parameters
    ----------
    curve : Bernstein
        Bernstein polynomial object whose degree will be elevated
    elev : int
        Degree by which to elevate the Bernstein polynomial (> 0)

    Returns
    -------
    None.

    """
    ax = curve.plot()
    curve.elev(elev).plot(ax)
    ax.set_title('Elevated Curve')

    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig('Figures/ElevatedCurve.'+FIG_FORMAT, format=FIG_FORMAT)


def arithmetic(c1, c2):
    """
    Show examples of arithmetic operations between Bernstein polynomials.

    Parameters
    ----------
    c1 : Bernstein
        First polynomial for showing arithmetic examples.
    c2 : Bernstein
        Second polynomial for showing arithmetic examples.

    Returns
    -------
    None.

    """
    ax = c1.plot(label='Curve 1')
    c2.plot(ax, label='Curve 2')
    ax.set_title('Curves C1 and C2')
    ax.legend()

    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig('Figures/Curve1AndCurve2.'+FIG_FORMAT, format=FIG_FORMAT)

    summation = c1 + c2
    product = c1*c2

    summation.plot()
    plt.title('Sum of C1 and C2')

    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig('Figures/SumOfC1AndC2.'+FIG_FORMAT, format=FIG_FORMAT)

    product.plot()
    plt.title('Product of C1 and C2')

    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig('Figures/ProductOfC1AndC2.'+FIG_FORMAT, format=FIG_FORMAT)

    # TODO add division once rational BP is finished


def extrema(c1, c2):
    """
    Find the extrema of two Bernstein polynomials using the Evaluating Extrema algorithm.

    Note that the timeit module is used to determine an average runtime for one of the calls to min(). Since it is an
    iterative procedure, the time can vary due to the number of iterations.

    Parameters
    ----------
    c1 : Bernstein
        First Bernstein polynomial whose extrema will be computed.
    c2 : Bernstein
        Second Bernstein polynomial whose extrema will be computed.

    Returns
    -------
    None.

    """
    c1mind1 = c1.min()
    c1mind2 = c1.min(1)
    c2mind1 = c2.min(dim=0)
    c2mind2 = c2.min(dim=1, globMin=100000, tol=1e-9)
    c1maxd1 = c1.max()

    print(f'First dimension of c1 min: {c1mind1}.')
    print(f'Second dimension of c1 min: {c1mind2}.')
    print(f'First dimension of c2 min: {c2mind1}.')
    print(f'Second dimension of c2 min: {c2mind2}.')
    print(f'First dimension of c1 max: {c1maxd1}')

    niter = 1000
    trun = timeit.timeit(lambda: c1.min(dim=1, globMin=100000, tol=1e-9), number=niter)/niter
    print(f'Average time to run the call \"c1.min(dim=1, globMin=100000, tol=1e-9)\"'
          f'over {niter} iterations: {trun} s')


def spatialDistance(c1, c2):
    minDist, t1, t2 = c1.minDist(c2)
    print(f'Minimum distance between curves is {minDist}')
    ax = c1.plot()
    c2.plot(ax)

    ax.plot([c1(t1)[0], c2(t2)[0]], [c1(t1)[1], c2(t2)[1]], 'r.-', label='Minimum Distance Vector')
    ax.legend()

    niter, trun = timeit.Timer(lambda: c1.minDist(c2)).autorange()
    print(f'Average time to run "c1.minDist(c2)" over {niter} iterations: {trun/niter} s')


def collisionDetection(c1, c2, c3):
    """
    Determine whether the curves collide with each other.

    If there is not a collision, collCheck returns 0. If a collision is possible, it will return 1.

    Parameters
    ----------
    c1 : Bernstein
        DESCRIPTION.
    c2 : Bernstein
        DESCRIPTION.
    c3 : Bernstein
        DESCRIPTION.

    Returns
    -------
    None.

    """
    res = c1.collCheck(c2)
    print(f'Collision between c1 and c2: {res}')

    niter, trun = timeit.Timer(lambda: c1.collCheck(c2)).autorange()
    print(f'Runtime: {trun/niter} s')

    res = c1.collCheck(c3)
    print(f'Collision between c1 and c3: {res}')

    niter, trun = timeit.Timer(lambda: c1.collCheck(c3)).autorange()
    print(f'Runtime: {trun/niter} s')

    res = c2.collCheck(c3)
    print(f'Collision between c2 and c3: {res}')

    niter, trun = timeit.Timer(lambda: c2.collCheck(c3)).autorange()
    print(f'Runtime: {trun/niter} s')


if __name__ == '__main__':
    # Creates a Figures directory if it doesn't already exist
    if SAVE_FIG:
        import os
        if not os.path.isdir('Figures'):
            os.mkdir('Figures')

    # Define control points as numpy arrays. Be sure to set the dtype to float.
    cpts1 = np.array([[0, 1, 2, 3, 4, 5], [5, 0, 2, 5, 7, 5]], dtype=float)
    cpts2 = np.array([[0, 2, 4, 6, 8, 10], [3, 7, 3, 5, 8, 9]], dtype=float)
    cpts3 = cpts1 + np.array([0, 8])[:, np.newaxis]
    t0 = 10  # Initial time
    tf = 20  # Final time

    c1 = Bernstein(cpts1, t0=t0, tf=tf)
    c2 = Bernstein(cpts2, t0=t0, tf=tf)
    c3 = Bernstein(cpts3, t0=0, tf=1)

    # =========================================================================
    # Examples of Bernstein polynomial properties
    # =========================================================================
    plt.close('all')

    # Property 1 - Convex Hull
    print('Convex Hull')
    convexHull(c1)
    print('---')

    # Property 2 - End Point Values
    print('End Points')
    endPoints(c1)
    print('---')

    # Property 3 - Derivatives
    print('Derivatives')
    derivatives(c1)
    print('---')

    # Property 4 - Integrals
    print('Integrals')
    integrals(c1)
    print('---')

    # Property 5 - The de Casteljau Algorithm
    print('The de Casteljau Algorithm')
    deCasteljau(c1, 15)
    print('---')

    # Property 6 - Degree Elevation
    print('Degree Elevation')
    degreeElevation(c1, 10)
    print('---')

    # Property 7 - Arithmetic Operations
    print('Arithmetic Operations')
    arithmetic(c1, c2)
    print('---')

    # =========================================================================
    # Examples of Bernstein polynomial algorithms
    # =========================================================================

    # Algorithm 1 - Evaluating Extrema
    print('Evaluating Extrema')
    extrema(c1, c2)
    print('---')

    # Algorithm 2 - Minimum Spatial Distance
    print('Minimum Spatial Distance')
    spatialDistance(c2, c3)
    print('---')

    # Algorithm 3 - Collision Detection
    print('Collision Detection')
    collisionDetection(c1, c2, c3)
    print('---')
