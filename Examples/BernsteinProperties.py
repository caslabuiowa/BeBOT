#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 17:51:00 2020

@author: ckielasjensen
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d

from polynomial.bernstein import Bernstein


def convexHull(curve):
    """
    Plots the convex hull of the Bernstein polynomial that is passed in

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


def endPoints(curve):
    """
    Compares the end points of a Bernstein polynomial to its control points.

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
    print('---')
    print(f'The first control point is: {cpts[:, 0]}')
    print(f'The first value of the polynomial is: {p1}')
    print(f'The second control point is: {cpts[:, -1]}')
    print(f'The second value of the polynomial is: {p2}')
    print('---')


def derivatives(curve):
    """
    Computes the derivative of the Bernstein polynomial passed in.

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


def integrals(curve):
    """
    Computes the integral of the Bernstein polynomial passed in.

    Parameters
    ----------
    curve : Bernstein
        Bernstein polynomial object whose integral will be computed.

    Returns
    -------
    None.

    """
    val = curve.integrate()
    print('---')
    print(f'Integral of Curve 1: {val}')
    print('---')

# TODO
def deCasteljau(curve, tdiv):
    ax = curve.plot()
    c1, c2 = curve.split(tdiv)

    c1.plot(ax)
    c2.plot(ax)


def degreeElevation(curve, elev):
    """
    Elevates the Bernstein polynomial and plots the original and elevated curve

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


def arithmetic(c1, c2):
    """
    Shows examples of arithmetic operations between Bernstein polynomials

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
    ax = c1.plot()
    c2.plot(ax)

    summation = c1 + c2
    product = c1*c2

    summation.plot()
    plt.title('Sum of C1 and C2')

    product.plot()
    plt.title('Product of C1 and C2')

    # TODO add division once rational BP is finished


if __name__ == '__main__':
    # Define control points as numpy arrays. Be sure to set the dtype to float.
    cpts1 = np.array([[0, 1, 2, 3, 4, 5], [5, 0, 2, 5, 7, 5]], dtype=float)
    cpts2 = np.array([[0, 2, 4, 6, 8, 10], [3, 7, 3, 5, 8, 9]], dtype=float)
    t0 = 10  # Initial time
    tf = 20  # Final time

    c1 = Bernstein(cpts1, t0=t0, tf=tf)
    c2 = Bernstein(cpts2, t0=t0, tf=tf)

    # =========================================================================
    # Examples of Bernstein polynomial properties
    # =========================================================================
    plt.close('all')

    # Property 1 - Convex Hull
    convexHull(c1)

    # Property 2 - End Point Values
    endPoints(c1)

    # Property 3 - Derivatives
    derivatives(c1)

    # Property 4 - Integrals
    integrals(c1)

    # Property 5 - The de Casteljau Algorithm
    deCasteljau(c1, 0.5)

    # Property 6 - Degree Elevation
    degreeElevation(c1, 10)

    # Property 7 - Arithmetic Operations
    arithmetic(c1, c2)
