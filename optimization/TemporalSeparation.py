#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 18:52:29 2020

@author: ckielasjensen
"""

import numpy as np

from constants import DEG_ELEV


def temporalSeparation(bpList, elev=DEG_ELEV):
    """
    Find the temporal separation between BPs using degree elevation.

    Using the arithmetic and degree elevation properties of Bernstein
    polynomials, this function finds the Euclidean distance between each curve
    at every point in time. The convex hull property can be used to find a
    conservative estimate of the minimum distance.

    Note that the control points representing the BP of the Euclidean distance
    can be negative. Even though they are negative, the curve itself will still
    be positive.

    Parameters
    ----------
    bpList : list
        List of Bernstein objects whose minimum temporal separation will be
        determined.

    Returns
    -------
    numpy.array
        Degree elevated control points representing the Euclidean distance
        between the Bernstein polynomials..

    """
    if len(bpList) < 2:
        return np.inf

    distVeh = []
    for i, traj in enumerate(bpList[:-1]):
        for traj2 in bpList[i+1:]:
            dv = traj - traj2
            distVeh.append(dv.normSquare().elev(elev).cpts)

    return np.array(distVeh).flatten()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from polynomial.bernstein import Bernstein

    cpts1 = np.array([[0, 1, 2, 3, 4, 5],
                      [4, 7, 3, 9, 4, 5]], dtype=float)
    cpts2 = np.array([[9, 8, 9, 7, 9, 8],
                      [0, 1, 2, 3, 4, 5]], dtype=float)
    cpts3 = np.array([[2, 3, 4, 5, 6, 6],
                      [10, 11, 12, 13, 14, 2]], dtype=float)
    cpts4 = np.array([[0, 1, 2, 3, 4, 5],
                      [1, 1, 1, 1, 1, 1]], dtype=float)

    c1 = Bernstein(cpts1)
    c2 = Bernstein(cpts2)
    c3 = Bernstein(cpts3)
    c4 = Bernstein(cpts4)

    bpList = [c1, c2, c3, c4]

    # Testing whether temporal separation is working properly. Note that
    # negative distances do make sense since it corresponds to the control
    # points and not the actual curve. However, if a negative distance is found
    # and the curves do not intersect, the degree elevation should be increased
    distVeh = temporalSeparation(bpList)

    print(distVeh)
    if np.any(distVeh < 0):
        print('[!] Warning, negative distance!')

    plt.close('all')
    ax = c1.plot()
    c2.plot(ax)
    c3.plot(ax)
    c4.plot(ax)
