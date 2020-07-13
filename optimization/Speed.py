#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 22:00:29 2020

@author: ckielasjensen
"""

import numpy as np

from constants import DEG_ELEV


def speed(bp, elev=DEG_ELEV):
    """
    Finds the speed of the Bernstein polynomial using degree elevation.

    Parameters
    ----------
    bp : Bernstein
        Bernstein polynomial whose speed will be determined using the L2 norm
        and degree elevation. The resulting BP will be elevated by DEG_ELEV
        defined in constants. Note that according to the product property, the
        norm squared will be of order 2N before being elevated.

    Returns
    -------
    speed : numpy.array
        1D numpy array of control points representing the BP of the L2 speed
        of the BP passed in.

    """
    speed = bp.diff().normSquare().elev(elev).cpts.flatten()

    return speed


if __name__ == '__main__':
    from polynomial.bernstein import Bernstein

    cpts = np.array([[0, 1, 2, 3, 4, 5],
                     [4, 5, 3, 6, 8, 7]], dtype=float)

    c = Bernstein(cpts)

    vel = speed(c)

    print(vel)
