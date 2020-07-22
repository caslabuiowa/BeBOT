#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:17:40 2020

@author: ckielasjensen
"""

import numpy as np

from constants import DEG_ELEV
from polynomial.bernstein import Bernstein
from polynomial.rationalbernstein import RationalBernstein


def angularRate(bp, elev=DEG_ELEV):
    bpdot = bp.diff()
    bpddot = bpdot.diff()

    xdot = bpdot.x
    ydot = bpdot.y
    xddot = bpddot.x
    yddot = bpddot.y

    num = yddot*xdot - xddot*ydot
    den = xdot*xdot + ydot*ydot

    cpts = num.elev(elev).cpts / den.elev(elev).cpts

    return cpts.squeeze()

    # wgts = den.elev(elev).cpts

    # return RationalBernstein(cpts, wgts, bp.t0, bp.tf)


if __name__ == '__main__':
    cpts = np.array([[0, 1, 2, 3, 4, 5],
                     [0, 1, 2, 8, 9, 3]], dtype=float)
    c = Bernstein(cpts)

    angRate = angularRate(c)

    print(angRate.cpts)
