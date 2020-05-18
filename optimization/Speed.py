#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 22:00:29 2020

@author: ckielasjensen
"""

import numpy as np

from constants import DEG_ELEV


def speed(bp):
    speed = bp.diff().normSquare().elev(DEG_ELEV).cpts.flatten()

    return speed


if __name__ == '__main__':
    from polynomial.bernstein import Bernstein

    cpts = np.array([[0, 1, 2, 3, 4, 5],
                     [4, 5, 3, 6, 8, 7]], dtype=float)

    c = Bernstein(cpts)

    vel = speed(c)

    print(vel)
