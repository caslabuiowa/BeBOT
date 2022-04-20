#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:33:25 2020

@author: ckielasjensen
"""

import numpy as np
import subprocess as sb


def minDist(poly1, poly2):

    with open('./ext/openGJK/example1_c/userP.dat', 'w') as f:
        npts = poly1.shape[1]
        f.write(str(npts))
        f.write('\n')
        np.savetxt(f, poly1.T)

    with open('./ext/openGJK/example1_c/userQ.dat', 'w') as f:
        npts = poly2.shape[1]
        f.write(str(npts))
        f.write('\n')
        np.savetxt(f, poly2.T)

    proc = sb.run('./gjk',
                  cwd='ext/openGJK/example1_c/',
                  capture_output=True)
    return float(proc.stdout)



if __name__ == '__main__':
    p1 = np.array([[4, 5, 2, 6],
                   [3, 6, 2, 7],
                   [0, 0, 0, 0]], dtype=float)

    p2 = np.array([[5, 7, 3, 8, 9],
                   [4, 7, 0, 2, 4],
                   [5, 5, 5, 5, 5]], dtype=float)

    p3 = p1.copy()
    p3[2, :] += 10

    print(minDist(p1, p2))
    print(minDist(p1, p3))