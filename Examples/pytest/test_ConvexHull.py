"""
Created on Sat Mar  6 13:42:53 2021

@author: ckielasjensen, Yiqing Gu
"""

from ConvexHull import convexHullProperty

import numpy as np
from polynomial.bernstein import Bernstein

def test_convexHull():
    
    critical_points1 = np.array([[0, 0, 5, 5],
                      [0, 3.3, 6.7, 10]], dtype=float)
    critical_points2 = np.array([[5, 5, 10, 10],
                      [0, 3.3, 6.7, 10]], dtype=float)
    critical_points3 = np.array([[10, 10, 0, 0],
                      [0, 4.4, 5.6, 10]], dtype=float)
    
    # Bernstein polynomials with specific time interval 
    trajectory1 = Bernstein(critical_points1, t0=0, tf=93)
    trajectory2 = Bernstein(critical_points2, t0=0, tf=93)
    trajectory3 = Bernstein(critical_points3, t0=0, tf=124)
    
    testObject = convexHullProperty(trajectory1, trajectory2, trajectory3)
    
    for i in range(0, len(testObject)):
        assert testObject[i].all() == testObject[i].all()

test_convexHull()

