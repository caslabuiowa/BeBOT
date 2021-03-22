"""
Created on Sat Mar  6 13:42:53 2021

@author: ckielasjensen, Yiqing Gu
"""

import pytest
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay

import random as random
import numpy as np
from polynomial.bernstein import Bernstein

# convex hull algorithm
def convexHullProperty(t1, t2, t3):
    
    result = []
    
    for cpts in [t1.cpts, t2.cpts, t3.cpts]:
        hull = ConvexHull(cpts.T)
        for simplex in hull.simplices:
            result.append(cpts[0, simplex])
            result.append(cpts[1, simplex])
    return result

# checking whether point(s) is/are in convex hull
# reference: https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
def in_hull(points, hull):
    
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
        
        result = hull.find_simplex(points)>=0
        print(result)
        
        assert result.all() == np.full((len(result),2), True, dtype = bool).all()

#The following is time limit, second in bracket.
#@pytest.mark.timeout(0.001)  
# test cases
def test_convexHull():
    
    # test case 1 (fixed points)
    critical_points1 = np.array([[0, 0, 5, 5],
                      [0, 3.3, 6.7, 10]], dtype=float)
    critical_points2 = np.array([[5, 5, 10, 10],
                      [0, 3.3, 6.7, 10]], dtype=float)
    critical_points3 = np.array([[10, 10, 0, 0],
                      [0, 4.4, 5.6, 10]], dtype=float)
    trajectory1 = Bernstein(critical_points1, t0=0, tf=93)
    trajectory2 = Bernstein(critical_points2, t0=0, tf=93)
    trajectory3 = Bernstein(critical_points3, t0=0, tf=124)
    testCase = convexHullProperty(trajectory1, trajectory2, trajectory3)
    testPoints = np.array([(1,1), (2, 2)])
    
    in_hull(testPoints, testCase)

    #test case 2 (random points for checking, might cause error)
    critical_points1 = np.array([[0, 0, 5, 5],
                      [0, 3.3, 6.7, 10]], dtype=float)
    critical_points2 = np.array([[5, 5, 10, 10],
                      [0, 3.3, 6.7, 10]], dtype=float)
    critical_points3 = np.array([[10, 10, 0, 0],
                      [0, 4.4, 5.6, 10]], dtype=float)
    trajectory1 = Bernstein(critical_points1, t0=0, tf=93)
    trajectory2 = Bernstein(critical_points2, t0=0, tf=93)
    trajectory3 = Bernstein(critical_points3, t0=0, tf=124)
    testCase = convexHullProperty(trajectory1, trajectory2, trajectory3)
    testPoints = np.random.randint(0, 10, size=(10, 2))
    print("Test points")
    print(testPoints)
    
    in_hull(testPoints, testCase)
    
    #test case 3 (random points and random time interval, might cause error)
    critical_points1 = np.array([[0, 0, 5, 5],
                      [0, 3.3, 6.7, 10]], dtype=float)
    critical_points2 = np.array([[5, 5, 10, 10],
                      [0, 3.3, 6.7, 10]], dtype=float)
    critical_points3 = np.array([[10, 10, 0, 0],
                      [0, 4.4, 5.6, 10]], dtype=float)
    t_01=random.randint(0, 50)
    t_f1=random.randint(55, 100)
    t_02=random.randint(0, 50)
    t_f2=random.randint(55, 100)
    t_03=random.randint(0, 50)
    t_f3=random.randint(55, 100)
    trajectory1 = Bernstein(critical_points1, t0=t_01, tf=t_f1)
    trajectory2 = Bernstein(critical_points2, t0=t_02, tf=t_f2)
    trajectory3 = Bernstein(critical_points3, t0=t_03, tf=t_f3)
    testCase = convexHullProperty(trajectory1, trajectory2, trajectory3)
    testPoints = np.random.randint(0, 10, size=(20, 2))
    print("Random time")
    print("t_01 " + str(t_01) + " t_f1 " + str(t_f1))
    print("t_02 " + str(t_02) + " t_f2 " + str(t_f2))
    print("t_03 " + str(t_03) + " t_f3 " + str(t_f3))
    print("Test points")
    print(testPoints)
    
    in_hull(testPoints, testCase)


if __name__ == '__main__':
    
    #run tests
    test_convexHull()

