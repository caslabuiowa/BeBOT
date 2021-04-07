"""

@author: ckielasjensen, Yiqing Gu

"""

import pytest
import numpy as np
import random as random
from polynomial.bernstein import Bernstein


# test for end points property
def endPointsProperty(t1, t2, t3):
    
    result = []
    
    # finding initial and final postions. Choosing a position to plot coordinates.
    for i, pt in enumerate(np.concatenate([t1.cpts[:, (0, -1)].T, t2.cpts[:, (0, -1)].T, t3.cpts[:, (0, -1)].T])):
        result.append(pt[0])
        result.append(pt[1]) 
    return result

def applyTests(figures, expectation):
    assert figures == expectation

#the following is time limit, second in bracket.
#@pytest.mark.timeout(0.001)  
#reference: https://pypi.org/project/pytest-timeout/

# test cases
def test_endPoint():
    
    # test case 1
    critical_points1 = np.array([[0, 0, 5, 5],
                      [0, 3.3, 6.7, 10]], dtype=float)
    critical_points2 = np.array([[5, 5, 10, 10],
                      [0, 3.3, 6.7, 10]], dtype=float)
    critical_points3 = np.array([[10, 10, 0, 0],
                      [0, 4.4, 5.6, 10]], dtype=float) 
    trajectory1 = Bernstein(critical_points1, t0=0, tf=93)
    trajectory2 = Bernstein(critical_points2, t0=0, tf=93)
    trajectory3 = Bernstein(critical_points3, t0=0, tf=124)
    expectedValue = [0.0, 0.0, 5.0, 10.0, 5.0, 0.0, 10.0, 10.0, 10.0, 0.0, 0.0, 10.0]
    endPointsResult = endPointsProperty(trajectory1, trajectory2, trajectory3)
    applyTests(endPointsResult, expectedValue)
   
    # test case 2 (random time interval)
    critical_points1 = np.array([[0, 0, 5, 5],
                      [0, 3.3, 6.7, 10]], dtype=float)
    critical_points2 = np.array([[5, 5, 10, 10],
                      [0, 3.3, 6.7, 10]], dtype=float)
    critical_points3 = np.array([[10, 10, 0, 0],
                      [0, 4.4, 5.6, 10]], dtype=float)
    t_01=random.randint(0, 30)
    t_f1=random.randint(40, 70)
    t_02=random.randint(0, 30)
    t_f2=random.randint(40, 70)
    t_03=random.randint(0, 30)
    t_f3=random.randint(40, 70)
    trajectory1 = Bernstein(critical_points1, t0=t_01, tf=t_f1)
    trajectory2 = Bernstein(critical_points2, t0=t_02, tf=t_f2)
    trajectory3 = Bernstein(critical_points3, t0=t_03, tf=t_f3)
    expectedValue = [0.0, 0.0, 5.0, 10.0, 5.0, 0.0, 10.0, 10.0, 10.0, 0.0, 0.0, 10.0]
    endPointsResult = endPointsProperty(trajectory1, trajectory2, trajectory3)
    applyTests(endPointsResult, expectedValue)
    print("Random time")
    print("t_01 " + str(t_01) + " t_f1 " + str(t_f1))
    print("t_02 " + str(t_02) + " t_f2 " + str(t_f2))
    print("t_03 " + str(t_03) + " t_f3 " + str(t_f3))
    
    #test case 3 (different critical points)
    critical_points1 = np.array([[0, 4, 6, 7],
                      [0, 5, 6, 8]], dtype=float)
    critical_points2 = np.array([[2, 5, 8, 9],
                      [2, 5, 8, 10]], dtype=float)
    critical_points3 = np.array([[10, 7, 4, 2],
                      [6, 4.4, 5.6, 10]], dtype=float) 
    trajectory1 = Bernstein(critical_points1, t0=0, tf=93)
    trajectory2 = Bernstein(critical_points2, t0=0, tf=93)
    trajectory3 = Bernstein(critical_points3, t0=0, tf=124)
    expectedValue = [0.0, 0.0, 7.0, 8.0, 2.0, 2.0, 9.0, 10.0, 10.0, 6.0, 2.0, 10.0]
    endPointsResult = endPointsProperty(trajectory1, trajectory2, trajectory3)
    applyTests(endPointsResult, expectedValue)

    #test case 4 (different critical points and time interval)
    critical_points1 = np.array([[0, 1, 2, 3],
                      [0, 3, 6, 8]], dtype=float)
    critical_points2 = np.array([[5, 6, 9, 9],
                      [0, 4, 8, 10]], dtype=float)
    critical_points3 = np.array([[10, 10, 7, 4],
                      [0, 4, 7, 9]], dtype=float)
    t_01=random.randint(0, 30)
    t_f1=random.randint(40, 70)
    t_02=random.randint(0, 30)
    t_f2=random.randint(40, 70)
    t_03=random.randint(0, 30)
    t_f3=random.randint(40, 70)
    trajectory1 = Bernstein(critical_points1, t0=t_01, tf=t_f1)
    trajectory2 = Bernstein(critical_points2, t0=t_02, tf=t_f2)
    trajectory3 = Bernstein(critical_points3, t0=t_03, tf=t_f3)
    expectedValue = [0.0, 0.0, 3.0, 8.0, 5.0, 0.0, 9.0, 10.0, 10.0, 0.0, 4.0, 9.0]
    endPointsResult = endPointsProperty(trajectory1, trajectory2, trajectory3)
    applyTests(endPointsResult, expectedValue)
    print("Random time")
    print("t_01 " + str(t_01) + " t_f1 " + str(t_f1))
    print("t_02 " + str(t_02) + " t_f2 " + str(t_f2))
    print("t_03 " + str(t_03) + " t_f3 " + str(t_f3))
  
if __name__ == '__main__':
    
    #run tests
    test_endPoint()
