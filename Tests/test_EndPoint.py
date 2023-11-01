"""

@author: ckielasjensen, Yiqing Gu

"""

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

def test_endPoint1():
    
    # conpo = control points
    # test case 1
    conpo1 = np.array([[0, 0, 5, 5],
                      [0, 3.3, 6.7, 10]], dtype=float)
    conpo2 = np.array([[5, 5, 10, 10],
                      [0, 3.3, 6.7, 10]], dtype=float)
    conpo3 = np.array([[10, 10, 0, 0],
                      [0, 4.4, 5.6, 10]], dtype=float) 
    trajectory1 = Bernstein(conpo1, t0=0, tf=93)
    trajectory2 = Bernstein(conpo2, t0=0, tf=93)
    trajectory3 = Bernstein(conpo3, t0=0, tf=124)
    expectedValue = [0.0, 0.0, 5.0, 10.0, 5.0, 0.0, 10.0, 10.0, 10.0, 0.0, 0.0, 10.0]
    endPointsResult = endPointsProperty(trajectory1, trajectory2, trajectory3)
    applyTests(endPointsResult, expectedValue)

def test_endPoint2():
    
    # test case 2 (random time interval)
    conpo1 = np.array([[0, 0, 5, 5],
                      [0, 3.3, 6.7, 10]], dtype=float)
    conpo2 = np.array([[5, 5, 10, 10],
                      [0, 3.3, 6.7, 10]], dtype=float)
    conpo3 = np.array([[10, 10, 0, 0],
                      [0, 4.4, 5.6, 10]], dtype=float)
    t_01=random.randint(0, 30)
    t_f1=random.randint(40, 70)
    t_02=random.randint(0, 30)
    t_f2=random.randint(40, 70)
    t_03=random.randint(0, 30)
    t_f3=random.randint(40, 70)
    trajectory1 = Bernstein(conpo1, t0=t_01, tf=t_f1)
    trajectory2 = Bernstein(conpo2, t0=t_02, tf=t_f2)
    trajectory3 = Bernstein(conpo3, t0=t_03, tf=t_f3)
    expectedValue = [0.0, 0.0, 5.0, 10.0, 5.0, 0.0, 10.0, 10.0, 10.0, 0.0, 0.0, 10.0]
    endPointsResult = endPointsProperty(trajectory1, trajectory2, trajectory3)
    applyTests(endPointsResult, expectedValue)
    print("Random time")
    print("t_01 " + str(t_01) + " t_f1 " + str(t_f1))
    print("t_02 " + str(t_02) + " t_f2 " + str(t_f2))
    print("t_03 " + str(t_03) + " t_f3 " + str(t_f3))

def test_endPoint3():
    
    #test case 3 (different critical points)
    conpo1 = np.array([[0, 4, 6, 7],
                      [0, 5, 6, 8]], dtype=float)
    conpo2 = np.array([[2, 5, 8, 9],
                      [2, 5, 8, 10]], dtype=float)
    conpo3 = np.array([[10, 7, 4, 2],
                      [6, 4.4, 5.6, 10]], dtype=float) 
    trajectory1 = Bernstein(conpo1, t0=0, tf=93)
    trajectory2 = Bernstein(conpo2, t0=0, tf=93)
    trajectory3 = Bernstein(conpo3, t0=0, tf=124)
    expectedValue = [0.0, 0.0, 7.0, 8.0, 2.0, 2.0, 9.0, 10.0, 10.0, 6.0, 2.0, 10.0]
    endPointsResult = endPointsProperty(trajectory1, trajectory2, trajectory3)
    applyTests(endPointsResult, expectedValue)
    
def test_endPoint4():

    #test case 4 (different critical points and time interval)
    conpo1 = np.array([[0, 1, 2, 3],
                      [0, 3, 6, 8]], dtype=float)
    conpo2 = np.array([[5, 6, 9, 9],
                      [0, 4, 8, 10]], dtype=float)
    conpo3 = np.array([[10, 10, 7, 4],
                      [0, 4, 7, 9]], dtype=float)
    t_01=random.randint(0, 30)
    t_f1=random.randint(40, 70)
    t_02=random.randint(0, 30)
    t_f2=random.randint(40, 70)
    t_03=random.randint(0, 30)
    t_f3=random.randint(40, 70)
    trajectory1 = Bernstein(conpo1, t0=t_01, tf=t_f1)
    trajectory2 = Bernstein(conpo2, t0=t_02, tf=t_f2)
    trajectory3 = Bernstein(conpo3, t0=t_03, tf=t_f3)
    expectedValue = [0.0, 0.0, 3.0, 8.0, 5.0, 0.0, 9.0, 10.0, 10.0, 0.0, 4.0, 9.0]
    endPointsResult = endPointsProperty(trajectory1, trajectory2, trajectory3)
    applyTests(endPointsResult, expectedValue)
    print("Random time")
    print("t_01 " + str(t_01) + " t_f1 " + str(t_f1))
    print("t_02 " + str(t_02) + " t_f2 " + str(t_f2))
    print("t_03 " + str(t_03) + " t_f3 " + str(t_f3))
    
def test_endPoint5():

    #test case 5 (random end points from 0 to 5 and random time interval)
    conpo1 = np.array(np.random.randint(0,5,size=(2,5)), dtype=float)
    conpo2 = np.array(np.random.randint(0,5,size=(2,5)), dtype=float)
    conpo3 = np.array(np.random.randint(0,5,size=(2,5)), dtype=float) 
    print(conpo1)
    t_01=random.randint(0, 30)
    t_f1=random.randint(40, 70)
    t_02=random.randint(0, 30)
    t_f2=random.randint(40, 70)
    t_03=random.randint(0, 30)
    t_f3=random.randint(40, 70)
    trajectory1 = Bernstein(conpo1, t0=t_01, tf=t_f1)
    trajectory2 = Bernstein(conpo2, t0=t_02, tf=t_f2)
    trajectory3 = Bernstein(conpo3, t0=t_03, tf=t_f3)
    expectedValue = [conpo1[0][0], conpo1[1][0],
                     conpo1[0][int(conpo1.size/2-1)], conpo1[1][int(conpo1.size/2-1)], 
                     conpo2[0][0], conpo2[1][0], 
                     conpo2[0][int(conpo2.size/2-1)], conpo2[1][int(conpo2.size/2-1)], 
                     conpo3[0][0], conpo3[1][0], 
                     conpo3[0][int(conpo3.size/2-1)], conpo3[1][int(conpo3.size/2-1)]]
    endPointsResult = endPointsProperty(trajectory1, trajectory2, trajectory3)
    applyTests(endPointsResult, expectedValue)
    print("Random time")
    print("t_01 " + str(t_01) + " t_f1 " + str(t_f1))
    print("t_02 " + str(t_02) + " t_f2 " + str(t_f2))
    print("t_03 " + str(t_03) + " t_f3 " + str(t_f3))
    
def test_endPoint6():

    #test case 6 (random end points from 0 to 10 and random time interval)
    conpo1 = np.array(np.random.randint(0,10,size=(2,5)), dtype=float)
    conpo2 = np.array(np.random.randint(0,10,size=(2,5)), dtype=float)
    conpo3 = np.array(np.random.randint(0,10,size=(2,5)), dtype=float) 
    print(conpo1)
    t_01=random.randint(0, 30)
    t_f1=random.randint(40, 70)
    t_02=random.randint(0, 30)
    t_f2=random.randint(40, 70)
    t_03=random.randint(0, 30)
    t_f3=random.randint(40, 70)
    trajectory1 = Bernstein(conpo1, t0=t_01, tf=t_f1)
    trajectory2 = Bernstein(conpo2, t0=t_02, tf=t_f2)
    trajectory3 = Bernstein(conpo3, t0=t_03, tf=t_f3)
    expectedValue = [conpo1[0][0], conpo1[1][0],
                     conpo1[0][int(conpo1.size/2-1)], conpo1[1][int(conpo1.size/2-1)], 
                     conpo2[0][0], conpo2[1][0], 
                     conpo2[0][int(conpo2.size/2-1)], conpo2[1][int(conpo2.size/2-1)], 
                     conpo3[0][0], conpo3[1][0], 
                     conpo3[0][int(conpo3.size/2-1)], conpo3[1][int(conpo3.size/2-1)]]
    endPointsResult = endPointsProperty(trajectory1, trajectory2, trajectory3)
    applyTests(endPointsResult, expectedValue)
    print("Random time")
    print("t_01 " + str(t_01) + " t_f1 " + str(t_f1))
    print("t_02 " + str(t_02) + " t_f2 " + str(t_f2))
    print("t_03 " + str(t_03) + " t_f3 " + str(t_f3))
  
if __name__ == '__main__':
    
    #run tests
    test_endPoint1()
    test_endPoint2()
    test_endPoint3()
    test_endPoint4()
    test_endPoint5()
    test_endPoint6()
