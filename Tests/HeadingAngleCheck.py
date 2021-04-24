"""

@author: ckielasjensen, Yiqing Gu
"""
import numpy as np
import random as random
from polynomial.bernstein import Bernstein

def headingAngleCheck(c1, c2):

    c1dot = c1.diff()
    c2dot = c2.diff()
    
    c1tan = c1dot.y / c1dot.x
    c2tan = c2dot.y / c2dot.x
    
    # make data can be printed
    cpts1 = np.concatenate([[np.linspace(c1tan.t0, c1tan.tf, c1tan.deg+1)],
                            c1tan.cpts])
    
    cpts2 = np.concatenate([[np.linspace(c2tan.t0, c2tan.tf, c2tan.deg+1)],
                            c2tan.cpts])

    for i in range (0,len(cpts1[0]-1)):
        print("first")
        print("t = " + str(cpts1[0][i]))
        print("heading angle is " + str(cpts1[1][i]))
        
    for i in range (0,len(cpts2[0]-1)):
        print("second")
        print("t = " + str(cpts2[0][i]))
        print("heading angle is " + str(cpts2[1][i]))

if __name__ == '__main__':
    # sample Control points
    cpts1 = np.array([[0, 2, 4, 6, 8, 10],
                      [5, 0, 2, 3, 10, 3]], dtype=float)
    cpts2 = np.array([[1, 3, 6, 8, 10, 12],
                      [6, 9, 10, 11, 8, 8]], dtype=float)
    
    c1 = Bernstein(cpts1, t0=10, tf=20)
    c2 = Bernstein(cpts2, t0=10, tf=20)
    
    headingAngleCheck(c1, c2)
    
    # random end points from 0 to 10 and random time interval
    critical_points1 = np.array([[random.randint(0, 10), random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)],
                      [random.randint(0, 10), random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)]], dtype=float)
    critical_points2 = np.array([[random.randint(0, 10), random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)],
                      [random.randint(0, 10), random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)]], dtype=float)
    print(critical_points1)
    print(critical_points2)
    t_01=random.randint(0, 50)
    t_f1=random.randint(60, 100)
    t_02=random.randint(0, 50)
    t_f2=random.randint(60, 100)
    trajectory1 = Bernstein(critical_points1, t0=t_01, tf=t_f1)
    trajectory2 = Bernstein(critical_points2, t0=t_02, tf=t_f2)
    
    headingAngleCheck(trajectory1, trajectory2)