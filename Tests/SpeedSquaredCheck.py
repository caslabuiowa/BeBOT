"""

@author: ckielasjensen, Yiqing Gu
"""
import numpy as np
import random as random
from polynomial.bernstein import Bernstein

def speedSquaredCheck(c):

    c1speed = c1.diff().normSquare()
    
    cpts = np.concatenate([[np.linspace(c1speed.t0, c1speed.tf, c1speed.deg+1)],
                            c1speed.cpts])
    
    #print all of speed data corresponding to its time
    for i in range (0,len(cpts[0]-1)):
        print("t = " + str(cpts[0][i]))
        print("speed is " + str(cpts[1][i]))

if __name__ == '__main__':
    # sample control points
    cpts1 = np.array([[0, 2, 4, 6, 8, 10],
                      [5, 0, 2, 3, 10, 3]], dtype=float)
    
    c1 = Bernstein(cpts1, t0=10, tf=20)
    
    speedSquaredCheck(c1)
    
    # conpo = control points
    # random end points from 0 to 10 and random time interval
    conpo = np.array(np.random.randint(0,10,size=(2,4)), dtype=float)
    print(conpo)
    t_01=random.randint(0, 30)
    t_f1=random.randint(40, 60)
    trajectory1 = Bernstein(conpo, t0=t_01, tf=t_f1)
    
    speedSquaredCheck(trajectory1)
    
