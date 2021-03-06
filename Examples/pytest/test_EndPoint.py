"""

@author: ckielasjensen, Yiqing Gu

"""

from EndPoint import endPointsProperty

import numpy as np
from polynomial.bernstein import Bernstein

def test_endPoint():
    
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
    
    expectedValue = [0.0, 0.0, 5.0, 10.0, 5.0, 0.0, 10.0, 10.0, 10.0, 0.0, 0.0, 10.0]
   
    assert endPointsProperty(trajectory1, trajectory2, trajectory3) == expectedValue

test_endPoint()
