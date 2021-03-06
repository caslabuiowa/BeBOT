"""

@author: ckielasjensen, Yiqing Gu
"""

from scipy.spatial import ConvexHull

def convexHullProperty(t1, t2, t3):
    
    result = []
    
    for cpts in [t1.cpts, t2.cpts]:
        hull = ConvexHull(cpts.T)
        for simplex in hull.simplices:
            result.append(cpts[0, simplex])
            result.append(cpts[1, simplex])
    
    return result
