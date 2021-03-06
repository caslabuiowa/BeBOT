"""

@author: ckielasjensen, Yiqing Gu

"""

import numpy as np

# test for end points property
def endPointsProperty(t1, t2, t3):
    
    result = []
    
    # finding initial and final postions. Choosing a position to plot coordinates.
    for i, pt in enumerate(np.concatenate([t1.cpts[:, (0, -1)].T, t2.cpts[:, (0, -1)].T, t3.cpts[:, (0, -1)].T])):
        result.append(pt[0])
        result.append(pt[1])
    
    return result
