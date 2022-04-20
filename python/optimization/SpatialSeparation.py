#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:43:52 2020

@author: ckielasjensen
"""

import numpy as np


def spatialSeparation(bpList):
    if len(bpList) < 2:
        return 0.0

    distVeh = []
    for i, traj in enumerate(bpList[:-1]):
        for traj2 in bpList[i+1:]:
            dv, _, _ = traj.minDist(traj2)
            distVeh.append(dv)

    return np.array(distVeh).flatten()


if __name__ == '__main__':
    pass
