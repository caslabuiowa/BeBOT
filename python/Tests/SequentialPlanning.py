#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:27:06 2020

@author: ckielasjensen
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import time

from polynomial.bernstein import Bernstein


def minDist(trajs):
    distances = []
    traj = trajs[0]
    for traj2 in trajs[1:]:
        dv = (traj-traj2).normSquare().elev(10).cpts.squeeze()
        distances.append(dv)

    return distances


def test(trajs):
    for i in range(len(trajs)-1):
        _ = minDist(trajs[:i+1])


if __name__ == '__main__':
    plt.close('all')
    df = pd.read_csv('../Examples/HawksLogo_1000pts.csv')

    finpts = np.concatenate([df.values, 100*np.ones((1000, 1))], 1)
    x = np.random.rand(1000, 1)*(finpts[:, 0].max() - finpts[:, 0].min()) + finpts[:, 0].min()
    y = np.random.rand(1000, 1)*(finpts[:, 1].max() - finpts[:, 1].min()) + finpts[:, 1].min()
    z = np.zeros((1000, 1))
    inipts = np.concatenate([x, y, z], 1)

    cpts = [np.linspace(pt[0], pt[1], 3).T for pt in zip(inipts, finpts)]
    trajs = [Bernstein(cpt) for cpt in cpts]

    ax = trajs[0].plot(showCpts=False)

    for traj in trajs[1:]: traj.plot(ax, showCpts=False)

    # now = time.time()
    # print('Full trajectories')
    # test(trajs)
    # print(f'Total time: {time.time()-now}')

    # now = time.time()
    # print('Partial Trajs')
    # for i in np.linspace(0, 900, 10).astype(int):
    #     test(trajs[i:i+100])
    # print(f'Total time: {time.time()-now}')
