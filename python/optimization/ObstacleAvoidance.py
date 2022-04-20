#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:50:50 2020

@author: ckielasjensen
"""
import numpy as np

from constants import DEG_ELEV
from polynomial.bernstein import Bernstein


def obstacleAvoidance(trajs, obstacles, elev=DEG_ELEV):
    distObs = []
    for traj in trajs:
        for obs in obstacles:
            do = traj - _obs2bp(obs, traj.deg, traj.t0, traj.tf)
            if elev is np.inf:
                distObs.append(do.normSquare().min())
            else:
                distObs.append(do.normSquare().elev(elev).cpts)

    return np.array(distObs).flatten()


def _obs2bp(obs, deg, t0, tf):
    cpts = np.empty((2, deg+1), dtype=float)
    cpts[0, :] = obs[0]
    cpts[1, :] = obs[1]

    return Bernstein(cpts, t0=t0, tf=tf)
