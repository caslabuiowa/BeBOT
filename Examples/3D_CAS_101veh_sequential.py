#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:39:08 2019

@author: ckielasjensen
"""

import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import pandas as pd
import scipy.optimize as sop
import time

import bezier as bez

DEG_ELEV = 30

def nonlcon(x, vidx, traj, nveh, params):
    """Nonlinear constraints of the optimization problem

    :param x: X vector over which the optimization is happening
    :type x: np.array
    :param vidx: Index of the current vehicle being planned
    :type vidx: int
    :param traj: 2D numpy array of existing vehicle trajectories, can be empty
    :type traj: np.array
    :param nveh: Current number of vehicles, not the total number of vehicles
        found within params
    :type nveh: int
    :param params: Parameter object containing values for the problem
    :type params: Parameters
    """
    c = []
    y = reshape(x, traj, params.ndim,
                params.inipts[vidx, :], params.finalpts[vidx, :])
    c.append(temporalSeparationConstraints(y, nveh, params.ndim,
                                           params.dsafe))

    return np.concatenate(c)


def temporalSeparationConstraints(y, nveh, ndim, maxSep):
    """Calculate the separation between vehicles.

    The maximum separation is found by degree elevation.

    :param x: Optimization vector
    :type x: np.array
    :param nveh: Number of vehicles
    :type nveh: int
    :param dim: Dimension of the vehicles
    :type dim: int
    :param maxSep: Maximum separation between vehicles
    :type maxSep: float
    :return: Minimum temporal separation between the vehicles
    """
    if nveh > 1:
        distVeh = np.empty(nveh-1)
        vehTraj = bez.Bezier(y[0:ndim, :])

        for i in range(1, nveh):
            tempTraj = bez.Bezier(y[i*ndim:(i+1)*ndim, :])
            dv = vehTraj - tempTraj
            distVeh[i-1] = dv.normSquare().elev(DEG_ELEV).cpts.min()

        return (distVeh - maxSep**2)

    else:
        return np.atleast_1d(0.0)


def cost(x, vidx, params):
    """Cost function of the optimization problem
    """
    # return 0
    y = reshape(x, np.atleast_2d([]), params.ndim, params.inipts[vidx, :],
                params.finalpts[vidx, :])
    return np.linalg.norm(np.diff(y))


@njit(cache=True)
def reshape(x, traj, ndim, inipt, finalpt):
    """Reshapes the X vector being optimized into a matrix for computation

    The X vector is a 1D vector of the following shape:
        [x0,1 ... x0,(n-1) y0,1 ... z0,(n-1) x2,1 ... znveh,(n-1)]
    where n is the degree of the Bernstein polynomial and nveh is the number of
    vehicles. In other words, the vector is the control points of the 0th
    vehicle of the first (X) dimension followed by the control points of the
    second (Y) dimension followed by the optional third (Z) dimension. Then the
    next vehicle follows in the same manner until all vehicles have been
    accounted for. Note that the initial and final points are NOT included in
    the X vector. This is because the problem can be formulated in such a way
    that we do not need to include the initial and final points in the equality
    constraints, thus slightly simplifying the problem.

    :param x: X vector over which the optimization is happening
    :type x: np.array
    :param traj: 2D numpy array of existing vehicle trajectories, can be empty
    :type traj: np.array
    :param ndim: Number of dimensions (most likely 2 or 3)
    :type ndim: int
    :param inipt: Initial point of the vehicle
    :type inipt: np.array
    :param finalpt: Final points of the current vehicle
    :type finalpt: np.array
    """
    y = np.concatenate((np.atleast_2d(inipt).reshape((-1, 1)),
                        x.reshape((ndim, -1)),
                        np.atleast_2d(finalpt).reshape((-1, 1))), axis=1)
    if traj.size > 0:
        return np.concatenate((traj, y))
    else:
        return y


def initguess(vidx, params):
    """Generates an initial guess of straight lines for all the vehicles

    :param vidx: Index of the current vehicle being planned
    :type vidx: int
    :param params: Parameter object containing values for the problem
    :type params: Parameters
    :return: Initial guess to be passed into the minimize function
    :rtype: np.array
    """
    x0 = np.empty((params.deg-1)*params.ndim)
    for d in range(params.ndim):
        idx = d*(params.deg-1)
        x0[idx:idx+params.deg-1] = np.linspace(params.inipts[vidx, d],
                                               params.finalpts[vidx, d],
                                               params.deg+1)[1:-1]

    return x0


class Parameters:
    """
    """
    def __init__(self, nveh, ndim, deg, volume, dsafe):
        self.nveh = nveh
        self.ndim = ndim
        self.deg = deg
        self.volume = volume
        self.dsafe = dsafe

#        # Generate initial and final points randomly within the control volume
#        self.inipts = volume*np.concatenate([
#                np.random.rand(nveh, ndim-1), np.zeros((nveh, 1))], axis=1)

        # Initial points
        np.random.seed(3)
        lines = np.atleast_2d(np.arange(1, 25, 2))
        x = np.repeat(lines, 12, axis=1).T
        y = np.repeat(lines, 12, axis=0).reshape((-1, 1))
        pts = np.concatenate((x, y), axis=1)
        idxs = np.random.choice(np.arange(len(pts)), size=nveh, replace=False)

        self.inipts = np.concatenate([pts[idxs, :], np.zeros((nveh, 1))],
                                     axis=1)

        # Random final points
#        self.finalpts = volume*np.concatenate([
#                np.random.rand(nveh, ndim-1), np.ones((nveh, 1))], axis=1)

        # Hawkeye logo final points
        df = pd.read_csv('CAS_PixelImage.csv')
        self.finalpts = np.concatenate((df.values, volume*np.ones((nveh, 1))),
                                       axis=1)
        self.finalpts = np.ascontiguousarray(self.finalpts)


if __name__ == '__main__':
    NVEH = 101      # Number of vehicles
    NDIM = 3        # Number of dimensions
    DEG = 3         # Order of the Bernstein polynomial approximation
    VOLUME = 100     # Length of an edge of the cubic volume being used
    DSAFE = 1       # Minimum safety distance between vehicles
    np.random.seed(3)
    # Locations of spherical obstacles
    OBS_LOC = np.array([
            (50, 50, 50)])

    params = Parameters(NVEH, NDIM, DEG, VOLUME, DSAFE)

    tstart = time.time()
    traj = np.atleast_2d([])
    for i in range(NVEH):
        print(f'Current Vehicle: {i}')
        x0 = initguess(i, params)

        def fun(x): return cost(x, i, params)
        cons = [{'type': 'ineq',
                 'fun': lambda x: nonlcon(x, i, traj, i+1, params)}]
        res = sop.minimize(fun, x0,
                           constraints=cons,
                           method='SLSQP',
                           options={'maxiter': 250,
                                    'disp': True,
                                    'iprint': 0})
        traj = reshape(res.x, traj, params.ndim, params.inipts[i, :],
                       params.finalpts[i, :])
    tend = time.time()
    print('===============================================================')
    print(f'Total computation time for {NVEH} vehicles: {tend-tstart}')
    print('===============================================================')

    temp = bez.Bezier(traj[0:NDIM, :])
    vehList = [temp]
    ax = temp.plot(showCpts=False)
    plt.plot([temp.cpts[0, -1]], [temp.cpts[1, -1]], [temp.cpts[2, -1]],
             'k.', markersize=15, zorder=-1)
    for i in range(NVEH):
        temp = bez.Bezier(traj[i*NDIM:(i+1)*NDIM, :])
        vehList.append(temp)
        temp.plot(ax, showCpts=False)
        plt.plot([temp.cpts[0, -1]], [temp.cpts[1, -1]], [temp.cpts[2, -1]],
                 'k.', markersize=40, zorder=10)
