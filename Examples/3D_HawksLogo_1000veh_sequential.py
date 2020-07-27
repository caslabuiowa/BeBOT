#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 13:33:45 2020

@author: ckielasjensen
"""

import os
import pickle

import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
import time

from constants import DEG_ELEV
from polynomial.bernstein import Bernstein


def nonlcon(x, vidx, traj, nveh, params):
    """Nonlinear constraints of the optimization problem.

    :param x: X vector over which the optimization is happening
    :type x: np.array
    :param vidx: Index of the current vehicle being planned
    :type vidx: int
    :param traj: 2D array of existing vehicle trajectories, can be empty
    :type traj: np.ndarray
    :param nveh: Current number of vehicles, not the total number of vehicles
        found within params
    :type nveh: int
    :param params: Parameter object containing values for the problem
    :type params: Parameters
    :return: Nonlinear constraints
    :rtype: np.ndarray
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
    :rtype: np.ndarray
    """
    if nveh > 1:
        # distVeh = np.empty(nveh-1)
        distVeh = []
        # vehTraj = bez.Bezier(y[0:ndim, :])
        vehTraj = Bernstein(y[0:ndim, :])

        for i in range(1, nveh):
            # tempTraj = bez.Bezier(y[i*ndim:(i+1)*ndim, :])
            tempTraj = Bernstein(y[i*ndim:(i+1)*ndim, :])
            dv = vehTraj - tempTraj
            # distVeh[i-1] = dv.normSquare().elev(DEG_ELEV).cpts.min()
            distVeh.append(dv.normSquare().elev(DEG_ELEV).cpts.squeeze())

        return (np.concatenate(distVeh) - maxSep**2)

    else:
        return np.atleast_1d(0.0)


def cost(x, vidx, params):
    """Cost function of the optimization problem."""
    # y = reshape(x, np.atleast_2d([]), params.ndim, params.inipts[vidx, :], params.finalpts[vidx, :])
    # return np.linalg.norm(np.diff(y))
    # accel = Bernstein(y, t0=params.t0, tf=params.tf).diff().diff().normSquare()

    # return accel.cpts.sum()
    if vidx:
        y = reshape(x, params.traj, 3, params.inipts[vidx, :], params.finalpts[vidx, :])
        nveh = y.shape[0] // 3
        dist = 0.0
        traj = Bernstein(y[0:3, :])

        for i in range(1, nveh):
            traj2 = Bernstein(y[i*3:(i+1)*3, :])
            dv = traj - traj2
            # dist += dv.normSquare().cpts.sum()
            temp = dv.normSquare().min()
            if temp < params.dsafe:
                dist -= np.inf
            dist += temp

        return -dist

    else:
        return 0.0


@njit(cache=True)
def reshape(x, traj, ndim, inipt, finalpt):
    """Reshape the X vector being optimized into a matrix for computation.

    The X vector is a 1D vector of the following shape:
        [x_{0,1} ... x_{0,(n-1)} y_{0,1} ... z_{0,(n-1)} x_{2,1} ...
         z_{nveh,(n-1)}]
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
    :type traj: np.ndarray
    :param ndim: Number of dimensions (most likely 2 or 3)
    :type ndim: int
    :param inipt: Initial point of the vehicle
    :type inipt: np.array
    :param finalpt: Final points of the current vehicle
    :type finalpt: np.ndarray
    :return: Vector reshaped into usable matrix for further computations
    :rtype: np.ndarray
    """
    y = np.concatenate((np.atleast_2d(inipt).reshape((-1, 1)),
                        x.reshape((ndim, -1)),
                        np.atleast_2d(finalpt).reshape((-1, 1))), axis=1)
    if traj.size > 0:
        return np.concatenate((traj, y))
    else:
        return y


def initguess(vidx, params):
    """Generate an initial guess of straight lines for all the vehicles.

    :param vidx: Index of the current vehicle being planned
    :type vidx: int
    :param params: Parameter object containing values for the problem
    :type params: Parameters
    :return: Initial guess to be passed into the minimize function
    :rtype: np.ndarray
    """
    x0 = np.empty((params.deg-1)*params.ndim)
    for d in range(params.ndim):
        idx = d*(params.deg-1)
        x0[idx:idx+params.deg-1] = np.linspace(params.inipts[vidx, d],
                                               params.finalpts[vidx, d],
                                               params.deg+1)[1:-1]

    return x0


def createFinpts(fname):
    """
    Create final points for the flight trajectories.

    Parameters
    ----------
    fname : string
        File name containing the final points.

    Returns
    -------
    finpts : numpy array
        Final points of the flight trajectories where the columns represent x, y, and z and the rows represent each
        vehicle. All the Z values are set to an altitude of 100m.

    """
    df = pd.read_csv(fname)
    finpts = np.ascontiguousarray(df.values.copy())
    finpts = np.concatenate([finpts, np.ones((1000, 1))*100], 1)
    return finpts


def createInipts(finalPts):
    """
    Create initial positions for 1000 aerial vehicle example. This will NOT work with any other number of points.

    Parameters
    ----------
    finalPts : np.array
        1000x3 numpy array of the final points where each row holds the final (x, y, z) position of a vehicle.

    Returns
    -------
    inipts : np.array
        1000x3 numpy array of the initial points in the same format as finalPts.

    """
    inipts = np.empty((1000, 3))
    inipts[:, 2] = np.zeros(1000)

    fminx = finalPts[:, 0].min()
    fmaxx = finalPts[:, 0].max()
    fminy = finalPts[:, 1].min()
    fmaxy = finalPts[:, 1].max()

    for i in range(5):
        for j in range(2):
            # The +1 and -1 here are so that we don't have overlapping initial points
            x = np.linspace(j*fmaxx/2+1, (j+1)*fmaxx/2-1, 20) + fminx
            y = np.linspace(i*fmaxy/5+1, (i+1)*fmaxy/5-1, 5) + fminy

            inipts[100*j+i*200:100*j+i*200+100, :2] = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    return inipts


def createBounds(params):
    """
    Create bounds for the optimization problem.

    Parameters
    ----------
    params : Parameters
        Problem parameters.

    Returns
    -------
    Bounds
        Bounds of the optimization problem.

    """
    # The -1 and +1 are so that we can have a feasible x0, this is due to how the final points were generated
    xlb = [params.inipts[:, 0].min()-1]*(params.deg - 1)
    ylb = [params.inipts[:, 1].min()-1]*(params.deg - 1)
    zlb = [params.inipts[:, 2].min()]*(params.deg - 1)

    # Only need the Z value from final points since we want to keep everything else in a box above the initial points
    xub = [params.inipts[:, 0].max()+1]*(params.deg - 1)
    yub = [params.inipts[:, 1].max()+1]*(params.deg - 1)
    zub = [params.finalpts[:, 2].max()]*(params.deg - 1)

    return Bounds(xlb+ylb+zlb, xub+yub+zub)

# def groupPts(inipts, finpts):
#     finptscpy = finpts.copy()
#     ptpairs = []
#     for i in range(5):
#         for j in range(2):
#             # Find the 5x20 rectangle of initial points
#             temp = [inipts[i*5 + j*500 + k*25:i*5 + j*500 + k*25 + 5] for k in range(20)]
#             initemp = np.concatenate(temp)

#             # Find the 100 final points that are closest to the CoM of the initial point rectangle
#             com = initemp.mean(0)
#             idxs = np.argpartition([np.linalg.norm(com[:2] - pt[:2]) for pt in finptscpy], 99)[:100]
#             fintemp = finptscpy[idxs]

#             # Remove the 100 points from the final points so that we don't map multiple initial
#             # points to a final point
#             mask = np.ones(finptscpy.shape[0], dtype=bool)
#             mask[idxs] = False
#             finptscpy = finptscpy[mask, :]

#             ptpairs.append((initemp, fintemp))

#     return ptpairs


class Parameters:
    """Constant parameters for the mission.

    :param nveh: Number of vehicles of the mission
    :type nveh: int
    :param ndim: Number of dimensions of the problem (i.e. 2D or 3D)
    :type ndim: int
    :param deg: Bernstein polynomial degree
    :type deg: int
    :param dsafe: Minimum safe distance between vehicles
    :type dsafe: float
    """

    def __init__(self, nveh, ndim, deg, dsafe, t0=0, tf=30):
        self.nveh = nveh
        self.ndim = ndim
        self.deg = deg
        self.dsafe = float(dsafe)
        self.t0 = t0
        self.tf = tf

        self.inipts = None
        self.finalpts = None


if __name__ == '__main__':
    plt.close('all')
    NDIM = 3        # Number of dimensions
    DEG = 2         # Order of the Bernstein polynomial approximation
    DSAFE = 0.9     # Minimum safety distance between vehicles
    np.random.seed(3)

    finpts = createFinpts('GroupedHawksLogo_1000pts.csv')
    inipts = createInipts(finpts)
    params = Parameters(1000, NDIM, DEG, DSAFE)

    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # for i in range(10):
    #     ax.plot(inipts[100*i:(i+1)*100, 0], inipts[100*i:(i+1)*100, 1],
    #             inipts[100*i:(i+1)*100, 2], '.', color=f'C{i}', markersize=10)
    #     ax.plot(finpts[100*i:(i+1)*100, 0], finpts[100*i:(i+1)*100, 1],
    #             finpts[100*i:(i+1)*100, 2], '.', color=f'C{i}', markersize=10, label=f'idx: {i}')

    # ax.legend()

    trajs = []
    tstart = time.time()
    for j in range(10):
        params.inipts = inipts[j*100:(j+1)*100, :]
        params.finalpts = finpts[j*100:(j+1)*100, :]
        bounds = createBounds(params)
        traj = np.atleast_2d([])
        params.traj = np.atleast_2d([])
        for i in range(100):
            print(f'Current Vehicle: {i + j*100}')
            x0 = initguess(i, params)

            def fun(x): return cost(x, i, params)
            res = minimize(fun,
                           x0,
                           # constraints=cons,
                           bounds=bounds,
                           method='SLSQP',
                           options={'maxiter': 250,
                                    'disp': True,
                                    'iprint': 0})

            traj = reshape(res.x, traj, params.ndim, params.inipts[i, :], params.finalpts[i, :])
            params.traj = traj.copy()

        trajs.append(traj)

    tend = time.time()
    trajs = np.concatenate(trajs)
    print('===============================================================')
    print(f'Total computation time for 1000 vehicles: {tend-tstart}')
    print('===============================================================')

    # Plot the trajectories
    plt.close('all')

    temp = Bernstein(trajs[0:NDIM, :])
    vehList = [temp]
    ax = temp.plot(showCpts=False)
    plt.plot([temp.cpts[0, -1]], [temp.cpts[1, -1]], [temp.cpts[2, -1]],
             'k.', markersize=15, zorder=10)
    for i in range(1000):
        temp = Bernstein(trajs[i*NDIM:(i+1)*NDIM, :])
        vehList.append(temp)
        temp.plot(ax, showCpts=False)
        plt.plot([temp.cpts[0, -1]], [temp.cpts[1, -1]], [temp.cpts[2, -1]],
                 'k.', markersize=15, zorder=10)

    distances = []
    for i, traj in enumerate(vehList[:-1]):
        for traj2 in vehList[i+1:]:
            dv = traj - traj2
            distances.append(dv.normSquare().min())

    minDist = np.array(distances)
    minDist.sort()
    print('10 smallest distances:')
    print(minDist[:10])

    i = 0
    while os.path.exists('1000_veh_sequential_' + str(i) + '.pickle'):
        i += 1
    with open('1000_veh_sequential_' + str(i) + '.pickle', 'wb') as f:
        pickle.dump(vehList, f)
