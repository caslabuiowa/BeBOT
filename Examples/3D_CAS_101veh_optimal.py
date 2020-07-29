#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 22:47:16 2020

@author: ckielasjensen
"""

import random
import time

import matplotlib._color_data as mcd
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds

# import bezier as bez
from polynomial.bernstein import Bernstein
from constants import DEG_ELEV


class Parameters:
    """
    """
    def __init__(self):
        self.deg = 2            # Order of approximation
        self.ndim = 3           # Number of dimensions
        self.dsafe = 0.75       # Minimum safe distance between vehicles (m)
        self.odsafe = 2         # Minimum safe distance from obstacles (m)

        self.obsLoc = np.array([(13, 10),
                                (18, 18),
                                (25, 11)])

        # Final points
        df = pd.read_csv('CAS_PixelImage.csv')
        nveh = df.shape[0]
        self.nveh = nveh
        self.finalPts = np.concatenate((df.values, 100*np.ones((nveh, 1))), axis=1)
        self.finalPts = np.ascontiguousarray(self.finalPts)

        # Initial points
        np.random.seed(3)
        lines = np.atleast_2d(np.arange(1, 25, 2))
        x = np.repeat(lines, 12, axis=1).T
        y = np.repeat(lines, 12, axis=0).reshape((-1, 1))
        pts = np.concatenate((x, y), axis=1)
        idxs = np.random.choice(np.arange(len(pts)), size=nveh, replace=False)

        self.iniPts = np.concatenate([pts[idxs, :], np.zeros((nveh, 1))], axis=1)


def init_guess(params):
    """
    """
    temp = np.empty((params.nveh*3, params.deg-1))
    x = np.linspace(params.iniPts[:, 0], params.finalPts[:, 0], params.deg+1)
    y = np.linspace(params.iniPts[:, 1], params.finalPts[:, 1], params.deg+1)
    z = np.linspace(params.iniPts[:, 2], params.finalPts[:, 2], params.deg+1)

    temp[::3, :] = x.T[:, 1:-1]
    temp[1::3, :] = y.T[:, 1:-1]
    temp[2::3, :] = z.T[:, 1:-1]

    x0 = temp.reshape(-1)

    return x0


@njit(cache=True)
def reshape(x, nveh, deg, iniPts, finalPts):
    """Reshapes the X vector being optimized into a matrix for computation

    Note that this only works for 2-dimensional problems.

    :param x: X vector over which the optimization is happening
    :type x: np.ndarray
    :param inipt: Initial point of the vehicle
    :type inipt: np.ndarray
    :param finalpt: Final points of the current vehicle
    :type finalpt: np.ndarray
    :return: Final time, tf, and reshaped vector, y
    :rtype: Tuple where tf is a float and y is a np.ndarray
    """
    ndim = 3

    # Reshape X
    y = np.empty((ndim*nveh, deg+1))
    y[:, 1:-1] = x.reshape((ndim*nveh, deg-1))

    # Initial and final points
    y[::3, 0] = iniPts[:, 0]
    y[1::3, 0] = iniPts[:, 1]
    y[2::3, 0] = iniPts[:, 2]
    y[::3, -1] = finalPts[:, 0]
    y[1::3, -1] = finalPts[:, 1]
    y[2::3, -1] = finalPts[:, 2]

    return y


def build_traj_list(y, params):
    """
    """
    trajs = []
    for i in range(params.nveh):
        trajs.append(Bernstein(y[i*params.ndim:(i+1)*params.ndim, :]))

    return trajs


def cost(x, params):
    """
    """
    y = reshape(x, params.nveh, params.deg, params.iniPts, params.finalPts)
    # trajs = build_traj_list(y, params)

    # return sum(sum([traj.diff().diff().normSquare().cpts.squeeze() for traj in trajs]))

    return _euclideanObjective(y, params.nveh, params.ndim)
    # return np.linalg.norm(np.diff(y))

@njit(cache=True)
def _euclideanObjective(y, nVeh, dim):
    """Sums the Euclidean distance between control points.

    The Euclidean difference between each neighboring pair of control points is
    summed for each vehicle.

    :param y: Optimized vector that has been reshaped using the reshapeVector
        function.
    :type y: numpy.ndarray
    :param nVeh: Number of vehicles
    :type nVeh: int
    :param dim: Dimension of the vehicles. Currently only works for 2D
    :type dim: int
    :return: Sum of the Euclidean distances
    :rtype: float
    """
    summation = 0.0
    temp = np.empty(dim)
    length = y.shape[1]
    for veh in range(nVeh):
        for i in range(length-1):
            for j in range(dim):
                temp[j] = y[veh*dim+j, i+1] - y[veh*dim+j, i]

            summation += _norm(temp)

    return summation


@njit(cache=True)
def _norm(x):
    summation = 0.0
    for val in x:
        summation += val*val

    return np.sqrt(summation)


def nonlinear_constraints(x, params):
    """
    """
    y = reshape(x, params.nveh, params.deg, params.iniPts, params.finalPts)
    trajs = build_traj_list(y, params)

    tempSep = temporal_separation_cons(trajs, params)
#    obsSep = obstacle_cons(trajs, params)

#    return np.concatenate([tempSep, obsSep])
    return tempSep


def temporal_separation_cons(trajs, params):
    """
    """
    nveh = params.nveh

    # loopCount = int(0.5*(nveh-1)*nveh)  # found using arithmetic series sum
    # distVeh = np.empty(loopCount)
    distVeh = []
    # idx = 0
    for i in range(nveh-1):
        for j in range(i+1, nveh):
            dv = trajs[i] - trajs[j]
            # distVeh[idx] = dv.normSquare().elev(DEG_ELEV).cpts.min()
            distVeh.append(dv.normSquare().elev(DEG_ELEV).cpts.squeeze())
            # idx += 1

    return np.concatenate(distVeh) - params.dsafe**2


def obstacle_cons(trajs, params):
    """
    """
    nveh = params.nveh
    nobs = params.obsLoc.shape[0]
    tf = trajs[0].tf

    yobs = np.repeat(params.obsLoc.reshape((-1, 1)), params.deg+1, 1)
    obstacles = build_traj_list(yobs, tf, params)

    loopCount = nveh*nobs
    distObs = np.empty(loopCount)
    idx = 0
    for i in range(nveh):
        for j in range(nobs):
            dv = trajs[i] - obstacles[j]
            distObs[idx] = dv.normSquare().elev(DEG_ELEV).cpts.min()
#            distObs[idx] = dv.normSquare().min()
            idx += 1

    return distObs - params.odsafe**2


def main():
    params = Parameters()
    x0 = init_guess(params)

    def fn(x): return cost(x, params)

    cons = [{'type': 'ineq',
             'fun': lambda x: nonlinear_constraints(x, params)}]

    tstart = time.time()
    results = minimize(fn, x0,
                       constraints=cons,
                       bounds=Bounds(-60, 160),
                       method='SLSQP',
                       options={'maxiter': 250,
                                'ftol': 1e-3,
                                'disp': True,
                                'iprint': 2})
    tend = time.time()
    print('===============================================================')
    print(f'Total computation time for {params.nveh} vehicles: {tend-tstart}')
    print('===============================================================')

    print(f'Results: {results.x}')

    print('Nonlcon:')
    print(nonlinear_constraints(results.x, params))

    y = reshape(results.x, params.nveh, params.deg, params.iniPts,
                params.finalPts)
    trajs = build_traj_list(y, params)

    ax = trajs[0].plot(showCpts=False)
    for traj in trajs[1:]:
        traj.plot(axisHandle=ax, showCpts=False)

    for curve in trajs:
        plt.plot([curve.cpts[0, -1]],
                 [curve.cpts[1, -1]],
                 [curve.cpts[2, -1]],
                 'k.', markersize=40)

#    plot_obstacles(params.obsLoc, ax, params)

#    plt.xlabel('X Position (m)')
#    plt.ylabel('Y Position (m)')
#    plt.xlim([0, 30])
#    plt.ylim([0, 25])

    return results, params


# def test():
#     params = Parameters()
#     x0 = init_guess(params, 10)
#     print(f'x0 = {x0.round(3)}')
#     tf, y = reshape(x0, params.nveh, params.deg, params.iniPts,
#                     params.iniSpeeds, params.iniAngs, params.finalPts,
#                     params.finalSpeeds, params.finalAngs)
#     print(f'tf = {tf}')
#     print(f'y = \n{y.round(3)}')

#     trajs = build_traj_list(y, tf, params)
#     temp_sep = temporal_separation_cons(trajs, params)
#     print(temp_sep.round(3))
#     obs_sep = obstacle_cons(trajs, params)
#     print(obs_sep.round(3))
#     print(np.atleast_2d(np.concatenate((temp_sep, obs_sep)).round(3)).T)
#     max_speed = max_speed_cons(trajs, tf, params)
    # print(max_speed.round(3))


def plot_obstacles(obstacles, ax, params):
    """
    """
    for obs in obstacles:
        color = random.choice(list(mcd.XKCD_COLORS.keys()))
        obsArtist = plt.Circle(obs,
                               radius=params.odsafe,
                               edgecolor='Black',
                               facecolor=color)
        ax.add_artist(obsArtist)


if __name__ == '__main__':
    plt.close('all')
    plt.rcParams.update({
            'font.size': 40,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'xtick.labelsize': 40,
            'ytick.labelsize': 40,
            'lines.linewidth': 4,
            'lines.markersize': 18
            })

    # results, params = main()

    params = Parameters()
    x0 = init_guess(params)

    def fn(x): return cost(x, params)
    bounds = Bounds([0, 0, 0]*params.nveh*(params.deg-1), [30, 35, 100]*params.nveh*(params.deg-1))
    cons = [{'type': 'ineq',
             'fun': lambda x: nonlinear_constraints(x, params)}]

    # for i in range(100):
    #     y = reshape(x0, params.nveh, params.deg, params.iniPts, params.finalPts)
    #     trajs = build_traj_list(y, params)
    #     fn(x0)
    #     temporal_separation_cons(trajs, params)

    tstart = time.time()
    results = minimize(fn, x0,
                        constraints=cons,
                        bounds=bounds,
                        method='SLSQP',
                        options={'maxiter': 250,
                                'ftol': 1e-3,
                                'disp': True,
                                'iprint': 2})
    tend = time.time()
    print('===============================================================')
    print(f'Total computation time for {params.nveh} vehicles: {tend-tstart}')
    print('===============================================================')

    print(f'Results: {results.x}')

    print('Nonlcon:')
    print(nonlinear_constraints(results.x, params))

    y = reshape(results.x, params.nveh, params.deg, params.iniPts,
                params.finalPts)
    trajs = build_traj_list(y, params)

    ax = trajs[0].plot(showCpts=False)
    for traj in trajs[1:]:
        traj.plot(axisHandle=ax, showCpts=False)

    for curve in trajs:
        plt.plot([curve.cpts[0, -1]],
                  [curve.cpts[1, -1]],
                  [curve.cpts[2, -1]],
                  'k.', markersize=40)

    y = reshape(results.x, params.nveh, params.deg, params.iniPts,
                    params.finalPts)
    trajs = build_traj_list(y, params)

    plt.show()
    # last run, 10184.20985 seconds (7/29/2020)
