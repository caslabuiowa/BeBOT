#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 13:48:10 2020

@author: ckielasjensen
"""

import matplotlib.pyplot as plt
from numba import njit
import numpy as np
from scipy.optimize import minimize, Bounds

from optimization.AngularRate import angularRate
from optimization.Speed import speed
from optimization.TemporalSeparation import temporalSeparation
from polynomial.bernstein import Bernstein


def initGuess(params):
    x0 = []
    for i in range(params.nveh):
        inimag = params.inispeeds[i]*params.tf/params.deg
        finalmag = params.finalspeeds[i]*params.tf/params.deg

        x1 = params.inipts[i, 0] + inimag*np.cos(params.inipsis[i])
        y1 = params.inipts[i, 1] + inimag*np.sin(params.inipsis[i])

        xn_1 = params.finalpts[i, 0] - finalmag*np.cos(params.finalpsis[i])
        yn_1 = params.finalpts[i, 1] - finalmag*np.sin(params.finalpsis[i])

        x0.append(np.linspace(x1, xn_1, params.deg-1)[1:-1])
        x0.append(np.linspace(y1, yn_1, params.deg-1)[1:-1])

    return np.concatenate(x0)


@njit(cache=True)
def reshape(x, deg, nveh, tf, inipts, finalpts, inispeeds, finalspeeds,
            inipsis, finalpsis):
    y = np.empty((2*nveh, deg+1))

    y[0::2, 0] = inipts[:, 0]
    y[1::2, 0] = inipts[:, 1]
    y[0::2, -1] = finalpts[:, 0]
    y[1::2, -1] = finalpts[:, 1]

    inimags = inispeeds*tf/deg
    finalmags = finalspeeds*tf/deg

    y[0::2, 1] = inipts[:, 0] + inimags*np.cos(inipsis)
    y[1::2, 1] = inipts[:, 1] + inimags*np.sin(inipsis)
    y[0::2, -2] = finalpts[:, 0] - finalmags*np.cos(finalpsis)
    y[1::2, -2] = finalpts[:, 1] - finalmags*np.sin(finalpsis)

    y[:, 2:-2] = x.reshape((2*nveh, -1))

    return y


def nonlcon(x, params):
    y = reshape(x, params.deg, params.nveh, params.tf, params.inipts,
                params.finalpts, params.inispeeds, params.finalspeeds,
                params.inipsis, params.finalpsis)
    trajs = buildTrajList(y, params.nveh, params.tf)

    speeds = [params.vmax**2 - speed(traj) for traj in trajs]
    angRates = [params.wmax**2 - angularRate(traj) for traj in trajs]
    separation = temporalSeparation(trajs) - params.dsafe**2

    # Note that we are using * here to unwrap speeds and angRates from the
    # lists that they are in so that concatenate works
    return np.concatenate([*speeds, *angRates, separation])


def buildTrajList(y, nveh, tf):
    trajs = []
    for i in range(nveh):
        trajs.append(Bernstein(y[2*i:2*(i+1), :], tf=tf))

    return trajs


def cost(x, params):
    y = reshape(x, params.deg, params.nveh, params.tf, params.inipts,
                params.finalpts, params.inispeeds, params.finalspeeds,
                params.inipsis, params.finalpsis)

    return _euclideanObjective(y, params.nveh, 2)


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


class Parameters:
    def __init__(self):
        self.nveh = 3
        self.deg = 5
        self.tf = 30
        self.dsafe = 1
        self.vmax = 10
        self.wmax = np.pi/2

        self.inipts = np.array([[0, 0],
                                [10, 0],
                                [20, 0]])
        self.inispeeds = np.array([1, 1, 1])
        self.inipsis = np.array([np.pi/2, np.pi/2, np.pi/2])

        self.finalpts = np.array([[20, 30],
                                  [0, 30],
                                  [10, 30]])
        self.finalspeeds = np.array([1, 1, 1])
        self.finalpsis = np.array([np.pi/2, np.pi/2, np.pi/2])


if __name__ == '__main__':
    params = Parameters()

    x0 = initGuess(params)

    def fn(x): return cost(x, params)
    cons = [{'type': 'ineq',
             'fun': lambda x: nonlcon(x, params)}]
    bounds = Bounds(-1000, 1000)

    results = minimize(fn, x0,
                       constraints=cons,
                       bounds=bounds,
                       method='SLSQP',
                       options={'maxiter': 250,
                                'disp': True,
                                'iprint': 2})

    y = reshape(results.x, params.deg, params.nveh, params.tf, params.inipts,
                params.finalpts, params.inispeeds, params.finalspeeds,
                params.inipsis, params.finalpsis)

    trajs = buildTrajList(y, params.nveh, params.tf)

    plt.close('all')
    ax = trajs[0].plot()
    for traj in trajs[1:]:
        traj.plot(ax)
