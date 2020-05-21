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
from optimization.ObstacleAvoidance import obstacleAvoidance
from optimization.Speed import speed
from optimization.TemporalSeparation import temporalSeparation
from polynomial.bernstein import Bernstein


def initGuess(params):
    """
    Initial guess for the optimizer.

    We use a straight line guess from the second control point to the second to
    last control point. This is because the first and last control points are
    defined by the initial and final positions and the second and second to
    last control points are defined by the initial and final points along with
    the initial and final speeds and angles.

    The initial guess vector is laid out as follows:
        [x_{1, 2}, ..., x_{1, n-2}, y_{1, 2}, ..., y_{1, n-2}, x_{2, 2}, ...,
         y_{2, n-2}, ..., x_{v, 2}, ..., y_{v, n-2}]
    Where x_{1, 2} is the second X control point of the first vehicle, n is the
    degree of polynomials being used, and v is the number of vehicles.

    Parameters
    ----------
    params : Parameters
        Class containing the parameters of the problem.

    Returns
    -------
    numpy.array
        1D vectory containing the initial guess for the optimizer. See above
        description for the layout of the vector.

    """
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
    """
    Reshapes the optimization vector X to a usable matrix for computing the
    cost and constraints.

    By keeping certain values constant, such as the initial and final
    positions, the reshape command can effectively be used to guarantee
    equality constraints are met without increasing the computational
    complexity of the optimization.

    See initGuess for the format of the x vector.

    The resulting y matrix is of the following format:
        [[x_{1, 0}, ..., x_{1, n}],
         [y_{1, 0}, ..., y_{1, n}],
         [x_{2, 0}, ..., x_{2, n}],
         [y_{2, 0}, ..., y_{2, n}],
         ...
         [x_{v, 0}, ..., x_{v, n}],
         [y_{v, 0}, ..., y_{v, n}]]
    Where x_{1, 0} is the 0th control point of the first vehicle in the X
    dimension, n is the degree of the polynomials being used, and v is the
    number of vehicles.

    Parameters
    ----------
    x : numpy.array
        Optimization vector to be reshaped.
    deg : int
        Degree of the polynomials being used.
    nveh : int
        Number of vehicles.
    tf : float
        Final time of the mission. Assuming initial time is 0.
    inipts : numpy.array
        Initial points of each vehicle where the rows correspond to the
        vehicles and the columns correspond to the X and Y positions (i.e.
        column 0 is the X column and column 1 is the Y column).
    finalpts : numpy.array
        Final points of each vehicle. Same format as inipts.
    inispeeds : numpy.array
        Initial speeds of each vehicle. Each entry corresponds to a vehicle.
    finalspeeds : numpy.array
        Final speeds of each vehicle. Each entry corresponds to a vehicle.
    inipsis : numpy.array
        Initial heading angles of each vehicle. Each entry corresponds to a
        vehicle.
    finalpsis : numpy.array
        Final heading angles of each vehicle. Each entry corresponds to a
        vehicle.

    Returns
    -------
    y : numpy.array
        Reshaped optimization vector. See above description for more info.

    """
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
    """
    Nonlinear constraints for the optimization problem.

    These constraints include maximum speed, maximum angular rate, minimum
    safe temporal distance between vehicles, and minimum safe distance between
    vehicles and obstacles.

    Parameters
    ----------
    x : numpy.array
        1D optimization vector.
    params : Parameters
        Parameters for the problem being solved.

    Returns
    -------
    numpy.array
        Degree elevated approximation of the nonlinear constraints of the
        problem where all constraints must be >= 0 to be feasible.

    """
    y = reshape(x, params.deg, params.nveh, params.tf, params.inipts,
                params.finalpts, params.inispeeds, params.finalspeeds,
                params.inipsis, params.finalpsis)
    trajs = buildTrajList(y, params.nveh, params.tf)

    speeds = [params.vmax**2 - speed(traj) for traj in trajs]
    angRates = [params.wmax**2 - angularRate(traj) for traj in trajs]
    separation = temporalSeparation(trajs) - params.dsafe**2
    obstacles = obstacleAvoidance(trajs, params.obstacles) - params.dobs**2

    # Note that we are using * here to unwrap speeds and angRates from the
    # lists that they are in so that concatenate works
    return np.concatenate([*speeds, *angRates, separation, obstacles])


def buildTrajList(y, nveh, tf):
    """
    Builds a list of Bernstein trajectory objects given the reshapped matrix y.

    Parameters
    ----------
    y : numpy.array
        Reshapped optimization vector.
    nveh : int
        Number of vehicles.
    tf : float
        Final time. Note that initial time is assumed to be 0.

    Returns
    -------
    trajs : list
        List of Bernstein trajectory objects.

    """
    trajs = []
    for i in range(nveh):
        trajs.append(Bernstein(y[2*i:2*(i+1), :], tf=tf))

    return trajs


def cost(x, params):
    """
    Returns the Euclidean cost of the current x vector.

    While there are many different possible cost functions that can be used,
    such as minimum acceleration, minimum jerk, minimum final time, etc., the
    Euclidean cost has been seen to work well for this specific problem.

    The Euclidean cost represents the sum of the Euclidean distance between
    all neighboring control points.

    Parameters
    ----------
    x : numpy.array
        Optimization vector.
    params : Parameters
        Problem parameters.

    Returns
    -------
    float
        Cost of problem at the current x value.

    """
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
        """
        Parameters for the current optimization problem.

        Returns
        -------
        None.

        """
        self.nveh = 3   # Number of vehicles
        self.deg = 7    # Degree of Bernstein polynomials being used
        self.tf = 30.0  # Final time. Note that the initial time is assumed 0
        self.dsafe = 1  # Minimum safe distance between vehicles
        self.dobs = 2   # Minimum safe distance between vehicles and obstacles
        self.vmax = 10  # Maximum speed
        self.wmax = 3*np.pi/2 # Maximum angular rate

        # Initial points
        self.inipts = np.array([[0, 0],
                                [10, 0],
                                [20, 0]])
        # Initial speeds
        self.inispeeds = np.array([1, 1, 1])
        # Initial heading angles
        self.inipsis = np.array([np.pi/2, np.pi/2, np.pi/2])

        # Final points
        self.finalpts = np.array([[20, 30],
                                  [0, 30],
                                  [10, 30]])
        # Final speeds
        self.finalspeeds = np.array([1, 1, 1])
        # Final heading angles
        self.finalpsis = np.array([np.pi/2, np.pi/2, np.pi/2])

        # Obstacle positions
        self.obstacles = np.array([[7, 11],
                                   [13, 18],
                                   [6, 23],
                                   [0, 15],
                                   [15, 5],
                                   [20, 23]])


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
    fig, ax = plt.subplots()
    for traj in trajs:
        traj.plot(ax, showCpts=False)
        pt10 = traj(10)
        plt.plot(pt10[0], pt10[1], 'x', markeredgewidth=7, zorder=10)

    for obs in params.obstacles:
        obsArtist = plt.Circle(obs, radius=params.dobs, edgecolor='Black')
        ax.add_artist(obsArtist)

    ax.set_xlim([-5, 25])
    ax.set_aspect('equal')
