#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 12:32:39 2020

@author: ckielasjensen
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit
import numpy as np
from scipy.optimize import Bounds, minimize
import time

from optimization.AngularRate import angularRate
from optimization.Speed import speed
from optimization.ObstacleAvoidance import obstacleAvoidance
from polynomial.bernstein import Bernstein
from utils import setRCParams, resetRCParams, saveFigs


FIG_DIR = 'Figures/Dubins'
FIG_FORMAT = 'svg'


# def setRCParams():
#     # Run this to make sure that the matplotlib plots have the correct font type
#     # for an IEEE publication. Also sets font sizes and line widths for easier
#     # viewing.
#     plt.rcParams.update({
#                 'font.size': 32,
#                 'pdf.fonttype': 42,
#                 'ps.fonttype': 42,
#                 'figure.titlesize': 32,
#                 'legend.fontsize': 24,
#                 'xtick.labelsize': 24,
#                 'ytick.labelsize': 24,
#                 'lines.linewidth': 4,
#                 'lines.markersize': 18,
#                 'figure.figsize': [13.333, 10]
#                 })
#     # plt.tight_layout()


# def resetRCParams():
#     # Reset the matplotlib parameters
#     plt.rcParams.update(plt.rcParamsDefault)


def animateTrajectory(trajectories):
    """Animates the trajectories

    """
    global ani

    curveLen = len(trajectories[0].curve[0])
    fig, ax = plt.subplots()
    [ax.plot(traj.curve[0], traj.curve[1], '-', lw=3) for traj in trajectories]
    lines = [ax.plot([], [], 'o', markersize=20)[0] for traj in trajectories]

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(frame):
        for i, line in enumerate(lines):
            traj = trajectories[i]
            try:
                line.set_data(traj.curve[0][frame],
                              traj.curve[1][frame])
            except IndexError:
                line.set_data(traj.curve[0][curveLen-frame-1],
                              traj.curve[1][curveLen-frame-1])
        return lines

    plt.axis('off')
    ani = animation.FuncAnimation(fig,
                                  animate,
                                  len(trajectories[0].curve[0])*2,
                                  init_func=init,
                                  interval=10,
                                  blit=True,
                                  repeat=True)

    plt.show()


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

    travelDist = np.linalg.norm(params.inipt - params.finalpt)
    tf = travelDist*2 / params.vmax

    inimag = params.inispeed*tf/params.deg
    finalmag = params.finalspeed*tf/params.deg

    x1 = params.inipt[0] + inimag*np.cos(params.inipsi)
    y1 = params.inipt[1] + inimag*np.sin(params.inipsi)

    xn_1 = params.finalpt[0] - finalmag*np.cos(params.finalpsi)
    yn_1 = params.finalpt[1] - finalmag*np.sin(params.finalpsi)

    x0.append(np.linspace(x1, xn_1, params.deg-1)[1:-1])
    x0.append(np.linspace(y1, yn_1, params.deg-1)[1:-1])

    return np.concatenate([*x0, [tf]])


@njit(cache=True)
def reshape(x, deg, inipt, finalpt, inispeed, finalspeed, inipsi, finalpsi):
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
    times : numpy.array
        Vector containing the final times of each vehicle.

    """
    tf = x[-1]

    y = np.empty((2, deg+1))

    y[:, 0] = inipt
    y[:, -1] = finalpt

    inimag = inispeed*tf/deg
    finalmag = finalspeed*tf/deg

    y[0, 1] = inipt[0] + inimag*np.cos(inipsi)
    y[1, 1] = inipt[1] + inimag*np.sin(inipsi)
    y[0, -2] = finalpt[0] - finalmag*np.cos(finalpsi)
    y[1, -2] = finalpt[1] - finalmag*np.sin(finalpsi)

    y[:, 2:-2] = x[:-1].reshape((2, -1))

    return y, tf


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
    y, tf = reshape(x, params.deg, params.inipt, params.finalpt, params.inispeed,
                       params.finalspeed, params.inipsi, params.finalpsi)
    traj = Bernstein(y, t0=0., tf=tf)

    maxSpeed = params.vmax**2 - speed(traj)
    angRate = angularRate(traj)
    angRateMax = params.wmax - angRate #params.wmax - angRate
    angRateMin = angRate + params.wmax
    separation = obstacleAvoidance([traj], params.obstacles, elev=params.degElev) - params.dsafe**2

    return np.concatenate([maxSpeed, angRateMax, angRateMin, separation])


def cost(x):
    """
    Returns the time cost of the current x vector.

    This cost function returns the sum of the final times for all vehicles,
    effectively minimizing the total combined flight time of all vehicles.

    Parameters
    ----------
    x : numpy.array
        Optimization vector.
    nveh : int
        Number of vehicles.

    Returns
    -------
    float
        Cost of problem at the current x value.

    """
    # y, tf = reshape(x, params.deg, params.inipt, params.finalpt, params.inispeed,
    #                    params.finalspeed, params.inipsi, params.finalpsi)
    # traj = Bernstein(y, t0=0., tf=tf)

    # return tf + traj.diff().diff().normSquare().cpts.sum()**2
    tf = x[-1]
    if tf > 0:
        return tf
    else:
        return np.inf


def plotConstraints(trajs, params, legNames):
    """
    Plots the constraints of the problem to verify whether they are being met.

    Parameters
    ----------
    trajs : list
        List of Bernstein trajectories.
    params : Parameters
        Parameters of the problem.

    Returns
    -------
    None.

    """
    XLIM = [-1, 11]
    speedFig, speedAx = plt.subplots()
    tanAngFig, tanAngAx = plt.subplots()
    angRateFig, angRateAx = plt.subplots()

    for i, traj in enumerate(trajs):
        xdot = traj.diff().x
        ydot = traj.diff().y
        xddot = xdot.diff()
        yddot = ydot.diff()

        speed = xdot*xdot + ydot*ydot
        speed.plot(speedAx, showCpts=False, label=legNames[i])

        tanAng = ydot / xdot
        tanAng.plot(tanAngAx, showCpts=False, label=legNames[i])

        num = yddot*xdot - xddot*ydot
        den = xdot*xdot + ydot*ydot
        angRate = num / den
        angRate.plot(angRateAx, showCpts=False, label=legNames[i])

    speedAx.plot(XLIM, [params.vmax**2]*2, '--', label=r'$v^2_{max}$')

    angRateAx.plot(XLIM, [params.wmax]*2, '--', label=r'$\omega_{max}$')
    angRateAx.plot(XLIM, [-params.wmax]*2, '--', label=r'$\omega_{min}$')

    speedAx.set_xlim(XLIM)
    speedAx.legend(fontsize=32)
    speedAx.set_xlabel('Time (s)')
    speedAx.set_ylabel(r'Squared Speed $\left( \frac{m}{s} \right)^2$')
    speedAx.set_title('Speed Constraints')
    tanAngAx.set_xlim(XLIM)
    tanAngAx.set_ylim([-25, 25])
    tanAngAx.legend(fontsize=32)
    tanAngAx.set_xlabel('Time (s)')
    tanAngAx.set_ylabel(r'$\tan (\psi)$')
    tanAngAx.set_title('Tangent of Heading Angle')
    angRateAx.set_xlim(XLIM)
    angRateAx.set_ylim([-7.5, 3])
    angRateAx.legend(fontsize=32)
    angRateAx.set_xlabel('Time (s)')
    angRateAx.set_ylabel(r'Angular Rate $\left( \frac{rad}{s} \right)$')
    angRateAx.set_title('Angular Velocity Constraints')


# def saveFigs():
#     import os
#     # Create a Figures directory if it doesn't already exist
#     if not os.path.isdir(FIG_DIR):
#         os.mkdir(FIG_DIR)

#     for i in plt.get_fignums():
#         fig = plt.figure(i)
#         ax = fig.get_axes()[0]
#         title = ax.get_title()
#         print(f'Saving figure {i} - {title}')

#         ax.set_title('')
#         plt.tight_layout()
#         plt.draw()
#         saveName = os.path.join(FIG_DIR, title.replace(' ', '_') + '.' + FIG_FORMAT)
#         fig.savefig(saveName, format=FIG_FORMAT)
#         ax.set_title(title)
#         plt.draw()

#     print('Done saving figures')


class Parameters:
    def __init__(self):
        """
        Parameters for the current optimization problem.

        Returns
        -------
        None.

        """
        self.deg = 10   # Degree of Bernstein polynomials being used
        self.dsafe = 1  # Minimum safe distance between vehicle and obstacles
        self.vmax = 5   # Maximum speed
        self.wmax = 1   # Maximum angular rate

        self.inipt = np.array([3, 0])
        self.inispeed = 1
        self.inipsi = np.pi/2 + 1e-6 # Add epsilon so that the tangent plot has real values

        self.finalpt = np.array([7, 10])
        self.finalspeed = 1
        self.finalpsi = np.pi/2 + 1e-6

        self.obstacles = np.array([[3, 2], [6, 7]])


if __name__ == '__main__':
    plt.close('all')
    setRCParams()

    # Set everything up for the optimization
    params = Parameters()
    x0 = initGuess(params)
    lb = np.array([-300]*2*(params.deg-3) + [0.001])
    ub = np.array([300]*2*(params.deg-3) + [np.inf])
    bounds = Bounds(lb, ub)

    legNames = ('No Elevation', 'Elevated by 30', 'Elevated by 100', 'Exact Extrema')
    legNamesI = iter(legNames)
    fig, ax = plt.subplots()
    trajs = []
    for degElev in [0, 30, 100, np.inf]:
        params.degElev = degElev
        cons = [{'type': 'ineq',
                 'fun': lambda x: nonlcon(x, params)}]

        tstart = time.time()
        # Call the optimizer
        results = minimize(cost, x0,
                           constraints=cons,
                           bounds=bounds,
                           method='SLSQP',
                           options={'maxiter': 250,
                                    'disp': True,
                                    'iprint': 1})
        print(f'Computation time: {time.time()-tstart}')

        # Plot everything
        y, tf = reshape(results.x, params.deg, params.inipt, params.finalpt, params.inispeed,
                        params.finalspeed, params.inipsi, params.finalpsi)
        trajs.append(Bernstein(y, t0=0., tf=tf))
        x0 = results.x

    for traj in trajs:
        traj.plot(ax, showCpts=False, label=next(legNamesI))

    obs1 = plt.Circle(params.obstacles[0], radius=params.dsafe, ec='k', fc='r')
    obs2 = plt.Circle(params.obstacles[1], radius=params.dsafe, ec='k', fc='g')
    ax.add_artist(obs1)
    ax.add_artist(obs2)

    ax.set_title('Dubins Car Time Optimal')
    ax.set_xlim([-0.5, 15.5])
    ax.set_ylim([-0.5, 10.5])
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()

    plotConstraints(trajs, params, legNames)

    saveFigs(FIG_DIR)

    plt.show()
    resetRCParams()
