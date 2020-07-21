#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 18:37:34 2020

@author: ckielasjensen
"""

from cartopy import crs
from cartopy.io.shapereader import natural_earth, Reader
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
from scipy.optimize import minimize, Bounds

from optimization.AngularRate import angularRate
from optimization.Speed import speed
from optimization.TemporalSeparation import temporalSeparation
from polynomial.bernstein import Bernstein
from utils import saveFigs


def setRCParams():
    # Run this to make sure that the matplotlib plots have the correct font type
    # for an IEEE publication. Also sets font sizes and line widths for easier
    # viewing.
    plt.rcParams.update({
                'font.size': 24,
                'figure.titlesize': 40,
                'pdf.fonttype': 42,
                'ps.fonttype': 42,
                # 'xtick.labelsize': 40,
                # 'ytick.labelsize': 40,
                'lines.linewidth': 4,
                'lines.markersize': 18,
                'figure.figsize': [13.333, 10]
                })
    # plt.tight_layout()


def resetRCParams():
    # Reset the matplotlib parameters
    plt.rcParams.update(plt.rcParamsDefault)


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
    times = []
    for i in range(params.nveh):
        travelDist = np.linalg.norm(params.inipts[i, :] -
                                    params.finalpts[i, :])
        tf = travelDist / np.mean([params.vmin, params.vmax])
        times.append(tf)

        inimag = params.inispeeds[i]*tf/params.deg
        finalmag = params.finalspeeds[i]*tf/params.deg

        x1 = params.inipts[i, 0] + inimag*np.cos(params.inipsis[i])
        y1 = params.inipts[i, 1] + inimag*np.sin(params.inipsis[i])

        xn_1 = params.finalpts[i, 0] - finalmag*np.cos(params.finalpsis[i])
        yn_1 = params.finalpts[i, 1] - finalmag*np.sin(params.finalpsis[i])

        x0.append(np.linspace(x1, xn_1, params.deg-1)[1:-1])
        x0.append(np.linspace(y1, yn_1, params.deg-1)[1:-1])



    return np.concatenate([*x0, times])


@njit(cache=True)
def reshape(x, deg, nveh, inipts, finalpts, inispeeds, finalspeeds,
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
    times = x[-nveh:]

    y = np.empty((2*nveh, deg+1))

    y[0::2, 0] = inipts[:, 0]
    y[1::2, 0] = inipts[:, 1]
    y[0::2, -1] = finalpts[:, 0]
    y[1::2, -1] = finalpts[:, 1]

    for i, tf in enumerate(times):
        inimags = inispeeds[i]*tf/deg
        finalmags = finalspeeds[i]*tf/deg

        y[2*i, 1] = inipts[i, 0] + inimags*np.cos(inipsis[i])
        y[2*i+1, 1] = inipts[i, 1] + inimags*np.sin(inipsis[i])
        y[2*i, -2] = finalpts[i, 0] - finalmags*np.cos(finalpsis[i])
        y[2*i+1, -2] = finalpts[i, 1] - finalmags*np.sin(finalpsis[i])

    y[:, 2:-2] = x[:-nveh].reshape((2*nveh, -1))

    return y, times


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
    y, times = reshape(x, params.deg, params.nveh, params.inipts,
                       params.finalpts, params.inispeeds, params.finalspeeds,
                       params.inipsis, params.finalpsis)
    trajs = buildTrajList(y, params.nveh, times)

    speeds = [speed(traj) for traj in trajs]
    minSpeeds = [vel - params.vmin**2 for vel in speeds]
    maxSpeeds = [params.vmax**2 - vel for vel in speeds]
    # angRates = [params.wmax**2 - angularRate(traj) for traj in trajs]
    angRates = [angularRate(traj) for traj in trajs]
    angRateMax = [params.wmax - angRate for angRate in angRates]
    angRateMin = [angRate + params.wmax for angRate in angRates]
    separation = temporalSeparation(trajs) - params.dsafe**2

    # Note that we are using * here to unwrap speeds and angRates from the
    # lists that they are in so that concatenate works
    return np.concatenate([*minSpeeds, *maxSpeeds, *angRateMax, *angRateMin, separation])


def buildTrajList(y, nveh, times):
    """
    Builds a list of Bernstein trajectory objects given the reshapped matrix y.

    Parameters
    ----------
    y : numpy.array
        Reshapped optimization vector.
    nveh : int
        Number of vehicles.
    times : numpy.array
        Vector containing the final times of each vehicle. Note that initial
        time is assumed to be 0.

    Returns
    -------
    trajs : list
        List of Bernstein trajectory objects.

    """
    trajs = []
    for i in range(nveh):
        trajs.append(Bernstein(y[2*i:2*(i+1), :], tf=times[i]))

    return trajs


def cost(x, nveh):
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
    times = x[-nveh:]

    return sum(times)


def plotConstraints(trajs, params):
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
    XLIM = [-1, max([traj.tf for traj in trajs])/(60*60)]
    speedFig, speedAx = plt.subplots()
    angRateFig, angRateAx = plt.subplots()
    distFig, distAx = plt.subplots()

    for i, traj in enumerate(trajs):
        time = np.linspace(traj.t0, traj.tf, traj.curve.shape[1])/(60*60)
        xdot = traj.diff().x
        ydot = traj.diff().y
        xddot = xdot.diff()
        yddot = ydot.diff()

        speed = xdot*xdot + ydot*ydot
        # speed.plot(speedAx, showCpts=False, label=f'Vehicle {i+1}')
        speedAx.plot(time, speed.curve.squeeze(), label=f'Vehicle {i+1}')

        num = yddot*xdot - xddot*ydot
        den = xdot*xdot + ydot*ydot
        angRate = (num.curve / den.curve).squeeze()**2
        # time = np.linspace(num.t0, num.tf, len(angRate))/(60*60)
        angRateAx.plot(time, angRate, label=f'Vehicle {i+1}')

    for i, traj in enumerate(trajs[:-1]):
        for j, traj2 in enumerate(trajs[i+1:]):
            dv = traj - traj2
            # dv.normSquare().plot(distAx, showCpts=False, label=f'$||Veh_{i+1} - Veh_{j+1}||^2$')
            time = np.linspace(dv.t0, dv.tf, dv.curve.shape[1])/(60*60)
            curve = dv.normSquare().curve.squeeze()
            distAx.plot(time, curve/(1000*1000), label=f'$||Veh_{i+1} - Veh_{i+2}||^2$')

    speedAx.plot(XLIM, [params.vmin**2]*2, 'r--', label=r'$v^2_{min}$')
    speedAx.plot(XLIM, [params.vmax**2]*2, 'r--', label=r'$v^2_{max}$')

    angRateAx.plot(XLIM, [params.wmax]*2, 'r--', label=r'$\omega^2_{max}$')
    angRateAx.plot(XLIM, [-params.wmax]*2, 'r--', label=r'$\omega^2_{min}$')

    distAx.plot(XLIM, [params.dsafe**2/(1000*1000)]*2, '--', label=r'$d^2_{s}$')

    # speedAx.set_xlim(XLIM)
    speedAx.legend()
    speedAx.set_xlabel('Time (hr)')
    speedAx.set_ylabel(r'Squared Speed $\left(\frac{m}{s}\right)^2$')
    speedAx.set_title('Speed Constraints')
    # angRateAx.set_xlim(XLIM)
    angRateAx.legend()
    angRateAx.set_xlabel('Time (hr)')
    angRateAx.set_ylabel(r'Angular Velocity $\left( \frac{rad}{s} \right)$')
    angRateAx.set_title('Angular Velocity Constraints')
    # distAx.set_xlim(XLIM)
    distAx.legend()
    distAx.set_xlabel('Time (hr)')
    distAx.set_ylabel(r'Squared Euclidean Distance $km^2$')
    distAx.set_title('Distance Constraints')


def drawUS(cities):
    """
    Draws a map of the US with cities marked on the map.

    An example of the cities that could be passed in is,

    cities = [
        [-117.1611, 32.7157],   # San Diego
        [-74.006, 40.7128],     # New York
        [-93.2650, 44.9778],    # Minneapolis
        [-122.3321, 47.6062],   # Seattle
        [-80.1918, 25.7617],    # Miami
        [-104.9903, 39.7392]    # Denver
        ]

    Parameters
    ----------
    cities : list
        2D list containing lat/long pairs of coordinates for each city.

    Returns
    -------
    ax : matplotlib.axes
        Axes on which the US and cities are plotted.

    """
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection=crs.LambertConformal())
    ax.set_extent([-125, -66.5, 20, 50], crs.Geodetic())

    name = 'admin_1_states_provinces_lakes_shp'
    states = natural_earth(resolution='110m', category='cultural', name=name)

    fc = [0.9375, 0.9375, 0.859375]
    ec = 'black'

    for state in Reader(states).geometries():
        ax.add_geometries([state], crs.PlateCarree(), fc=fc, ec=ec)

    for city in cities:
        ax.plot(city[0], city[1], 'ko', markersize=15,
                transform=crs.Geodetic())

    return ax


class Parameters:
    def __init__(self):
        """
        Parameters for the current optimization problem.

        Returns
        -------
        None

        """
        self.nveh = 4           # Number of vehicles
        self.deg = 5            # Degree of Bernstein polynomials being used
        self.dsafe = 5000       # Minimum safe distance between vehicles (m)
        self.vmin = 200         # Minimum speed (m/s)
        self.vmax = 260         # Maximum speed (m/s)
        self.wmax = 0.05236     # Maximum angular rate (rad/s)

        # Initial points
        inipts = np.array([
            [-117.1611, 32.7157],   # San Diego
            [-74.006, 40.7128],     # New York
            [-93.2650, 44.9778],    # Minneapolis
            [-122.3321, 47.6062]    # Seattle
            ])
        # Convert initial points from lat/lon to Google's Mercator projection
        self.inipts = crs.GOOGLE_MERCATOR.transform_points(crs.Geodetic(), *inipts.T)[:, :-1]
        # Initial speeds
        self.inispeeds = self.vmin*np.ones(4) + 5
        # Initial heading angles
        self.inipsis = np.array([0, np.pi, 0, 0])

        # Final points
        finalpts = np.array([
            [-93.2650, 44.9778],    # Minneapolis
            [-122.3321, 47.6062],   # Seattle
            [-80.1918, 25.7617],    # Miami
            [-104.9903, 39.7392]    # Denver
        ])
        # Convert final points from lat/lon to Goggle's Mercator projection
        self.finalpts = crs.GOOGLE_MERCATOR.transform_points(crs.Geodetic(), *finalpts.T)[:, :-1]
        # Final speeds
        self.finalspeeds = self.vmin*np.ones(4) + 5
        # Final heading angles
        self.finalpsis = np.array([0, np.pi, -np.pi/2, 0])


if __name__ == '__main__':
    plt.close('all')
    setRCParams()
    params = Parameters()

    # Set everything up for the optimization
    x0 = initGuess(params)
    def fn(x): return cost(x, params.nveh)
    cons = [{'type': 'ineq',
             'fun': lambda x: nonlcon(x, params)}]
    lb = np.array([-np.inf]*2*params.nveh*(params.deg-3) + params.nveh*[0.001])
    ub = np.array([np.inf]*2*params.nveh*(params.deg-3) + params.nveh*[np.inf])
    bounds = Bounds(lb, ub)

    # Call the optimizer
    # Since the cost function is optimizing final time (in seconds), we set the 'ftol' option to 60 since an extra
    # minute on 3+ hour flight won't really matter all that much (the default value is 1e-6)
    results = minimize(fn, x0,
                       constraints=cons,
                       bounds=bounds,
                       method='SLSQP',
                       options={'maxiter': 300,
                                'disp': True,
                                'ftol': 60,
                                'iprint': 2})

    # Plot everything
    y, times = reshape(results.x, params.deg, params.nveh, params.inipts,
                       params.finalpts, params.inispeeds, params.finalspeeds,
                       params.inipsis, params.finalpsis)

    trajs = buildTrajList(y, params.nveh, times)

    # plt.close('all')
    cities = [
        [-117.1611, 32.7157],   # San Diego
        [-74.006, 40.7128],     # New York
        [-93.2650, 44.9778],    # Minneapolis
        [-122.3321, 47.6062],   # Seattle
        [-80.1918, 25.7617],    # Miami
        [-104.9903, 39.7392]    # Denver
        ]
    ax = drawUS(cities)
    for i, traj in enumerate(trajs):
        # Not plotting the last point due to a bug in matplotlib/cartopy
        ax.plot(traj.curve[0, :-1], traj.curve[1, :-1], label=f'Vehicle {i+1}', transform=crs.GOOGLE_MERCATOR)
    ax.legend()
    ax.set_title('Flight Trajectories')
    # ax.name = 'cartopy'  # Needed due to a bug in matplotlib/cartopy

    plotConstraints(trajs, params)

    saveFigs('ATC_Figures')

    plt.show()
    resetRCParams()
