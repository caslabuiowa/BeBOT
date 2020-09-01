#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 21:29:21 2020

@author: ckielasjensen
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d

from constants import DEG_ELEV
from polynomial.bernstein import Bernstein
from polynomial.rationalbernstein import RationalBernstein


SAVE_FIG = True         # Set to True to save figures
FIG_FORMAT = 'svg'      # Used for the output format when saving figures
FIG_DIR = 'Figures_3D'     # Directory in which to save the figures
XLIM = [0, 11]
YLIM = [0, 11]
ZLIM = [0, 11]
ELEV = 55               # Elevation angle for 3D plot
AZIM = 45               # Azimuth angle for 3D plot

class Sphere:
    def __init__(self, x, y, z, r, color=None):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.color = color

    @property
    def center(self):
        return (self.x, self.y, self.z)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        u = np.linspace(0, 2*np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))*self.r + self.x
        y = np.outer(np.sin(u), np.sin(v))*self.r + self.y
        z = np.outer(np.ones(np.size(u)), np.cos(v))*self.r + self.z
        if self.color:
            ax.plot_surface(x, y, z, color=self.color, **kwargs)
        else:
            ax.plot_surface(x, y, z, **kwargs)


def setRCParams():
    # Run this to make sure that the matplotlib plots have the correct font type
    # for an IEEE publication. Also sets font sizes and line widths for easier
    # viewing.
    plt.rcParams.update({
                'font.size': 40,
                'pdf.fonttype': 42,
                'ps.fonttype': 42,
                'xtick.labelsize': 40,
                'ytick.labelsize': 40,
                'lines.linewidth': 4,
                'lines.markersize': 18,
                'figure.figsize': [13.333, 10]
                })
    # plt.tight_layout()


def resetRCParams():
    # Reset the matplotlib parameters
    plt.rcParams.update(plt.rcParamsDefault)


def formatPlot(ax, title, xlabel='X Position (m)', ylabel='Y Position (m)', zlabel='Z Position (m)'):
    ax.set_title(title)
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_zlim(ZLIM)
    ax.set_xlabel(xlabel, labelpad=30)
    ax.set_ylabel(ylabel, labelpad=30)
    ax.set_zlabel(zlabel, labelpad=30)
    ax.view_init(ELEV, AZIM)


def initPlot(c1, c2, obs):
    ax = c1.plot(color='C0', label=r'$\mathbf{C}^{[3]}(t)$')
    c2.plot(ax, color='C1', label=r'$\mathbf{C}^{[4]}(t)$')
    obs.plot(ax)

    formatPlot(ax, '3D Initial Figure')
    ax.legend()


def endPoints(c1, c2, obs):
    ax = c1.plot(showCpts=False, color='C0')
    c2.plot(ax, showCpts=False, color='C1')
    obs.plot(ax)

    for i, pt in enumerate(np.concatenate([c1.cpts[:, (0, -1)].T, c2.cpts[:, (0, -1)].T])):
        ax.plot([pt[0]], [pt[1]], [pt[2]], 'k.')

        # This if statement is purely for modifying the position of the drawn coordinates
        if i == 0:
            ax.text(pt[0]+3, pt[1]-3, pt[2], f'({pt[0]}, {pt[1]})')
        elif i == 1:
            ax.text(pt[0]-1, pt[1]-1, pt[2], f'({pt[0]}, {pt[1]})')
        elif i == 3:
            ax.text(pt[0]+2, pt[1]+2, pt[2], f'({pt[0]}, {pt[1]})')
        else:
            ax.text(pt[0], pt[1], pt[2], f'({pt[0]}, {pt[1]})')

    formatPlot(ax, '3D End Points')


def convexHull(c1, c2, obs):
    ax = c1.plot(color='C0')
    c2.plot(ax, color='C1')
    obs.plot(ax)

    for cpts in [c1.cpts, c2.cpts]:
        plotCvxHull(cpts, ax)

    formatPlot(ax, '3D Convex Hull')


def convexHullSplit(c1, c2, obs):
    c1L, c1R = c1.split(0.5*(c1.tf-c1.t0)+c1.t0)
    c2L, c2R = c2.split(0.5*(c2.tf-c2.t0)+c2.t0)

    ax = c1L.plot(color='C0')
    c1R.plot(ax, color='C2')
    c2L.plot(ax, color='C1')
    c2R.plot(ax, color='C3')

    for cpts in [c1L.cpts, c1R.cpts, c2L.cpts, c2R.cpts]:
        plotCvxHull(cpts, ax)

    obs.plot(ax)
    formatPlot(ax, '3D Split Convex Hull')


def convexHullElev(c1, c2, obs):
    c1E = c1.elev(10)
    c2E = c2.elev(10)

    ax = c1E.plot(color='C0')
    c2E.plot(ax, color='C1')

    for cpts in [c1E.cpts, c2E.cpts]:
        plotCvxHull(cpts, ax)

    obs.plot(ax)
    formatPlot(ax, '3D Elevated Convex Hull')


def speedSquared(c1, c2):
    fig, ax = plt.subplots()

    c1speed = c1.diff().normSquare()
    c2speed = c2.diff().normSquare()

    c1speed.plot(ax, color='C0', label=r'$||\dot \mathbf{C}^{[3]}(t)||^2$')
    c2speed.plot(ax, color='C1', label=r'$||\dot \mathbf{C}^{[4]}(t)||^2$')

    cpts1 = np.concatenate([[np.linspace(c1speed.t0, c1speed.tf, c1speed.deg+1)],
                            c1speed.cpts])
    cpts2 = np.concatenate([[np.linspace(c2speed.t0, c2speed.tf, c2speed.deg+1)],
                            c2speed.cpts])
    plotCvxHull2D(cpts1, ax)
    plotCvxHull2D(cpts2, ax)

    ax.set_title('3D Speed Squared')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Squared Speed $\left( \frac{m}{s} \right)^2$')
    ax.legend()


def accelSquared(c1, c2):
    fig, ax = plt.subplots()

    c1speed = c1.diff().diff().normSquare()
    c2speed = c2.diff().diff().normSquare()

    c1speed.plot(ax, color='C0', label=r'$||\ddot \mathbf{C}^{[3]}(t)||^2$')
    c2speed.plot(ax, color='C1', label=r'$||\ddot \mathbf{C}^{[4]}(t)||^2$')

    cpts1 = np.concatenate([[np.linspace(c1speed.t0, c1speed.tf, c1speed.deg+1)],
                            c1speed.cpts])
    cpts2 = np.concatenate([[np.linspace(c2speed.t0, c2speed.tf, c2speed.deg+1)],
                            c2speed.cpts])
    plotCvxHull2D(cpts1, ax)
    plotCvxHull2D(cpts2, ax)

    ax.set_title('3D Acceleration Squared')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Squared Acceleration $\left( \frac{m}{s^2} \right)^2$')
    ax.legend()


def distSqr(c1, c2, obs):
    fig, ax = plt.subplots()

    obsPoly = Bernstein(np.atleast_2d(obs.center).T*np.ones((3, c1.deg+1)), min(c1.t0, c2.t0), max(c1.tf, c2.tf))

    c1c2 = c1 - c2
    c1obs = c1 - obsPoly
    c2obs = c2 - obsPoly

    c1c2.normSquare().plot(ax, label=r'$||\mathbf{C}^{[3]}(t) - \mathbf{C}^{[4]}(t)||^2$')
    c1obs.normSquare().plot(ax, label=r'$||\mathbf{C}^{[3]}(t) - \mathbf{Obs}(t)||^2$')
    c2obs.normSquare().plot(ax, label=r'$||\mathbf{C}^{[4]}(t) - \mathbf{Obs}(t)||^2$')

    ax.set_title('3D Squared Distance Between Trajectories and Obstacle', wrap=True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Squared Distance $(m^2)$')
    ax.legend()


def _angularRate(bp):
    bpdot = bp.diff()
    bpddot = bpdot.diff()

    xdot = bpdot.x
    ydot = bpdot.y
    xddot = bpddot.x
    yddot = bpddot.y

    num = yddot*xdot - xddot*ydot
    den = xdot*xdot + ydot*ydot

    cpts = num.elev(DEG_ELEV).cpts / den.elev(DEG_ELEV).cpts
    wgts = den.elev(DEG_ELEV).cpts
    # cpts = num.cpts / den.cpts
    # wgts = den.cpts

    return RationalBernstein(cpts, wgts, bp.t0, bp.tf)


def plotCvxHull2D(cpts, ax):
    hull = ConvexHull(cpts.T)
    for simplex in hull.simplices:
        ax.plot(cpts[0, simplex], cpts[1, simplex], 'k:')


def saveFigs():
    import os
    # Create a Figures directory if it doesn't already exist
    if not os.path.isdir(FIG_DIR):
        os.mkdir(FIG_DIR)

    for i in plt.get_fignums():
        fig = plt.figure(i)
        ax = fig.get_axes()[0]
        title = ax.get_title()
        print(f'Saving figure {i} - {title}')

        ax.set_title('')
        plt.tight_layout()
        plt.draw()
        saveName = os.path.join(FIG_DIR, title.replace(' ', '_') + '.' + FIG_FORMAT)
        fig.savefig(saveName, format=FIG_FORMAT)
        ax.set_title(title)
        plt.draw()

    print('Done saving figures')


def plotCvxHull(cpts, ax):
    hull = ConvexHull(cpts.T)
    for simplex in hull.simplices:
        tri = Poly3DCollection((np.array([cpts[0, simplex], cpts[1, simplex], cpts[2, simplex]]).T))
        tri.set_alpha(0.5)
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)


if __name__ == '__main__':
    plt.close('all')
    setRCParams()

    obs = Sphere(4, 5, 5, 1, color='b')

    cpts3 = np.array([[7, 3, 1, 1, 3, 7],
                      [1, 2, 3, 8, 3, 5],
                      [0, 2, 1, 9, 8, 10]], dtype=float)
    cpts4 = np.array([[1, 1, 4, 4, 8, 8],
                      [5, 6, 9, 10, 8, 6],
                      [1, 1, 3, 5, 11, 6]], dtype=float)

    c3 = Bernstein(cpts3, t0=10, tf=20)
    c4 = Bernstein(cpts4, t0=10, tf=20)

    initPlot(c3, c4, obs)
    endPoints(c3, c4, obs)
    convexHull(c3, c4, obs)
    convexHullSplit(c3, c4, obs)
    convexHullElev(c3, c4, obs)
    speedSquared(c3, c4)
    accelSquared(c3, c4)
    distSqr(c3, c4, obs)

    if SAVE_FIG:
        saveFigs()

    plt.show()
    resetRCParams()
