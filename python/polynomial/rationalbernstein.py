#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:38:24 2020

@author: ckielasjensen
"""

import matplotlib.pyplot as plt
from numba import njit
import numpy as np
from scipy.special import binom

from polynomial.base import Base

# TODO de Cast for rational BP
# TODO elevate for rational BP
# TODO elevate degree if weight of rBP is negative
# TODO function to normalize rBP to canonical form (w0 = wn = 1)
# TODO add more functionality into the Base class to avoid rewriting code for both BP and rBPs


class RationalBernstein(Base):
    def __init__(self, cpts=None, weights=None, t0=0.0, tf=1.0):
        assert cpts.shape == weights.shape, 'The shape of cpts and weights should be the same.'
        super().__init__(cpts=cpts, t0=t0, tf=tf)
        # self._wgts = np.array(weights, dtype=float, ndmin=2)

        if weights is not None:
            assert type(weights) is np.ndarray, 'Weights must be a numpy array'
            assert weights.dtype == 'float64', 'Weights dtype must be numpy float64'
            if self.dim == 1:
                self._wgts = np.atleast_2d(weights)
            else:
                self._wgts = weights
            assert self._wgts.shape == self._cpts.shape, 'Weights and control points must be the same shape.'
        elif cpts is not None:
            raise TypeError('Rational Bernstein polynomial requires weights')

    @property
    def wgts(self):
        return self._wgts

    @wgts.setter
    def wgts(self, value):
        self._curve = None

        if (isinstance(value, np.ndarray) and
                value.ndim == 2 and
                value.dtype == 'float64'):
            newWgts = value
        else:
            newWgts = np.array(value, ndmin=2, dtype=np.float64)

        self._dim = newWgts.shape[0]
        self._deg = newWgts.shape[1] - 1
        self._wgts = newWgts

    def plot(self, ax=None, npts=1001, showCpts=True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        if self.dim == 1:
            c = self.curve(npts=npts)
            T = np.linspace(self._t0, self._tf, npts)
            ax.plot(T, c.squeeze(), **kwargs)

            if showCpts:
                ax.plot(np.linspace(self.t0, self.tf, self.deg+1),
                        self.cpts.squeeze(), '.--')

        # TODO plot higher dimensions
        else:
            raise Exception('Plotting for higher dimensions is not supported yet.')

    def curve(self, t=None, npts=1001):
        if t is not None:
            return _ratBernPolyPt(self._cpts, self._wgts, t, self._t0, self._tf)

        elif self._curve is None or self._curve.shape[1] != npts:
            self._curve = _ratBernPoly(self._cpts, self._wgts, npts)

        return self._curve

    def min(self, dim=0, globMin=np.inf, tol=1e-6):
        """Returns the minimum value of the Bernstein polynomialin a single
        dimension

        Finds the minimum value of the Bernstein polynomial. This is done by
        first checking the first and last control points since the first and
        last point lie on the curve. If the first or last control point is not
        the minimum value, the curve is split at the lowest control point. The
        new minimum value is then defined as the lowest control point of the
        two new curves. This continues until the difference between the new
        minimum and old minimum values is within the desired tolerance.

        :param dim: Which dimension to return the minimum of.
        :type dim: int
        :param tol: Tolerance of the minimum value.
        :type tol: float
        :param maxIter: Maximum number of iterations to search for the minimum.
        :type maxIter: int
        :return: Minimum value of the Bernstein polynomial. None if maximum
            iterations is met.
        :rtype: float or None
        """
        ub = self.cpts[dim, (0, -1)].min()
        if ub < globMin:
            globMin = ub

        minIdx = self.cpts[dim, :].argmin()
        lb = self.cpts[dim, minIdx]

        # Prune if the global min is less than the lower bound
        if globMin < lb:
            return globMin

        # If we are within the desired tolerance, return
        if ub - lb < tol:
            return globMin

        # Otherwise split and continue
        else:
            tdiv = (minIdx/self.deg)*(self.tf - self.t0) + self.t0
            c1, c2 = self.split(tdiv)
            c1min = c1.min(dim=dim, globMin=globMin, tol=tol)
            c2min = c2.min(dim=dim, globMin=globMin, tol=tol)

            return min(c1min, c2min)

    def split(self, tDiv):
        """Split the rational Bernstein polynomial in two at point tDiv.

        Paper Reference: Property 8: Rational Recursive Algorithm

        :param tDiv: Point at which to split the curve
        :type tDiv: float
        :return: Tuple of curves. One before the split point and one after.
        :rtype: tuple(Bernstein, Bernstein)
        """
        cpts1 = []
        cpts2 = []
        wgts1 = []
        wgts2 = []

        if np.isnan(tDiv):
            print('[!] Warning, tDiv is {}, changing to 0.'.format(tDiv))
            tDiv = 0

        for d in range(self.dim):
            left, right = _ratDeCasteljauSplit(self.cpts[d, :], self.wgts[d, :], tDiv=tDiv, t0=self.t0, tf=self.tf)
            cpts1.append([left[0]])
            cpts2.append([right[0][::-1]])
            wgts1.append([left[1]])
            wgts2.append([right[1][::-1]])

        c1 = RationalBernstein(cpts=np.concatenate(cpts1), weights=np.concatenate(wgts1), t0=self.t0, tf=tDiv)
        c2 = RationalBernstein(cpts=np.concatenate(cpts2), weights=np.concatenate(wgts2), t0=tDiv, tf=self.tf)

        return c1, c2


def _ratBernPoly(cpts, wgts, npts):
    dim = cpts.shape[0]
    n = cpts.shape[1] - 1
    T = np.linspace(0, 1, npts)
    num = np.empty((dim, npts))
    den = np.empty((dim, npts))
    B = _bernBasisMat(n)

    for i, t in enumerate(T):
        powBasis = np.power(t, range(n+1))
        basis = powBasis.dot(B)
        num[:, i] = basis.dot((cpts*wgts).T)
        den[:, i] = basis.dot(wgts.T)

    return num / den


def _ratBernPolyPt(cpts, wgts, t, t0, tf):
    n = cpts.shape[1] - 1
    tau = _t2tau(t, t0, tf)
    B = _bernBasisMat(n)

    powBasis = np.power(tau, range(n+1))
    basis = powBasis.dot(B)
    num = basis.dot((cpts*wgts).T)
    den = basis.dot(wgts.T)

    return num / den


def _ratDeCasteljau(cpts, wgts, t0=0., tf=1.):
    pass


@njit(cache=True)
def _t2tau(t, t0, tf):
    return (t-t0) / (tf-t0)


def _bernBasis(n, i, t, t0=0.0, tf=1.0):
    return binom(n, i) * (t-t0)**i * (tf - t)**(n-i) / (tf - t0)**n


def _bernBasisMat(n):
    b = np.zeros((n+1, n+1))

    for k in np.arange(0, n+1):
        for i in np.arange(k, n+1):
            b[i, k] = (-1)**(i-k) * binom(n, i) * binom(i, k)

    return b


@njit(cache=True)
def _ratDeCasteljauSplit(cpts, wgts, tDiv, t0=0.0, tf=1.0):
    """Use the de Casteljau algorithm to split the rational Bernstein polynomial at tDiv.

    This function is similar to the de Casteljau curve function but instead of
    drawing a curve, it returns two sets of control points which define the
    curve to the left and to the right of the split point, tDiv.

    Paper Reference: Property 5: The de Casteljau Algorithm

    :param cpts: Control points defining the 1D Bernstein polynomial.
    :type cpts: numpy.ndarray(dtype=numpy.float64)
    :param tDiv: Point at which to divide the curve.
    :type tDiv: float
    :param tf: Final tau value for the 1D curve. Default is 1.0. Note that the
        Bernstein polynomial is defined on the range of [0, tf].
    :type tf: float
    :return: Returns a tuple of numpy arrays. The zeroth element is the
        control points defining the curve to the left of tDiv. The first
        element is the control points defining the curve to the right of tDiv.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    tDiv = (tDiv-t0) / (tf-t0)
    cptsLeft = np.zeros(cpts.size)
    cptsRight = np.zeros(cpts.size)
    wgtsLeft = np.zeros(wgts.size)
    wgtsRight = np.zeros(wgts.size)
    idx = 0

    newCpts = cpts.copy()
    cptsTemp = cpts.copy()
    newWgts = wgts.copy()
    wgtsTemp = wgts.copy()
    while newCpts.size > 1:
        cptsLeft[idx] = cptsTemp[0]
        cptsRight[idx] = cptsTemp[-1]
        wgtsLeft[idx] = wgtsTemp[0]
        wgtsRight[idx] = wgtsTemp[-1]
        idx += 1

        cptsTemp = np.empty(newCpts.size-1)
        wgtsTemp = np.empty(newWgts.size-1)
        for i in range(cptsTemp.size):
            wgtsTemp[i] = (1-tDiv)*newWgts[i] + tDiv*newWgts[i+1]
            a = newWgts[i] / wgtsTemp[i]
            b = newWgts[i+1] / wgtsTemp[i]
            cptsTemp[i] = (1-tDiv)*a*newCpts[i] + tDiv*b*newCpts[i+1]

        newCpts = cptsTemp.copy()
        newWgts = wgtsTemp.copy()

    cptsLeft[-1] = cptsRight[-1] = newCpts[0]
    wgtsLeft[-1] = wgtsRight[-1] = newWgts[0]

    return (cptsLeft, wgtsLeft), (cptsRight, wgtsRight)


if __name__ == '__main__':
    plt.close('all')
    npts = 1001
    cpts = np.array([[0, 1, 2, 3, 4, 5],
                     [3, 7, 5, 2, 6, 2]], dtype=float)
    wgts = np.array([[0.5, 1, 10, 1, -1, 3],
                     [0.5, 1, 10, 1, -1, 3]], dtype=float)

    ratc = _ratBernPoly(cpts, wgts, npts)

    plt.plot(ratc[0, :], ratc[1, :])
    plt.plot(cpts[0, :], cpts[1, :], '.--')

    plt.show()
