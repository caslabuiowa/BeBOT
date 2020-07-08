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


class RationalBernstein(Base):
    def __init__(self, cpts=None, weights=None, t0=0.0, tf=1.0):
        assert cpts.shape == weights.shape, 'The shape of cpts and weights should be the same.'
        super().__init__(cpts=cpts, t0=t0, tf=tf)
        self._wgts = np.array(weights, ndmin=2)

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
