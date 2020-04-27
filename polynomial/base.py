#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:37:59 2020

@author: ckielasjensen
"""

from collections import defaultdict

import numpy as np


class Base:
    """Parent class used for storing Bernstein parameters

    :param cpts: Control points used to define the polynomial. The degree of
        the polynomial is equal to the number of columns - 1. The dimension of
        the polynomial is equal to the number of rows.
    :type cpts: numpy.ndarray or None
    :param t0: Initial time of the Bezier curve trajectory.
    :type t0: float
    :param tf: Final time of the Bezier curve trajectory.
    :type tf: float
    """
    splitCache = defaultdict(dict)
    elevMatCache = defaultdict(dict)
    prodMatCache = defaultdict(dict)
    diffMatCache = defaultdict(dict)
    coefCache = dict()

    def __init__(self, cpts=None, t0=0.0, tf=1.0):
        self._curve = None
        self._t0 = float(t0)
        self._tf = float(tf)

        if cpts is not None:
            if cpts.ndim == 1:
                self._cpts = np.atleast_2d(cpts)
                self._dim = 1
                self._deg = cpts.size - 1
            else:
                self._cpts = cpts
                self._dim = self._cpts.shape[0]
                self._deg = self._cpts.shape[1] - 1
        else:
            self._dim = None
            self._deg = None

    @property
    def cpts(self):
        return self._cpts

    @cpts.setter
    def cpts(self, value):
        self._curve = None

        if (isinstance(value, np.ndarray) and
                value.ndim == 2 and
                value.dtype == 'float64'):
            newCpts = value
        else:
            newCpts = np.array(value, ndmin=2, dtype=np.float64)

        self._dim = newCpts.shape[0]
        self._deg = newCpts.shape[1] - 1
        self._cpts = newCpts

    @property
    def deg(self):
        return self._deg

    @property
    def degree(self):
        return self._deg

    @property
    def dim(self):
        return self._dim

    @property
    def dimension(self):
        return self._dim

    @property
    def t0(self):
        return self._t0

    @t0.setter
    def t0(self, value):
        self._t0 = float(value)

    @property
    def tf(self):
        return self._tf

    @tf.setter
    def tf(self, value):
        self._tf = float(value)
