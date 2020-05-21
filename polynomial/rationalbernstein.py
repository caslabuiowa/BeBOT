#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:38:24 2020

@author: ckielasjensen
"""

import matplotlib.pyplot as plt
import numpy as np

from polynomial.base import Base
# from bernstein import Bernstein

class RationalBernstein(Base):
    """
    """
    def __init__(self, cpts=None, weights=None, t0=0.0, tf=1.0):
        super().__init__(cpts=cpts, t0=t0, tf=tf)
        self._weights = np.array(weights, ndmin=2)

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        num = Bernstein(self.cpts, t0=self.t0, tf=self.tf)
        den = Bernstein(self.weights, t0=self.t0, tf=self.tf)

        if self.dim == 1:
            plt.plot(num.tau, num.curve / den.curve)

        elif self.dim == 2:
            ax.plot(num.curve[0, :]/den.curve[0, :], num.cur)
