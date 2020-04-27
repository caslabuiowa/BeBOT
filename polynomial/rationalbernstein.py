#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:38:24 2020

@author: ckielasjensen
"""

import numpy as np

from polynomial.base import Base

class RationalBernstein(Base):
    """
    """
    def __init__(self, cpts=None, weights=None, t0=0.0, tf=1.0):
        super().__init__(cpts=cpts, t0=t0, tf=tf)
        self._weights = np.array(weights, ndmin=2)
