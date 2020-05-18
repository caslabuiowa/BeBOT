#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 00:31:46 2020

@author: ckielasjensen
"""

import matplotlib.pyplot as plt

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
            'lines.markersize': 18
            })

# Reset the matplotlib parameters
#    plt.rcParams.update(plt.rcParamsDefault)
