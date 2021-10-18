#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 11:47:09 2020

@author: ckielasjensen
"""
import os

import matplotlib.pyplot as plt


def saveFigs(figDir='Figures', figFormat='svg', dpi=1200):
    # Create a Figures directory if it doesn't already exist
    if not os.path.isdir(figDir):
        os.mkdir(figDir)

    for i in plt.get_fignums():
        fig = plt.figure(i)
        ax = fig.get_axes()[0]
        title = ax.get_title()
        if title == '':
            print('[!] Warning: No figure title, figure will be saved without a name.')
        print(f'Saving figure {i} - {title}')

        ax.set_title('')
        plt.tight_layout()
        plt.draw()
        saveName = os.path.join(figDir, title.replace(' ', '_') + '.' + figFormat)
        fig.savefig(saveName, format=figFormat, dpi=dpi)
        ax.set_title(title)
        plt.draw()

    print('Done saving figures')


if __name__ == '__main__':
    pass
