# -*- coding: utf-8 -*-
"""
Custom Matplotlib colors

@author: Fabian Roth
"""

import matplotlib.colors
from cycler import cycler

# For testing
from matplotlib import pyplot as plt
import numpy as np

colorsets = []

colorsets.append({'rich-black-fogra-29':    '#001219ff',
                  'blue-sapphire':          '#005f73ff',
                  'viridian-green':         '#0a9396ff',
                  'middle-blue-green':      '#94d2bdff',
                  'medium-champagne':       '#e9d8a6ff',
                  'gamboge':                '#ee9b00ff',
                  'alloy-orange':           '#ca6702ff',
                  'rust':                   '#bb3e03ff',
                  'rufous':                 '#ae2012ff',
                  'ruby-red':               '#9b2226ff'})

colorsets.append({'lapis-lazuli':   '#0d5b96ff',
                  'viridian-green': '#06938cff',
                  'rust':           '#be3a0eff',
                  'tangerine':      '#f0870fff'})


colorlists = [list(c.values()) for c in colorsets]

def get_cycler(n=None, cs=0):
    cl = colorlists[cs]
    n = len(cl) if n is None else n
    if n > len(cl):
        print(f'Warning: {n} colors requested but only {len(cl)} in colorset')
    coler_indx = np.linspace(0, len(cl)-1, n, dtype=np.int32)
    return cycler(color=[cl[i] for i in coler_indx])

def get_cmap(cs=0):
    return matplotlib.colors.LinearSegmentedColormap.from_list("", colorlists[cs])

# %% Testing

# %% Scatter
x,y,c = zip(*np.random.rand(30,3)*4-2)
norm=plt.Normalize(-2,2)
plt.scatter(x, y, c=c, cmap=get_cmap(0), norm=norm)
plt.colorbar()
plt.show()

# %% Bar plot
data = [[1.3e-1, 2.5e-3], [1.3e-5, 1.5e-3], [1.8e-2, 1.6e2]]
x = np.arange(len(data))
data = np.array(data).T

labels = [r'only $f_2$', r'$f_2$ and $\nabla f_2$', r'only $\nabla f_2$']
width = 0.35  # the width of the bars

fig, ax = plt.subplots(dpi=600, figsize=(3,4))
ax.set_prop_cycle(get_cycler(2, 1))
ax.bar(x - width/2, data[0], width, label='Output loss', zorder=3)
ax.bar(x + width/2, data[1], width, label='Gradient loss', zorder=3)
ax.grid(zorder=0)
ax.set_xticks(x, labels)
#plt.grid(which='both')
plt.legend()
plt.yscale('log')
#plt.title('Evaluation Losses')
plt.show()

# %% Line Plot
fig, ax = plt.subplots(dpi=600)
ax.set_prop_cycle(get_cycler(cs=1))

x = np.linspace(-3, 3, 101)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = x*x

ax.plot(x, y1, label='sin')
ax.plot(x, y2, label='cos')
ax.plot(x, y3, label=r'$x^2$')

plt.legend()
plt.title('Some Functions')
plt.show()