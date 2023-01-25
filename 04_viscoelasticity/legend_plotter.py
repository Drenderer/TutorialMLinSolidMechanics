"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 4: Viscoelasticity
==================
Authors: Fabian Roth
         

Plotting nice legends
01/2023
"""

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

plt.figure(dpi=500)

# legend=[r'$\bf{A}$: $\omega = 1$, $A = 1$', 
#         r'$\bf{B}$: $\omega = 1$, $A = 2$',
#         r'$\bf{C}$: $\omega = 2$, $A = 3$']
legend=[r'$\bf{A}$: $\omega = 1$, $A = 0.5$', 
        r'$\bf{B}$: $\omega = 2$, $A = 6$']

handles = []
for l in legend:
    h = Rectangle((0,0), 1, 1, fill=False, edgecolor='none',
                                 visible=False)
    handles.append(h)

plt.legend(handles, legend, loc='center')
