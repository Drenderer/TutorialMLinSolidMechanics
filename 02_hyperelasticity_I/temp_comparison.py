# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 14:13:51 2022

@author: fabia

Not made to be executed. Just temp file for making fancy plots
"""

# %% Plot losses per load case
avg_losses = []
std_losses = []
load_case_names = [t['load_case'] for t in test]
for load_case in load_case_names:      # loop over test data to get load case names
    loss_aggregate = []
    for r in results:
        loss_aggregate.append(r['load_case_losses'][load_case])
        
    avg_loss = np.mean(loss_aggregate)
    std_loss = np.std(loss_aggregate)
    avg_losses.append(avg_loss)
    std_losses.append(std_loss)
    
avg_losses2 = []
std_losses2 = []
for load_case in load_case_names:      # loop over test data to get load case names
    loss_aggregate = []
    for r in no_weight_results:
        loss_aggregate.append(r['load_case_losses'][load_case])
        
    avg_loss = np.mean(loss_aggregate)
    std_loss = np.std(loss_aggregate)
    avg_losses2.append(avg_loss)
    std_losses2.append(std_loss)

x1 = np.array([0, 1, 2])
x2 = np.array([3.5, 4.5])
x = [0,1,2,3.5,4.5]
fig, ax = plt.subplots(dpi=600, figsize=(3,4))
ax.bar(x1-0.22, avg_losses2[:3], yerr=std_losses2[:3], color=dh.colors['b2'],
       align='center', ecolor=dh.colors['b4'], capsize=5,
       label='Training cases', width=0.44)
ax.bar(x1+0.22, avg_losses[:3], yerr=std_losses[:3], color=dh.colors['b3'],
       align='center', ecolor=dh.colors['b4'], capsize=5,
       label='Training cases weighted', width=0.44)
ax.bar(x2-0.22, avg_losses2[3:], yerr=std_losses2[3:], color=dh.colors['o3'],
       align='center', ecolor=dh.colors['o2'], capsize=5,
       label='Test cases', width=0.44)
ax.bar(x2+0.22, avg_losses[3:], yerr=std_losses[3:], color=dh.colors['o5'],
       align='center', ecolor=dh.colors['o3'], capsize=5,
       label='Test cases weighted', width=0.44)
ax.set_xticks(x, load_case_names, zorder=3)
ax.grid(zorder=0)
ax.set_axisbelow(True)
ax.set_title('Average loss per load case')
plt.xticks(rotation='vertical')
plt.yscale('log')
plt.legend(prop={'size': 7})
plt.show()