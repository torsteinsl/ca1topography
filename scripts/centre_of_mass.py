# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 13:27:27 2022

@author: torstsl


This script takes a session and calculates the centre of mass in each ratemap
for that session. The centres of masses are then compared between A-B and A-A'.
The scipts loads the place cell ID and does calculations only on place cells. 
Lastly, the differences in centres of mass is plotted with 95 % CI and with 
a violin plot fitting a distribution to the underlying data. 

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
from scipy.ndimage import center_of_mass
# from src.loading_data import load_session_dict
from math import dist

#%% Load data

# session_dict = load_session_dict()
session_dict = pickle.load(open(r'N:\axon2pmini\Recordings\102124\280922/session_dict.pickle','rb'))

session_ratemaps = session_dict['Ratemaps']

place_cell_dict = session_dict['Placecell']
place_cell = np.unique(np.concatenate([place_cell_dict['NAT0'][0],place_cell_dict['NAT1'][0],place_cell_dict['NAT2'][0]]))

#%% Calculate all centres of mass and distance between centres

centre_mass_dict = {}
dist_AB = []
dist_AA = []

for neuron in place_cell: 
    centre_mass_dict['N'+str(neuron)] = np.zeros((3,2))
    for sess in range(0,len(session_ratemaps)):
        ratemap_PC = session_ratemaps['dfNAT'+str(sess)]['N'+str(neuron)]
        centreMass = center_of_mass(ratemap_PC) # Coordinates: y, x
        
        centre_mass_dict['N'+str(neuron)][sess,:]= np.array(centreMass)
        
        if neuron == 5:
           fig, ax = plt.subplots()          
           ax.imshow(ratemap_PC,cmap='viridis')
           ax.scatter(centreMass[1],centreMass[0],marker='o', color='crimson', s = 30)  # y,x
           ax.set_title('Tuning map and centre of mass (N5)')
           plt.axis('off')
           
    dist_AB.append(dist(centre_mass_dict['N'+str(neuron)][0,:],centre_mass_dict['N'+str(neuron)][1,:])) # Coordinates are x,y
    dist_AA.append(dist(centre_mass_dict['N'+str(neuron)][0,:],centre_mass_dict['N'+str(neuron)][2,:])) # Coordinates are x,y 

#%% Plot an example for neuron 5
fig, ax = plt.subplots(1,3)
fig.suptitle('Remapping for N5',fontsize=20)
for ii in range(0,len(session_ratemaps)):
    ratemap_PC = session_ratemaps['dfNAT'+str(ii)]['N5']
    centreMass = center_of_mass(ratemap_PC)
    ax[ii].grid(b=None)
    ax[ii].axis('off')
    ax[ii].imshow(ratemap_PC, cmap='viridis')
    ax[ii].plot(centreMass[1],centreMass[0],marker='o', color='crimson', markersize = 7.5)  # y,x
    ax[0].set_title('A')
    ax[1].set_title('B')
    ax[2].set_title('A\'')
    plt.tight_layout()
#%% Plot the distances

dist_list = dist_AA.copy()+dist_AB.copy()
box_list = ['AA\'']*len(dist_AA)+['AB']*len(dist_AB)

distances = np.array(dist_list)*2.5 # Correct the binning size (2.5 cm/bin)

dfCM = pd.DataFrame({'Distance':distances,'Box':box_list})

fig, ax = plt.subplots(1,2,sharey=False)
plt.suptitle('Distance in centre of mass for place cells, n = '+str(len(place_cell)))
ax[0] = sns.barplot(x='Box',y='Distance',data=dfCM,ax=ax[0],estimator=np.nanmean, ci=95, errwidth = 1.5,capsize=0.05,palette='viridis')
ax[1] = sns.violinplot(x='Box',y='Distance',data=dfCM,ax=ax[1],split=False,palette='viridis')
ax[1].legend([],[],frameon=False)  

ax[0].set(xlabel=None,ylabel='Distance (cm)')
ax[1].set(xlabel=None, ylabel=None)

plt.tight_layout()
