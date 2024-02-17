# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:40:59 2023

@author: torstsl
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr

#%% Load data

scale = 1.1801 # Pixels/µm --> 1 pixel = 0.847.. µm 
binning = 2.5 # cm/spatial bin in rate map

palette = list(sns.color_palette("viridis", 256).as_hex())
contrast = list(sns.color_palette('OrRd', 256).as_hex())

sessionName = ['Open field','Object','Object moved', 'Open field']


# Load data paths
results_folder = r'C:\Users\torstsl\Projects\axon2pmini\results'

with open(results_folder+'/sessions_overview.txt') as f:
    sessions = f.read().splitlines() 
f.close()

#%% Load primary session

session = sessions[7]

session_dict = pickle.load(open(session+'\session_dict.pickle','rb'))
print('Successfully loaded session_dict from: '+str(session))

nSessions = int(session_dict['ExperimentInformation']['Session'])

placecell_dict = session_dict['Placecell'] 
placecells = np.unique(np.concatenate([placecell_dict['NAT0'][0], placecell_dict['NAT1'][0], placecell_dict['NAT2'][0]]))

PCs = placecell_dict['NAT0'][0]
nPC = len(PCs)

rmapA = session_dict['Ratemaps']['dfNAT0'] 
rmapB = session_dict['Ratemaps']['dfNAT1'] 
rmapA2 = session_dict['Ratemaps']['dfNAT2'] 

debugging = False

if debugging == True:
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(rmapA['N1'])
    ax[1].imshow(rmapB['N1'])
    ax[2].imshow(rmapA2['N1'])
    plt.tight_layout()
    
if debugging == True:
    fig, ax = plt.subplots(1,4)
    for x in range(4):
        rmap = np.rot90(rmapB['N1'], x)
        ax[x].imshow(rmap)
        ax[x].set_title(str(90*x)+' deg')
    plt.tight_layout()
    
#%% Correlate at different rotations

rot0 = np.full(nPC, np.nan)
rot90 = np.full(nPC, np.nan)
rot180 = np.full(nPC, np.nan)
rot270 = np.full(nPC, np.nan)
A2A = np.full(nPC, np.nan)

for cellNo, PC in enumerate(placecells): 
    mapA = rmapA['N'+str(PC)].flatten()
    mapB = rmapB['N'+str(PC)]
    mapA2 = rmapA2['N'+str(PC)].flatten()
    
    rot0[cellNo] = pearsonr(mapA, mapB.flatten())[0]
    rot90[cellNo] = pearsonr(mapA, np.rot90(mapB,1).flatten())[0]
    rot180[cellNo] = pearsonr(mapA, np.rot90(mapB,2).flatten())[0]
    rot270[cellNo] = pearsonr(mapA, np.rot90(mapB,3).flatten())[0]
    A2A[cellNo] = pearsonr(mapA, mapA2)[0]

