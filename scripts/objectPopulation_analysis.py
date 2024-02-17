# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 17:35:00 2024

@author: torstsl

Object population analysis

"""

#%%
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, circmean

#%% Define constants and variables, load path to all object sessions

scale = 1.1801 # Pixels/µm --> 1 pixel = 0.847.. µm 
binning = 2.5 # cm/spatial bin in rate map

palette = list(sns.color_palette("viridis", 256).as_hex())
contrast = list(sns.color_palette('OrRd', 256).as_hex())

sessionName = ['Open field','Object','Object moved', 'Open field']

# Load data paths
results_folder = r'C:\Users\torstsl\Projects\axon2pmini\results'

with open(results_folder+'/ObjectSessions_overview.txt') as f:
    objectSessions = f.read().splitlines() 
f.close()

# Load results
objectResults_dict = pickle.load(open(results_folder+'\objectResults_dict.pickle','rb'))

#%% Calculate the angukar mean for the object cells by anatomical distance

distances = np.linspace(15,100,18, dtype = int)

objectPop, objectPopShuffle = {}, {}
for d in distances: 
    objectPop[str(d)] = []
    objectPopShuffle[str(d)] = []
    
for key in objectResults_dict.keys():
    oCells = objectResults_dict[key]['angle2object'].size
    arrAnat = objectResults_dict[key]['anatomicalDiff']
    arrAng = objectResults_dict[key]['angularDiff']
    
    # Create a 2D array with zeros
    arrAnat2d = np.zeros([oCells, oCells])
    arrAng2d = np.zeros([oCells, oCells])
    
    # Fill the upper triangular part with values
    arrAnat2d[np.triu_indices(oCells, k=1)] = arrAnat
    arrAng2d[np.triu_indices(oCells, k=1)] = arrAng

    # Fill the lower triangular part with values    
    arrAnat2d += arrAnat2d.T
    arrAng2d += arrAng2d.T
    
    # Calculate the angular mean of the closest neighbours
    for d in distances: 
         for cellNo in range(oCells):
            anatBool = (arrAnat2d[cellNo,:] < d) & (arrAnat2d[cellNo,:] > 0)
            angles = np.radians(arrAng2d[cellNo, anatBool])
            objectPop[str(d)].append(np.rad2deg(circmean(angles)))

            # Shuffle the indecies to make a comparison
            if len(angles) > 0:
                        
                shuffler = np.arange(oCells)
                shuffler = np.delete(shuffler,cellNo) # Remove self
                
                for shuffleNo in range(25):
                    
                    shuffleIdx = np.random.choice(shuffler, len(angles), replace = False) # Shuffle idx
                
                    anglesShuffle = np.radians(arrAng2d[cellNo, shuffleIdx])
                    objectPopShuffle[str(d)].append(np.rad2deg(circmean(anglesShuffle)))
            
# Concatenate the data
for d in distances: 
    arr = np.array(objectPop[str(d)])
    noNanIdx = ~np.isnan(arr)
    objectPop[str(d)] = arr[noNanIdx]
    
    arrS = np.array(objectPopShuffle[str(d)])
    noNanIdxS = ~np.isnan(arrS)
    objectPopShuffle[str(d)] = arrS[noNanIdxS]
    
#%% Plot it
dfPlot = pd.DataFrame({key: pd.Series((value)) for key, value in objectPop.items()})
dfShuffle = pd.DataFrame({key: pd.Series((value)) for key, value in objectPopShuffle.items()})

# Box plot
fig, ax = plt.subplots()
sns.boxplot(data = dfPlot, color = palette[140], showfliers=False)
ax.set_title('Difference in place field distance from A to B')
ax.set_xlabel('Anatomical distance (µm)')
ax.set_ylabel('Angule from object (deg)')
ax.spines[['top', 'right']].set_visible(False)

# Point plot with IQR
lower, upper = np.nanpercentile(dfShuffle, 25, axis = 0), np.nanpercentile(dfShuffle, 75, axis = 0)

fig, ax = plt.subplots(figsize=(5.,4.5))
sns.pointplot(data = dfPlot.melt(), estimator = np.nanmedian, x = 'variable', y = 'value', color = palette[120], capsize = 0.75)
ax.fill_between(np.arange(18), lower, upper, alpha = 0.25, color = palette[120], edgecolor = None)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel('Angle to object (deg)')
ax.set_xlabel('Anatomical distance (µm)')
ax.set_title('Median angle to object')
ax.set_xticks(np.linspace(-3,17,21))
ax.set_xticklabels(np.linspace(0,100,21, dtype = int))
ax.set_ylim([0,150])
plt.tight_layout()

plt.savefig('N:/axon2pmini/Article - Topography/Figures/Figure 6/FigX_pop.svg', format = 'svg')   

# Statistics
all_arrays = dfPlot.to_numpy() 
kruskal(*[col for col in all_arrays.T], nan_policy='omit')  # Transpose to handle columns
