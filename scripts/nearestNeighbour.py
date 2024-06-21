# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:01:53 2024

@author: torstsl

Nearest neigbour analysis (not including Moran)



"""

import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from scipy.stats import pearsonr, ttest_ind, ranksums
from src.cell_activity_analysis import get_placefields_distance, cohen_d
from statsmodels.stats.weightstats import ttost_ind

#%% Get the cell centre and distance between all cells

def get_cell_data(session_dict):
    
    nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)
    
    maxproj = session_dict['ExperimentInformation']['ImagingOptions']['max_proj']
    
    yrange = session_dict['ExperimentInformation']['ImagingOptions']['yrange'] # Valid y-range used for cell detection
    xrange = session_dict['ExperimentInformation']['ImagingOptions']['xrange'] # Valid x-range used for cell detection
    
    cell_centre = np.full([nCells,2],np.nan) 
    
    for cellNo in range(nCells):
        cell_centre[cellNo] = session_dict['ExperimentInformation']['CellStat'][cellNo]['med'][::-1] # Get cell centre, xy (anatomical)
        cell_centre[cellNo][0] -= xrange[0]
        cell_centre[cellNo][1] -= yrange[0]
    
    # Calculate distance matix
    distanceMtx =  np.linalg.norm(cell_centre - cell_centre[:,None], axis = -1)
    distanceMtx[np.diag_indices_from(distanceMtx)] = np.nan # Put distance to self as NaN

    return maxproj, cell_centre, distanceMtx

#%% Calculate the correlation between nearest neighbours 

def calc_correlations(session_dict, distanceMtx):
        
    near_corr = {}
    near_control = {}
    far_corr = {}
    
    anat_distanceNear, anat_distanceFar = {}, {}
    
    nSessions = len(session_dict['dfNAT'])
    placecell_dict = session_dict['Placecell']
    
    for sessionNo in range(nSessions):
        key = 'NAT'+str(sessionNo)
        
        pcs = placecell_dict[key][0] # NATEX ID
        
        near_corr[key] = np.full([len(pcs)], np.nan)
        near_control[key] = np.full([len(pcs), len(pcs)], np.nan)
        far_corr[key] = np.full([len(pcs)], np.nan)
        
        anat_distanceNear[key] = np.full([len(pcs)], np.nan)
        anat_distanceFar[key] = np.full([len(pcs)], np.nan)
        
        # Correlate each PC to its nearest neighbouring PC
        for no, pCell in enumerate(pcs): # NATEX ID
            rPC = session_dict['Ratemaps']['df'+key]['N'+str(pCell)].flatten()  
            
            # Make submatrix of just place cells
            rows, cols = pcs-1, pcs-1 # From NATEX to Pythonic
            pcMtx = distanceMtx[np.ix_(rows, cols)]
            
            # Get the nearest neighbour
            nearestDist = np.nanmin(pcMtx[no, :]) # Pythonic ID
            temp_idx = np.where(pcMtx[no,:] == nearestDist)[0][0] # Pythonic ID
            nearestPC = pcs[temp_idx] # From Pythonic to NATEX
            
            rNearest = session_dict['Ratemaps']['df'+key]['N'+str(nearestPC)].flatten()
            
            near_corr[key][no] = pearsonr(rPC, rNearest)[0]
            anat_distanceNear[key][no] = nearestDist
            
            # Get the most far off place cell
            farOff = np.nanmax(pcMtx[no, :]) # Pythonic ID
            temp_idx = np.where(pcMtx[no,:] == farOff)[0][0] # Pythonic ID
            farPC = pcs[temp_idx] # From Pythonic to NATEX
            
            rFar = session_dict['Ratemaps']['df'+key]['N'+str(farPC)].flatten()
            
            far_corr[key][no] = pearsonr(rPC, rFar)[0]
            anat_distanceFar[key][no] = farOff
          
            # Control by comparing all PCs to each other
            for num, PC in enumerate(pcs):
                if num > no:
                    rControl = session_dict['Ratemaps']['df'+key]['N'+str(PC)].flatten()
                    near_control[key][no, num] = pearsonr(rPC, rControl)[0]
            
            # Copy to make a symmetric matrix, autocorrelation is set to NaN (computes faster)
            near_control[key] = np.triu(near_control[key]) + np.triu(near_control[key], 1).T
            
    return near_corr, near_control, far_corr, anat_distanceNear, anat_distanceFar

#%% Calculate the field distance between nearest neighbours 

def calc_fielddist(session_dict, distanceMtx, field_distance):
    
    nSessions = len(session_dict['dfNAT'])
    placecell_dict = session_dict['Placecell']
    
    near_dist = {}
    far_dist = {}
    
    for sessionNo in range(nSessions):
        key = 'NAT'+str(sessionNo)
        
        pcs = placecell_dict[key][0] # NATEX ID
        near_dist[key] = np.full([len(pcs)], np.nan)
        far_dist[key] = np.full([len(pcs)], np.nan)
        
        # Find distance from each PC to its nearest neighbouring PC
        for no, pCell in enumerate(pcs): # NATEX ID  
            
            # Make submatrix of just place cells
            rows, cols = pcs-1, pcs-1 # From NATEX to Pythonic
            pcMtx = distanceMtx[np.ix_(rows, cols)]
            
            # Get the nearest neighbour
            nearestDist = np.nanmin(pcMtx[no, :]) # Pythonic ID
            nearestPC = np.where(pcMtx[no,:] == nearestDist)[0][0] # Pythonic ID
            near_dist[key][no] = field_distance[key][no, nearestPC]
            
            # Get the most far off place cell
            farOffDist = np.nanmax(pcMtx[no, :]) # Pythonic ID
            farPC = np.where(pcMtx[no,:] == farOffDist)[0][0] # Pythonic ID
            far_dist[key][no] = field_distance[key][no, farPC]

    return near_dist, far_dist

#%% Define constants
sessionName = ['A','B','A\'']
NATs = ['NAT0', 'NAT1', 'NAT2']

scale = 1.1801 # Pixels/µm --> 1 pixel = 0.847.. µm 
binning = 2.5
minPeak = 0.2

palette = list(sns.color_palette("viridis", 256).as_hex())
contrast = list(sns.color_palette('OrRd', 256).as_hex())

#%% Loop over the sessions

results_folder = r'C:\Users\torstsl\Projects\axon2pmini\results'

with open(results_folder+'/sessions_overview.txt') as f:
    sessions = f.read().splitlines() 
f.close()

nearestNeighbour = {} 

for session in sessions:
    
   # Load session_dict
   session_dict = pickle.load(open(session+'\session_dict.pickle','rb'))
   print('Successfully loaded session_dict from: '+str(session))
   
   key = session_dict['Animal_ID'] + '-' + session_dict['Date']
   
   # Get cell centre and distance between them
   maxproj, cell_centre, distanceMtx = get_cell_data(session_dict)
      
   # Calculate the correaltion between the nearest neighbour and all PCs
   near_corr, near_control, far_corr, anat_distanceNear, anat_distanceFar = calc_correlations(session_dict, distanceMtx)
   
   # Get the place field data
   placefield, placefield_coords, field_distance = get_placefields_distance(session_dict)

   # Calculate the field distance between the nearest neoghbours (field_distance above is all)
   near_dist, far_dist = calc_fielddist(session_dict, distanceMtx, field_distance)

   # Put the results in a dict
   nearestNeighbour[key] = {'maxproj': maxproj,
                            'cell_centre': cell_centre,
                            'distanceMtx': distanceMtx,
                            'near_corr': near_corr,
                            'near_control': near_control,
                            'far_corr': far_corr,
                            'anat_distanceNear': anat_distanceNear,
                            'anat_distanceFar': anat_distanceFar,
                            'field_distance': field_distance, 
                            'near_dist': near_dist,
                            'far_dist': far_dist
                            }
    
   # Store the output
   with open(results_folder+'/nearestNeighbour.pickle','wb') as handle:
       pickle.dump(nearestNeighbour, handle, protocol=pickle.HIGHEST_PROTOCOL)
   print('Successfully saved results_dict in '+ results_folder)

#%% Load the pickle if you already ran the analysis
nearestNeighbour = pickle.load(open(r'C:\Users\torstsl\Projects\axon2pmini\results\nearestNeighbour.pickle','rb'))

#%% Prepare the data

nearCorr, nearCorrCtrl= [], []
nearDist, nearDistCtrl = [], []
farCorr, farDist = [], []

for key in nearestNeighbour.keys():
    for NAT in NATs[0:2]:
        
        nearCorr.append(nearestNeighbour[key]['near_corr'][NAT])
        nearDist.append(nearestNeighbour[key]['near_dist'][NAT])
        
        farCorr.append(nearestNeighbour[key]['far_corr'][NAT])
        farDist.append(nearestNeighbour[key]['far_dist'][NAT])
        
        tempCorr = nearestNeighbour[key]['near_control'][NAT]
        idx = np.triu_indices_from(tempCorr, k=1)
        tempCorr = tempCorr[idx[0], idx[1]]
        nearCorrCtrl.append(tempCorr)
        
        tempDist = nearestNeighbour[key]['field_distance'][NAT]
        idx = np.triu_indices_from(tempDist, k=1)
        tempDist = tempDist[idx[0], idx[1]]
        nearDistCtrl.append(tempDist)
        
#%% Plot the data and stats

dataCorr, controlCorr = np.concatenate(nearCorr), np.concatenate(nearCorrCtrl)

# Create control size matched per sample
c = []
for num in range(len(nearCorr)):
    c.append(np.random.choice(nearCorrCtrl[num], nearCorr[num].size, replace=False))

ctrl = np.concatenate(c)

res = ttost_ind(dataCorr, ctrl, -0.05, 0.05)


sns.boxplot(data= [dataCorr, ctrl])

# Stats
np.median(dataCorr), np.median(ctrl)
np.mean(dataCorr), np.mean(ctrl)
ttest_ind(dataCorr, ctrl, nan_policy = 'omit')
ranksums(dataCorr, ctrl)
cohen_d(dataCorr, ctrl)
ttost_ind(dataCorr, ctrl, -0.05, 0.05)

# Plot the correlations
dfPlotCorr = pd.DataFrame({
    'Values': np.concatenate([dataCorr, ctrl]),
    'Category': ['Nearest\nneighbour'] * len(dataCorr) + ['Control\npair'] * len(ctrl)})

fig, ax = plt.subplots(figsize=(2,4))
sns.boxplot(data = dfPlotCorr, x = 'Category', y = 'Values', palette = 'viridis', fliersize = 0.5)
ax.set_xlabel(None)
ax.set_ylabel('Tuning map correlation')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

plt.savefig('N:/axon2pmini/Article - Topography/Figures/NN/tuningmap.svg', format = 'svg') 

# Stats
rands = np.random.choice(controlCorr, len(dataCorr), replace=False)
statCorr = ttest_ind(dataCorr, controlCorr, nan_policy = 'omit')
print(statCorr)
ranksums(dataCorr, rands)
cohen_d(dataCorr, rands)

ttost_ind(dataCorr, rands, -0.05, 0.05)

# Plot the correlations with rands
dfPlotCorr = pd.DataFrame({
    'Values': np.concatenate([dataCorr, rands]),
    'Category': ['NN'] * len(dataCorr) + ['Rand pairs'] * len(rands)})

fig, ax = plt.subplots(figsize=(2,4))
sns.boxplot(data = dfPlotCorr, x = 'Category', y = 'Values', palette = 'viridis', fliersize = 0.5)
ax.set_xlabel(None)
ax.set_ylabel('Tuning map correlation')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

# Plot the place field distances
dataDist, controlDist = np.concatenate(nearDist), np.concatenate(nearDistCtrl)

c = []
for num in range(len(nearCorr)):
    c.append(np.random.choice(nearDistCtrl[num], nearDist[num].size, replace=False))

ctrlDist = np.concatenate(c)

# Plot the plce field distance
dfPlotDist = pd.DataFrame({
    'Values': np.concatenate([dataDist, ctrlDist]),
    'Category': ['Nearest\nneighbour'] * len(dataDist) + ['Control\npair'] * len(ctrlDist)})

fig, ax = plt.subplots(figsize=(2,4))
sns.boxplot(data = dfPlotDist, x = 'Category', y = 'Values', palette = 'viridis', fliersize = 0.5)
ax.set_xlabel(None)
# plt.xticks(rotation=45)
ax.set_ylabel('Place field distance (cm)')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

statDist = ttest_ind(dataDist, ctrlDist, nan_policy = 'omit')

plt.savefig('N:/axon2pmini/Article - Topography/Figures/NN/distance.svg', format = 'svg') 

# Stats
d, c = dataDist[~np.isnan(dataDist)], ctrlDist[~np.isnan(ctrlDist)]

np.median(d), np.median(c)
np.mean(d), np.mean(c)
ttest_ind(d, c, nan_policy = 'omit')
ranksums(d, c)
cohen_d(d, c)
ttost_ind(d, c, -1.25, 1.25)

#%% Test near vs. far

# For correlation
dfCorrFar = pd.DataFrame({
    'Values': np.concatenate([np.concatenate(nearCorr), np.concatenate(farCorr)]),
    'Category': ['Nearest\nneighbour'] * len(np.concatenate(nearCorr)) +
    ['Far-off'] * len(np.concatenate(farCorr))})

fig, ax = plt.subplots(figsize=(2,4))
sns.boxplot(data = dfCorrFar, x = 'Category', y = 'Values', palette = 'viridis', fliersize = 0.5)
ax.set_xlabel(None)
ax.set_ylabel('Correlation (r)')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

fig, ax = plt.subplots(figsize=(3.5,5))
sns.stripplot(data = dfCorrFar, x = 'Category', y = 'Values', palette = 'viridis', size = 1)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel(None)
ax.set_ylabel('Correlation (r)')
plt.tight_layout()

# Stats
d1, c1 = np.concatenate(nearCorr), np.concatenate(farCorr)

np.mean(d1), np.mean(c1)
np.median(d1), np.median(c1)
ttest_ind(d1, c1)
ranksums(d1, c1)
cohen_d(d1, c1)
ttost_ind(d1, c1, -0.05, 0.05)

# For distance
dfDistFar = pd.DataFrame({
    'Values': np.concatenate([np.concatenate(nearDist), np.concatenate(farDist)]),
    'Category': ['Nearest\nneighbour'] * len(np.concatenate(nearDist)) +
    ['Far-off'] * len(np.concatenate(farDist))})

fig, ax = plt.subplots(figsize=(2,4))
sns.boxplot(data = dfDistFar, x = 'Category', y = 'Values', palette = 'viridis', fliersize = 0.5)
ax.set_xlabel(None)
ax.set_ylabel('Field distance (cm)')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

# Stats
d2, c2 = np.concatenate(nearDist), np.concatenate(farDist)
d2, c2 = d2[~np.isnan(d2)], c2[~np.isnan(c2)]

np.mean(d2), np.mean(c2)
np.median(d2), np.median(c2)
ttest_ind(d2, c2)
ranksums(d2, c2)
cohen_d(d2, c2)
ttost_ind(d1, c1, -2.5, 2.5)

#%% What are the nearest distances?

distNear, distFar = [], []

for key in nearestNeighbour.keys():
    for NAT in NATs[0:2]:
        
        distNear.append(nearestNeighbour[key]['anat_distanceNear'][NAT])
        distFar.append(nearestNeighbour[key]['anat_distanceFar'][NAT])
        
n, f = np.concatenate(distNear), np.concatenate(distFar) 

data = {'Nearest': n, 'Furthest': f}

fig, ax = plt.subplots(1,2, figsize=(3,4))
sns.stripplot(data = n, ax = ax[0], palette = 'viridis', size = 1)
sns.stripplot(data = f, ax = ax[1], palette = 'viridis', size = 1)
ax[0].spines[['top', 'right']].set_visible(False)
ax[1].spines[['top', 'right']].set_visible(False)
ax[0].set_xlabel(None)
ax[0].set_xticklabels(['Nearest'])
ax[0].set_ylabel('Anatomical distance (μm)')
ax[1].set_xlabel(None)
ax[1].set_xticklabels(['Furthest'])
plt.tight_layout()

#%% Scatter plot of correlation and distance to nearest neighbour
dfScatter = {'Correlation': d1, 'Distance': n}

fig, ax = plt.subplots()
sns.scatterplot(data = dfScatter, x = 'Distance', y = 'Correlation', 
                palette = 'viridis', edgecolor = None, size = 0.01, alpha = 0.5, legend = False)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

dfScatter2 = {'Correlation': c1, 'Distance': f}

fig, ax = plt.subplots()
sns.scatterplot(data = dfScatter2, x = 'Distance', y = 'Correlation', 
                palette = 'viridis', edgecolor = None, size = 0.01, alpha = 0.5, legend = False)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
