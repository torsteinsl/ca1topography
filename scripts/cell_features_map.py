# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 07:53:57 2023

@author: torstsl

- Get the max proj. map
- Get the masks for all cells
- Plot all, color code by feature

"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import opexebo as op
from src.cell_activity_analysis import plot_maxproj_feature, get_cell_masks

#%% Initiate session
results_folder = r'C:\Users\torstsl\Projects\axon2pmini\results'

with open(results_folder+'/sessions_overview.txt') as f:
    sessions = f.read().splitlines() 
f.close()

# Iterate over sessions and count all PCs
count_PC_dict = {}  
  
# for session in sessions: 
session = sessions[7]
                   
# Load session_dict
session_dict = pickle.load(open(session+'\session_dict.pickle','rb'))
print('Successfully loaded session_dict from: '+str(session))

nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)
nSessions = len(session_dict['dfNAT'])
session_string = ['A', 'B', "A'"]

placecell_dict = session_dict['Placecell']
placecells = np.unique(np.concatenate([placecell_dict['NAT0'][0],placecell_dict['NAT1'][0],placecell_dict['NAT2'][0]]))

#%% Get maxprojection and cell masks 
maxprojection, maxprojection_mask = get_cell_masks(session_dict['ExperimentInformation'])

#%% Plots
# For loop over all masks of place cells to get them separately
fig, ax = plt.subplots(figsize = (10,10))    
ax.set_title('Max projection, '+str(nCells)+' cells')
ax.set_xlabel(None)  
ax.set_xticks([])
ax.set_ylabel(None)  
ax.set_yticks([])
ax.invert_yaxis()
ax.set_facecolor('black')
for cellNo in range(nCells):
    ax.contour(maxprojection_mask.mask[cellNo], colors = 'w', linewidths = 0.15, alpha = 0.5)
plt.tight_layout()    

# For loop over all masks of place cells to get them separately
fig, ax = plt.subplots(figsize = (10,10))    
ax.set_title('Max projection, '+str(nCells)+' cells')
ax.imshow(maxprojection,cmap='gist_gray')
ax.axis('off')
for cellNo in range(nCells):
    ax.contour(maxprojection_mask.mask[cellNo], colors = 'w', linewidths = 0.15, alpha = 0.5)
plt.tight_layout()    

#%% Calculate spatial information for all cells
SI = {}

for sessionNo in range(nSessions):
    key = 'NAT'+str(sessionNo)
    SI[key] = np.full(nCells, np.nan)
    
    dfNAT = session_dict['dfNAT']['df'+key]
    timestamps = dfNAT.Timestamps.to_numpy()
    headpos = dfNAT[['Head_X','Head_Y']].to_numpy()
    
    x_edges = np.linspace(headpos[:,0].min(), headpos[:,0].max(), 33)
    y_edges = np.linspace(headpos[:,1].min(), headpos[:,1].max(), 33)
    
    occupancy = op.analysis.spatial_occupancy(timestamps, headpos.T, 80, bin_edges = (x_edges, y_edges))[0]
                                                  
    for cellNo in range(nCells):
        ratemap = session_dict['Ratemaps']['df'+key]['N'+str(cellNo+1)]
        
        SI[key][cellNo] = op.analysis.rate_map_stats(ratemap, occupancy)['spatial_information_content']

#%% Plot maximum intensity projections with all cells color coded by SI
for sessionNo in range(nSessions):
    title_string = session_string[sessionNo]+': SI'
    plot_maxproj_feature(maxprojection,maxprojection_mask, SI['NAT'+str(sessionNo)], title_string, background = 'maxint')  

#%% Plot maximum intensity projections with place cells color coded by SI
# PC = placecells - 1 
for sessionNo in range(nSessions):
    PC = placecell_dict['NAT'+str(sessionNo)][0]-1
    title_string = session_string[sessionNo]+': SI'
    plot_maxproj_feature(maxprojection,maxprojection_mask[PC], SI['NAT'+str(sessionNo)][PC], title_string, background = 'maxint')    
    plot_maxproj_feature(maxprojection,maxprojection_mask[PC], SI['NAT'+str(sessionNo)][PC], title_string, background = 'black')    

#%% Place field numbers and size (sum of area of all fields) 
numFields = {}
sizeFields = {} 

for sessionNo in range(nSessions):
    key = 'NAT'+str(sessionNo)
    numFields[key] = np.full(nCells, np.nan)
    sizeFields[key] = np.full(nCells, np.nan)
    
    for cellNo in range(nCells):
        if  type(session_dict['Placefields'][key]['N'+str(cellNo+1)]) == tuple: # If not, there is no place field and the value is NaN
            numFields[key][cellNo] = len(session_dict['Placefields'][key]['N'+str(cellNo+1)][0])
            sizeFields[key][cellNo] = 0
            
            for fieldNo in range(int(numFields[key][cellNo])):
                sizeFields[key][cellNo] += session_dict['Placefields'][key]['N'+str(cellNo+1)][0][fieldNo]['area']
                      
#%% Plot distribution of numFields and sizeFields        

for sessionNo in range(nSessions):
    PC = placecell_dict['NAT'+str(sessionNo)][0]-1
    title_string = session_string[sessionNo]+': numField'
    plot_maxproj_feature(maxprojection,maxprojection_mask[PC], numFields['NAT'+str(sessionNo)][PC], title_string, background = 'maxint')    
    plot_maxproj_feature(maxprojection,maxprojection_mask[PC], numFields['NAT'+str(sessionNo)][PC], title_string, background = 'black') 
    
for sessionNo in range(nSessions):
    PC = placecell_dict['NAT'+str(sessionNo)][0]-1
    title_string = session_string[sessionNo]+': sizeField'
    plot_maxproj_feature(maxprojection,maxprojection_mask[PC], sizeFields['NAT'+str(sessionNo)][PC], title_string, background = 'maxint')    
    plot_maxproj_feature(maxprojection,maxprojection_mask[PC], sizeFields['NAT'+str(sessionNo)][PC], title_string, background = 'black')  