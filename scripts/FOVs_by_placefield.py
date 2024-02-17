# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:53:44 2023

@author: torstsl


"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.cell_activity_analysis import plot_maxproj_feature, get_cell_masks
from scipy.stats import binned_statistic_2d

#%% Initiate session
results_folder = r'C:\Users\torstsl\Projects\axon2pmini\results'

with open(results_folder+'/sessions_overview.txt') as f:
    sessions = f.read().splitlines() 
f.close()

session_string = ['A', 'B', "A'"]

palette = list(sns.color_palette("viridis", 256).as_hex())
contrast = list(sns.color_palette('OrRd', 256).as_hex())

#%% Get the place field (just one prominent place field per cell (all cells))

def get_placefields(session_dict):

    nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)
    nSessions = len(session_dict['dfNAT'])
    placecell_dict = session_dict['Placecell']

    placefield = {}
    
    for sessionNo in range(nSessions):
        key = 'NAT'+str(sessionNo)
        placefield[key]=[]
        
        for cellNo in range(1, nCells +1): # NATEX index
            # placecell_dict['NAT'+str(sessionNo)][0]-1: # NATEX index converted to pythonic by -1 
            
            if type(session_dict['Placefields'][key]['N'+str(cellNo)]) == float:
                placefield[key].append([np.nan, np.nan])
                
                if cellNo in placecell_dict[key][0]:
                    print('Data lost for '+ key +' N ' + str(cellNo))
                    
            else: 
                fields, fields_map = session_dict['Placefields'][key]['N'+str(cellNo)] # Grab the fields data
                     
                if len(fields) == 1: # Only 1 detected field
                    placefield[key].append([fields[0], fields_map])
                
                elif len(fields) > 1: # More than 1 detected field, take the most prominent one (highest mean rate)
                    field_mean_rates = []
                    
                    for fieldNo in range(len(fields)): field_mean_rates.append(fields[fieldNo]['mean_rate'])
                    
                    field_idx = np.where(field_mean_rates==max(field_mean_rates))[0][0]
                   
                    placefield[key].append([fields[field_idx], fields_map])

    return placefield

#%% Plot the FOV with place cells from box A, color code by the position of the place field

for session in sessions:
    
    # Load session_dict
    session_dict = pickle.load(open(session+'\session_dict.pickle','rb'))
    print('Successfully loaded session_dict from: '+str(session))
    
    key = session_dict['Animal_ID'] + '-' + session_dict['Date']
    
    nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)
    nSessions = len(session_dict['dfNAT'])
    
    placecell_dict = session_dict['Placecell']['NAT0'][0]
    
    # Get maxprojection and cell masks 
    maxprojection, maxprojection_mask = get_cell_masks(session_dict['ExperimentInformation'])

    # Get place fields
    placefields = get_placefields(session_dict)
    
    # Get the coordinates for the place field (within a 32*32 array) of place cells
    nPC = len(placecell_dict)
    
    coords = np.full([nPC, 2], np.nan)
    
    for n, pcNo in enumerate(placecell_dict):
        field = placefields['NAT0'][pcNo-1]
        if type(field[0]) == dict:
            coords[n, :] = field[0]['centroid_coords']
            
    coords = coords*2.5 # Multiply by binning cm/bin   
    
    if np.sum(np.isnan(coords)) > 0:
        idx = np.where(np.isnan(coords[:,0]))[0]         
        placecell_dict = np.delete(placecell_dict, idx)
        coords = np.delete(coords, idx, axis = 0)
        
    # Get which bin each cell belongs to in a 4x4 matrix of the box
    bins = 4,4
    xedges, yedges = np.linspace(0,80,bins[0]+1), np.linspace(0,80,bins[1]+1)
    
    # hist, xedges, yedges = np.histogram2d(coords[:,0], coords[:,1], bins = (xedges,yedges)) 
    
    stat, xedges, yedges, binnum = binned_statistic_2d(coords[:,0], coords[:,1], None, 'count', bins = (xedges,yedges), expand_binnumbers = True)
    
    # Assign one color to each 4x4 part of the box
    palette2d = []
    skipper = int(np.round(256/np.prod(bins)-1))
    for x in range(np.prod(bins)): palette2d.append(palette[skipper*x])
    
    colorbin = np.array(binnum).T-1
    
    color = [[]]*(len(colorbin))
    
    for x in range(len(xedges)-1):
        xind = colorbin[:,0] == x
        for y in range(len(yedges)-1):
            yind = colorbin[:,1] == (y)
            both = np.where(xind & yind)[0]
            for c in both: color[c] = palette2d[bins[0]*x+y]
    
    # Plot the place cells in box A by which "box in the box" they belong to 
    fig, ax = plt.subplots(figsize = (8,8))
    fig.patch.set_alpha(0)
    ax.set_title('Anatomical distribution of place cells')
    ax.axis('off')
    sns.heatmap(maxprojection, ax = ax, cmap='gist_gray', alpha = 0.75, cbar = False, rasterized = True)
    ax.set_aspect('equal')
    for n, pcNo in enumerate(placecell_dict):
        cs = ax.contourf(maxprojection_mask[pcNo], colors=color[n], alpha = 0.45)
        for c in cs.collections: c.set_rasterized(True)
        
    plt.savefig(r'N:\axon2pmini\Article\Figures\Supplementary\FOV'+(key)+'.svg', format = 'svg')
    
    fig, ax = plt.subplots(figsize = (8,8))
    fig.patch.set_alpha(0)
    ax.set_title('Anatomical distribution of place cells')
    ax.axis('off')
    ax.set_aspect('equal')
    for n, pcNo in enumerate(placecell_dict):
        cs = ax.contourf(maxprojection_mask[pcNo], colors=color[n], alpha = 0.8)
        for c in cs.collections: c.set_rasterized(True)
        
    plt.savefig(r'N:\axon2pmini\Article\Figures\Supplementary\maskFOV'+(key)+'.svg', format = 'svg')
    
#%% Make the box
box = np.arange(16).reshape(4,4)
fig, ax = plt.subplots()
sns.heatmap(data = box, cmap = 'viridis', cbar = False)
ax.set_aspect('equal')
plt.axis('off')

plt.savefig(r'N:\axon2pmini\Article\Figures\Supplementary\box_4x4.svg', format = 'svg')