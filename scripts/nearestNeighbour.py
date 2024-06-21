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
