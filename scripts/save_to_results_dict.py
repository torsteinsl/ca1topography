# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:37:15 2023

@author: torstsl
"""

import pickle
import numpy as np
from src.loading_data import load_results_dict
#%% Load sessions to run through
results_folder = r'C:\Users\torstsl\Projects\axon2pmini\results'

with open(results_folder+'/sessions_overview.txt') as f:
    sessions = f.read().splitlines() 
f.close()

results_dict = load_results_dict()

#%% Go through all sessions
for session in sessions: 
    
    # Load session_dict
    session_dict = pickle.load(open(session+'\session_dict.pickle','rb'))
    print('Successfully loaded session_dict from: '+str(session))
    
    #%% Get data from topography_pairwise_distance   
    nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)
    nSessions = len(session_dict['dfNAT'])
    
    scale = 1.1801 # Pixels/µm --> 1 pixel = 0.847.. µm 
    binning = 2.5
    
    placecell_dict = session_dict['Placecell']
    
    # Create variable with just one prominent place field per cell (all cells)
    placefield = {}
    
    for sessionNo in range(nSessions):
        placefield['NAT'+str(sessionNo)]=[]
        
        for cellNo in range(1, nCells +1): # NATEX index
            
            if type(session_dict['Placefields']['NAT'+str(sessionNo)]['N'+str(cellNo)]) == float:
                placefield['NAT'+str(sessionNo)].append([np.nan, np.nan])
                
                if cellNo in placecell_dict['NAT'+str(sessionNo)][0]:
                    print('Data lost for NAT '+str(sessionNo) +' N ' + str(cellNo))
                    
            else: 
                fields, fields_map = session_dict['Placefields']['NAT'+str(sessionNo)]['N'+str(cellNo)] # Grab the fields data
                     
                if len(fields) == 1: # Only 1 detected field
                    placefield['NAT'+str(sessionNo)].append([fields[0], fields_map])
                
                elif len(fields) > 1: # More than 1 detected field, take the most prominent one (highest mean rate)
                    field_mean_rates = []
                    
                    for fieldNo in range(len(fields)): field_mean_rates.append(fields[fieldNo]['mean_rate'])
                    
                    field_idx = np.where(field_mean_rates==max(field_mean_rates))[0][0]
                   
                    placefield['NAT'+str(sessionNo)].append([fields[field_idx], fields_map])
                
    # Get field coordinates, anatomical distance and calculate field distances
    centroid_coords = {}
    cell_centre = {}
    
    field_distance = {}
    anatomical_distance = {}
    
    for sessionNo in range(nSessions):
        key = 'NAT'+str(sessionNo)
        nPC = len(placecell_dict[key][0]) # Changes per session
        
        centroid_coords[key] = np.full([nPC,2], np.nan)
        cell_centre[key] = np.full([nPC,2], np.nan) 
        
        for cellNo in range(nPC):
            PC = placecell_dict[key][0][cellNo] # NATEX index
            
            if type(placefield[key][PC-1][0]) == dict:
                centroid_coords[key][cellNo] = placefield[key][PC-1][0]['centroid_coords'][::-1] # Coordinates are x,y, Python index
                cell_centre[key][cellNo] = session_dict['ExperimentInformation']['CellStat'][PC-1]['med'][::-1] # Python index
            else: 
                centroid_coords[key][cellNo] = np.nan, np.nan
                cell_centre[key][cellNo] = np.nan, np.nan
                
        # Calculate the pairwise distance between place field peaks of placecells, multiply by binning to get answer in cm
        field_distance[key] = (np.linalg.norm(centroid_coords[key] - centroid_coords[key][:,None], axis=-1))*binning # Scale by binning size
        
        # Calculate the pairwise distance between ROI location
        anatomical_distance[key] = np.linalg.norm(cell_centre[key] - cell_centre[key][:,None], axis=-1)/scale # Scaled by µm/pixel
    
    # Plot field distance and anatomical distance pairs
    f_dist = [[]]*nSessions
    a_dist = [[]]*nSessions
    
    for sessionNo in range(nSessions):
        mask = np.triu(np.ones_like(field_distance['NAT'+str(sessionNo)], dtype=bool),1)
    
        f_dist[sessionNo] = field_distance['NAT'+str(sessionNo)][mask==True]
        a_dist[sessionNo] = anatomical_distance['NAT'+str(sessionNo)][mask==True]
    
    #%% Store the f_dist and a_dist from this session 
    results_dict[session] = {'a_dist': a_dist, 'f_dist': f_dist}    
    
    with open(results_folder+'/results_dict.pickle','wb') as handle:
        pickle.dump(results_dict,handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Successfully saved results_dict in '+results_folder)