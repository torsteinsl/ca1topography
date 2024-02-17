# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 14:29:40 2023

@author: torstsl

This script is made to look at the neighbours of singel place cells.

It uses the distortion and motion corrected corrected max projection images 
and find the cells' centre. For each single place cell, other place cells at
different anatomical distances are grouped into "neighbourhoods"/"cohorts". 
For all cells that are classified as place cells within each trial (A-B-A'), 
the correlation between the centre cell to all the neighbours is calculated. 
The mean of this correlation is returned. This is then done for all cells, 
and the results plotted for each single trial as a function of anatomical 
distance from the centre cell. 

Similarily, it uses the same approach to calculate the distance between place 
fields from a center/reference cell to all other place cells within a given
anatomical distance. The mean of the mean distance for all cells is then
plotted similarily as the correlation. 

"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from src.cell_activity_analysis import ratemap_correlation

#%% Get the maximum intensitity projection image from the session and the cell centre (max projection coordinates)
def calc_centre_dist(session_dict):

    nSessions = len(session_dict['dfNAT'])

    placecell_dict = session_dict['Placecell']
    placecells = np.unique(np.concatenate([placecell_dict['NAT0'][0], placecell_dict['NAT1'][0], placecell_dict['NAT2'][0]]))

    maxproj = session_dict['ExperimentInformation']['ImagingOptions']['max_proj']
    
    yrange = session_dict['ExperimentInformation']['ImagingOptions']['yrange'] # Valid y-range used for cell detection
    xrange = session_dict['ExperimentInformation']['ImagingOptions']['xrange'] # Valid x-range used for cell detection
    
    # Make an array of centre coordinates for all placecells
    cell_centre = np.full([len(placecells),2],np.nan) 
    counter = 0
    for cellNo in placecells-1: # NATEX index of placecells converted to python index  
        cell_centre[counter] = session_dict['ExperimentInformation']['CellStat'][cellNo]['med'][::-1] # Get cell centre, xy (anatomical)
        cell_centre[counter][0] -= xrange[0]
        cell_centre[counter][1] -= yrange[0]
        
        counter += 1
    
    # Get the distance between cells/ROIs for all placecells
    centre_dist = np.linalg.norm(cell_centre - cell_centre [:,None], axis=-1)/scale # Distance in µm
    
    # Make a dict of distances between placecells per session
    centre_dist_dict = {}
    
    for sessionNo in range(nSessions):
        sessionPC = np.isin(placecells, placecell_dict['NAT'+str(sessionNo)]) # The filter (array of booleans)
        PC_centre = cell_centre[sessionPC,:]
        centre_dist_dict['NAT'+str(sessionNo)] = np.linalg.norm(PC_centre - PC_centre [:,None], axis=-1)/scale # Distance in µm
    
    return maxproj, centre_dist, centre_dist_dict

#%% Get the neighbours of each place cell

def get_neighbours(session_dict, centre_dist, centre_dist_dict):
    
    nSessions = len(session_dict['dfNAT'])

    placecell_dict = session_dict['Placecell']
    placecells = np.unique(np.concatenate([placecell_dict['NAT0'][0], placecell_dict['NAT1'][0], placecell_dict['NAT2'][0]]))
    
    distances = np.linspace(15,100,18)
    cohorts = [[np.nan]]*len(placecells)
    
    # For all place cells
    for cellNo in range(len(placecells)):
        counter = 0
        cohort = []
        
        for increment in distances:
            idx = np.where((centre_dist[:,cellNo] < increment) & (centre_dist[:,cellNo] > 0))[0]
            cohort.append(placecells[idx]) # NATEX index of place cells
            counter += 1
        
        # List of length placecells, with an array for each cells neighbours at different distances (increment)    
        cohorts[cellNo] = cohort 
          
    # Get the neighbours for each place cell, but filtered by session (compare placecells in A only to other place cells in A)
    cohorts_dict = {}
    
    for sessionNo in range(nSessions):
        nPC = len(placecell_dict['NAT'+str(sessionNo)][0])
        cohorts_dict['NAT'+str(sessionNo)] = [[np.nan]]*nPC
        
        for cellNo in range(nPC):
            counter = 0
            cohort = []
        
            for increment in distances:
                # Indexes where the distances from this place cell (in an array of distances to other place cells in this session)
                # to other place cells are < increment distance, but > 0 (excluding self). Index corresponds to number ([0,nPC>]).
                idx = np.where((centre_dist_dict['NAT'+str(sessionNo)][:,cellNo] < increment) &
                               (centre_dist_dict['NAT'+str(sessionNo)][:,cellNo] > 0))[0]
                cohort.append(idx)  
                    
                counter += 1
            
            # Each cells neighbours at different instances (increment), index is ([0,nPC>]), hence not NATEX ID
            cohorts_dict['NAT'+str(sessionNo)][cellNo] = cohort 

    return cohorts, cohorts_dict, distances

#%% Get ratemaps for all placecells from the dict into a 3D np.array
def ratemaps_matrix(session_dict):
        
    nSessions = len(session_dict['dfNAT'])
    placecell_dict = session_dict['Placecell']
    
    ratemaps_dict = {}
    
    for sessionNo in range(nSessions):
        nPC = len(placecell_dict['NAT'+str(sessionNo)][0])
        ratemaps = np.ma.zeros([nPC, 32 , 32])
        
        for cellNo in range(nPC):
            cell = placecell_dict['NAT'+str(sessionNo)][0][cellNo]
            ratemaps[cellNo] = session_dict['Ratemaps']['dfNAT'+str(sessionNo)]['N'+str(cell)]
            
        ratemaps_dict['NAT'+str(sessionNo)] = ratemaps  # Just the session's place cells

    return ratemaps_dict

#%% Correlate the ratemaps for all other cells in the cohort to the test cell

def correlate_ratemaps(session_dict, ratemaps_dict, distances, cohorts_dict, **kwargs):
    
    plotter = kwargs.get('plotter', False)
    
    nSessions = len(session_dict['dfNAT'])
    placecell_dict = session_dict['Placecell']
    
    cell_stat_dict = {}
    control_stat_dict = {}
                         
    nControls = 25
    
    for sessionNo in range(nSessions):
        nPC = len(placecell_dict['NAT'+str(sessionNo)][0])
        cell_stat_dict['NAT'+str(sessionNo)] = [[np.nan]]*nPC
        control_stat_dict['NAT'+str(sessionNo)] = [[np.nan]]*nPC
        
        for cellNo in range(nPC):
            
            control_stat_dict['NAT'+str(sessionNo)][cellNo] = np.full([nControls, len(distances)], np.nan)
            
            r_values = []
            
            mainArray = ratemaps_dict['NAT'+str(sessionNo)][cellNo] # Reference cell
            
            for increment in range(len(distances)):
                # Get the neighbours from cohort, already indexed pythonic ([0,nPC>])
                neighbours = cohorts_dict['NAT'+str(sessionNo)][cellNo][increment] 
                testArray = ratemaps_dict['NAT'+str(sessionNo)][neighbours]
                
                stats = ratemap_correlation(mainArray, testArray)
                r_values.append(stats[0])
                
                for controlNo in range(nControls):
                    # Do a control with the same number of random place cells
                    pcs = np.arange(nPC).copy()
                    pcs = np.delete(pcs, cellNo) # Remove reference cell from the array
        
                    controlNeighbours = np.random.choice(pcs, len(neighbours), replace = False)
                    controlArray = ratemaps_dict['NAT'+str(sessionNo)][controlNeighbours]
                    
                    statsControl = ratemap_correlation(mainArray, controlArray)
                    control_stat_dict['NAT'+str(sessionNo)][cellNo][controlNo, increment] = statsControl[0]
                    
            # Per place cell for this session, the array is the mean correlation for each distance in 'distances'
            cell_stat_dict['NAT'+str(sessionNo)][cellNo] = np.array(r_values)
            
        # Plot the mean of mean correlations for all place cells as a function of distance from the cell centre
    if plotter == True:
        
        session_string = ['A', 'B', 'A\'']
        cmap = plt.get_cmap('viridis')
        rgb_cm = cmap.colors 
        
        fig, ax = plt.subplots()
        
        for sessionNo in range(nSessions):
            cell_stat_arr = np.array(cell_stat_dict['NAT'+str(sessionNo)])
            print('Num cells at 10 µm in', session_string[sessionNo],':', 
                  np.sum(~np.isnan(cell_stat_arr[:,0])),'/',str(len(placecell_dict['NAT'+str(sessionNo)][0])))
            print('Num cells at 15 µm in', session_string[sessionNo],':', 
                  np.sum(~np.isnan(cell_stat_arr[:,1])),'/',str(len(placecell_dict['NAT'+str(sessionNo)][0])))
            print('Num cells at 20 µm in', session_string[sessionNo],':', 
                  np.sum(~np.isnan(cell_stat_arr[:,2])),'/',str(len(placecell_dict['NAT'+str(sessionNo)][0])))
            stats_mean = np.nanmean(cell_stat_arr, axis = 0)
            
            # Calculate confidence interval
            std = np.nanstd(cell_stat_arr, axis = 0)
            lengths = np.sum(~np.isnan(cell_stat_arr), axis = 0)
            ci = 1.96 * std/np.sqrt(lengths)
            
            ax.plot(distances, stats_mean, label = session_string[sessionNo], color = rgb_cm[sessionNo*90]) 
            ax.fill_between(distances, (stats_mean-ci), (stats_mean+ci), color = rgb_cm[sessionNo*90], alpha = 0.2, edgecolor = None)
            # ax.legend(frameon = False, ncol = 3)
            ax.legend(frameon = False)
            
        ax.set_xlim(min(distances), max(distances))    
        ax.set_ylim([0,0.15])    
        ax.set_xlabel('Anatomical distance (µm)')
        ax.set_ylabel('Correlation (r)')
        ax.set_title('Mean correlation for nearby place cells')
        ax.spines[['top', 'right']].set_visible(False)
        
        plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg')   

    return cell_stat_dict, control_stat_dict
 
#%% Get the place fields and the pairwise distance for all place cells per session

def get_placefields_distance(session_dict):

    nSessions = len(session_dict['dfNAT'])
    placecell_dict = session_dict['Placecell']
    
    placefield = {}
    placefield_coords = {}
    field_distance = {}
    
    for sessionNo in range(nSessions):
        nPC = len(placecell_dict['NAT'+str(sessionNo)][0])
        placefield['NAT'+str(sessionNo)]=[]
        placefield_coords['NAT'+str(sessionNo)] = np.full([nPC,2],np.nan)
        
        for placecellNo in range(nPC):    
            # Get the cell number and convert to NATEX id
            cellNo = placecell_dict['NAT'+str(sessionNo)][0][placecellNo]+1
            
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
      
            # Get the place field coordinates (centroid) for each placecells place field (x,y)
            if type(placefield['NAT'+str(sessionNo)][placecellNo][0]) == dict:
                placefield_coords['NAT'+str(sessionNo)][placecellNo] = placefield['NAT'+str(sessionNo)][placecellNo][0]['centroid_coords'][::-1]
            if type(placefield['NAT'+str(sessionNo)][placecellNo][0]) != dict:
                print('No place field found: Session',sessionNo,'Cell',cellNo)
                placefield_coords['NAT'+str(sessionNo)][placecellNo] = placefield['NAT'+str(sessionNo)][placecellNo] # Set's the missing data to NaN
        
        # Calculate the pairwise distance between place fields of placecells, multiply by binning to get answer in cm   
        field_distance['NAT'+str(sessionNo)] = (np.linalg.norm(placefield_coords['NAT'+str(sessionNo)]            
           - placefield_coords['NAT'+str(sessionNo)][:,None], axis=-1))*2.5 

    return placefield, placefield_coords, field_distance

#%% Get the mean field distance in increments from a centre cell

def calc_mean_field_dist(session_dict, field_distance, distances, cohorts_dict, **kwargs):
    
    plotter = kwargs.get('plotter', False)
    
    nSessions = len(session_dict['dfNAT'])
    placecell_dict = session_dict['Placecell']
    
    nControls = 25
    
    field_stat_dict = {}
    field_control_dict = {}
    
    for sessionNo in range(nSessions):
        nPC = len(placecell_dict['NAT'+str(sessionNo)][0])
        
        field_stat_dict['NAT'+str(sessionNo)] = [[np.nan]]*nPC
        field_control_dict['NAT'+str(sessionNo)] = [[np.nan]]*nPC
        for cellNo in range(nPC):
            
            cell_field_distances = field_distance['NAT'+str(sessionNo)][:,cellNo] 
            field_control_dict['NAT'+str(sessionNo)][cellNo] = np.full([nControls, len(distances)], np.nan)   
            
            d_values = [] 
            
            for increment in range(len(distances)):
                # Get the neighbours from cohort, already indexed pythonic ([0,nPC>])
                neighbours = cohorts_dict['NAT'+str(sessionNo)][cellNo][increment] 
                
                testDistances = cell_field_distances[neighbours]
                
                d_values.append(np.nanmean(testDistances))
                
                for controlNo in range(nControls):
                    # Do a control with the same number of random place cells
                    pcs = np.arange(nPC).copy()
                    pcs = np.delete(pcs, cellNo) # Remove reference cell from the array
        
                    controlNeighbours = np.random.choice(pcs, len(neighbours), replace = False)
                    controlArray = cell_field_distances[controlNeighbours]
                    
                    field_control_dict['NAT'+str(sessionNo)][cellNo][controlNo, increment] = np.nanmean(controlArray)
                
            # Per place cell for this session, the array is the mean correlation for each distance in 'distances'
            field_stat_dict['NAT'+str(sessionNo)][cellNo] = np.array(d_values)

    # Plot the mean place field distance for all place cells as a function of distance from the cell centre
    if plotter == True: 
        
        session_string = ['NAT0', 'NAT1', 'NAT2']
        cmap = plt.get_cmap('viridis')
        rgb_cm = cmap.colors 
       
        fig, ax = plt.subplots()
        
        for sessionNo in range(nSessions):
            field_stat_arr = np.array(field_stat_dict['NAT'+str(sessionNo)])
        
            field_dist_mean = np.nanmean(field_stat_arr, axis = 0)
            
            # Calculate confidence interval
            std = np.nanstd(field_stat_arr, axis = 0)
            lengths = np.sum(~np.isnan(field_stat_arr), axis = 0)
            ci = 1.96 * std/np.sqrt(lengths)
            
            ax.plot(distances, field_dist_mean, label = session_string[sessionNo], color = rgb_cm[sessionNo*90]) 
            ax.fill_between(distances, (field_dist_mean-ci), (field_dist_mean+ci), color = rgb_cm[sessionNo*90], alpha = 0.2, edgecolor = None)
            ax.legend(loc = 'upper right', frameon = False)
            
        ax.set_xlim(min(distances), max(distances))    
        ax.set_ylim([38,45]) 
        ax.set_xlabel('Anatomical distance (µm)')
        ax.set_ylabel('Field distance (cm)')
        ax.set_title('Place field distance for nearby place cells')
        ax.spines[['top', 'right']].set_visible(False)
        
        plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg')   

    return field_stat_dict, field_control_dict

#%% Define constants
sessionName = ['A','B','A\'']

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

neighbourhood_dict = {}

if __name__ == "__main__":
    
    for session in sessions: 
        
        # Load session_dict
        session_dict = pickle.load(open(session+'\session_dict.pickle','rb'))
        print('Successfully loaded session_dict from: '+str(session))
        
        key = session_dict['Animal_ID'] + '-' + session_dict['Date']
        
        maxproj, centre_dist, centre_dist_dict = calc_centre_dist(session_dict)
        
        cohorts, cohorts_dict, distances = get_neighbours(session_dict, centre_dist, centre_dist_dict)
        
        ratemaps_dict = ratemaps_matrix(session_dict)
        
        cell_stat_dict, control_stat_dict = correlate_ratemaps(session_dict, ratemaps_dict, distances, cohorts_dict, plotter = False)
        
        placefield, placefield_coords, field_distance = get_placefields_distance(session_dict)
        
        field_stat_dict, field_control_dict = calc_mean_field_dist(session_dict, field_distance, distances, cohorts_dict, plotter = False)
        
        # Put the results in a dict
        neighbourhood_dict[key] = {'maxproj': maxproj,
                                   'centre_dist': centre_dist,
                                   'centre_dist_dict': centre_dist_dict,
                                   'cohorts': cohorts,
                                   'cohorts_dict': cohorts_dict, 
                                   'cell_stat_dict': cell_stat_dict,
                                   'control_stat_dict': control_stat_dict,
                                   'placefield': placefield,
                                   'placefield_coords': placefield_coords,
                                   'field_distance': field_distance,
                                   'field_stat_dict': field_stat_dict,
                                   'field_control_dict': field_control_dict}
        
        # Store the output
        with open(results_folder+'/neighbourhood_dict.pickle','wb') as handle:
            pickle.dump(neighbourhood_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Successfully saved results_dict in '+ results_folder)
        