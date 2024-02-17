# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 12:10:11 2023

@author: torstsl

field_neighbours.py

This script uses the same approach as in neighbourhood.py

Here, one session is loaded. For all place cells, their place fields are found, 
and their anatomical distance. Then, it looks at each cell and finds the other
cells who's place field is nearby to it. It does so by an expanding circle, 
inclusing cells within the circle. This distance is then compared to the 
anatomical distance between the unique cell pairs. 

The results are plotted as a line (one per subsession) and as a series of 
violin plots. This is done for the concatenated results (results from bin 1 is
kept within bin 2, and so forth), as well as for the "excluding" data, where
the results per bin is excluding data from the previous bin.
    Data per bin = limit[0] < x < limit [1]

For now, it only looks at the place field loaction (mean). It could also be
possible to have a look at cells with overlapping place fields of some degree, 
either a percentage or correation. And perhaps look at several place fields?

"""
import pickle
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

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

    return maxproj, centre_dist_dict, centre_dist

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

#%% Get the anatomical and field distance

def calc_distances(placefield, session_dict):
    
    nSessions = len(session_dict['dfNAT'])
    placecell_dict = session_dict['Placecell']
    
    centroid_coords = {}
    cell_centres = {}
    
    field_distance = {}
    anatomical_distance = {}
    
    for sessionNo in range(nSessions):
        key = 'NAT'+str(sessionNo)
        nPC = len(placecell_dict[key][0]) # Changes per session
        
        centroid_coords[key] = np.full([nPC,2], np.nan)
        cell_centres[key] = np.full([nPC,2], np.nan) 
        
        for cellNo in range(nPC):
            PC = placecell_dict[key][0][cellNo] # NATEX index
            
            if type(placefield[key][PC-1][0]) == dict:
                centroid_coords[key][cellNo] = placefield[key][PC-1][0]['centroid_coords'][::-1] # Coordinates are x,y, Python index
                cell_centres[key][cellNo] = session_dict['ExperimentInformation']['CellStat'][PC-1]['med'][::-1] # Python index
            else: 
                centroid_coords[key][cellNo] = np.nan, np.nan
                cell_centres[key][cellNo] = np.nan, np.nan
                
        # Calculate the pairwise distance between place field peaks of placecells, multiply by binning to get answer in cm
        field_distance[key] = (np.linalg.norm(centroid_coords[key] - centroid_coords[key][:,None], axis=-1))*binning # Scale by binning size
        
        # Calculate the pairwise distance between ROI location
        anatomical_distance[key] = np.linalg.norm(cell_centres[key] - cell_centres[key][:,None], axis=-1)/scale # Scaled by µm/pixel    

    return centroid_coords, cell_centres, field_distance, anatomical_distance

#%% Get the anatomical distance between cells that have place fields in certain distances from each other

def calc_close_distances(field_distance, anatomical_distance, session_dict):
    
    nSessions = len(session_dict['dfNAT'])
    placecell_dict = session_dict['Placecell']
    
    distances = np.linspace(5,50,10, dtype = int) # cm distance between place fields
    expanding_distance = {}
    
    expandingControl = {}
    nControls = 25
    
    for sessionNo in range(nSessions): 
        key = 'NAT'+str(sessionNo)
      
        nPC = len(placecell_dict[key][0])
        expandingControl[key] = [[]]*nPC
        expanding_distance[key] = np.full([nPC, len(distances)], np.nan)
        
        field_dist = field_distance[key].copy()
        np.fill_diagonal(field_dist, np.nan) # Set distances to selves to NaNs
        
        for cellNo in range(nPC):
            
            expandingControl[key][cellNo] = np.full([nControls, len(distances)], np.nan)
        
            for incrementNo, increment in enumerate(distances):    
                idx = np.where(field_dist[cellNo,:] < increment)[0]
                anatDist = np.nanmean(anatomical_distance[key][cellNo,idx])
                expanding_distance[key][cellNo, incrementNo] = anatDist
                
                for controlNo in range(nControls):
                    anatomicalArray = anatomical_distance[key][cellNo,:].copy()
                    anatomicalArray = np.delete(anatomicalArray, cellNo)
                    controlArray = np.random.choice(anatomicalArray, len(idx), replace = False)
                    
                    # Random picked, size matched number of place cell's anatomical distances to the reference cell, mean per shuffle
                    expandingControl[key][cellNo][controlNo, incrementNo] = np.nanmean(controlArray)
 
    return distances, expanding_distance, expandingControl 
 
#%% Plot the results from the expanding circle

def plot_expanding_circle(distances, expandig_circle, session_dict):
    
    nSessions = len(session_dict['dfNAT'])
    sessionName = ['A','B','A\'']
    
    fig, ax = plt.subplots(sharex = True, sharey = True)
    ax.set_title('Anatomical distance of place field neighbours')
    
    for sessionNo in range(nSessions):
        key = 'NAT'+str(sessionNo)
        data = np.full([len(distances),1], np.nan)
        ci = np.full([len(distances),1], np.nan)
        
        for incrementNo in range(len(distances)):
            data[incrementNo] = np.nanmean(expanding_distance[key][incrementNo])
            
            # Calculate confidence interval
            std = np.nanstd(expanding_distance[key][incrementNo], axis = 0)
            lengths = np.sum(~np.isnan(expanding_distance[key][incrementNo]), axis = 0)
            ci[incrementNo] = 1.96 * std/np.sqrt(lengths)
            y1, y2 = np.array(data - ci), np.array(data + ci)
    
        ax.plot(distances, data, label = sessionName[sessionNo], color = palette[(1+sessionNo)*75])
        ax.fill_between(distances[:], y1[:,0], y2[:,0], color = palette[(1+sessionNo)*75], alpha = 0.3, edgecolor = None) 
    
    ax.set_ylabel('Anatomical distance (µm)')
    ax.set_xlabel('Place field distance (cm)')
    ax.set_xlim(min(distances), max(distances))
    # ax.set_ylim([200, 220])
    
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon = False)
    plt.tight_layout()
    
    plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg')   


#%% Plot the results from the expanding circle as a series of histograms

# for x in distances: plt.figure(), plt.hist(expanding_distance[key][str(x)],100)

    # Plot the results as a series of violins
    fig, ax = plt.subplots(3,1, figsize = (10,6), sharex = True)
    plt.suptitle('Anatomical distance for place cells with close by place fields')
    fig.supxlabel('Place field distance (cm)')
    fig.supylabel('Anatomical distance (µm)')
    
    for sessionNo in range(nSessions):
        key = 'NAT'+str(sessionNo)
        
        df = pd.DataFrame()
    
        for keys in expanding_distance[key].keys():
            df_new = pd.DataFrame({keys: expanding_distance[key][keys]})
            df = pd.concat([df, df_new], axis = 1)
        
        sns.violinplot(data=df, ax = ax[sessionNo], palette = 'viridis')
        ax[sessionNo].spines[['top', 'right']].set_visible(False)
        ax[sessionNo].set_ylabel(sessionName[sessionNo], rotation = 0)
        
    plt.tight_layout()
    
    plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg')   
    
    # Similar as above, but sessions concatenated
    all_data = {}
    for keys in distances:
        str(keys)
        n = np.array([])
    
        for sessionNo in range(nSessions): # Concider changing this to just A and B (not duplicate A with A'?)
            key = 'NAT'+str(sessionNo)
            
            n = np.concatenate([n, expanding_distance[key][str(keys)]])
       
        all_data[str(keys)] = n
    
    df = pd.DataFrame.from_dict(all_data, orient = 'index').T
    
    fig, ax = plt.subplots(figsize = (12,4))
    ax.set_title('Anatomical distance for place cells with close by place fields')
    ax.set_xlabel('Place field distance (cm)')
    ax.set_ylabel('Anatomical distance (µm)')
    
    sns.violinplot(data=df, ax = ax, palette = 'viridis')
    ax.spines[['top', 'right']].set_visible(False)
        
    plt.tight_layout()
    
    plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg')

#%% Get the anatomical distance between cells that have place fields in certain distances from each other

# Excluding cells from the previous bin
def calc_close_distances_ex(field_distance, distances, anatomical_distance, session_dict):
    
    nSessions = len(session_dict['dfNAT'])
    placecell_dict = session_dict['Placecell']

    expanding_distance_ex = {}
    expandingControl_ex = {}
    
    nControls = 25
    
    for sessionNo in range(nSessions): 
        key = 'NAT'+str(sessionNo)
        expanding_distance_ex[key] = {}
        for x in distances: expanding_distance_ex[key][str(x)] = []
        
        nPC = len(placecell_dict[key][0])
        expandingControl_ex[key] = [[np.nan]]*nPC
        
        field_dist = field_distance[key].copy()
        np.fill_diagonal(field_dist, np.nan) # Set distances to selves to NaNs
        
        for cellNo in range(nPC):
            expandingControl_ex[key][cellNo] = np.full([nControls, len(distances)], np.nan)
        
            for incrementNo in range(len(distances)): 
                if incrementNo == 0:
                    
                    increment = distances[incrementNo]
                    idx = np.where(field_dist[cellNo,:] < increment)[0]
                    expanding_distance_ex[key][str(increment)].append(anatomical_distance[key][cellNo,idx])
                    
                elif incrementNo > 0:
                    idx = np.where((field_dist[cellNo,:] < distances[incrementNo]) & 
                                   (field_dist[cellNo,:] > distances[incrementNo-1]))[0]
                    expanding_distance_ex[key][str(distances[incrementNo])].append(anatomical_distance[key][cellNo,idx])
                    
                for controlNo in range(nControls):
                    anatomicalArray = anatomical_distance[key][cellNo,:].copy()
                    anatomicalArray = np.delete(anatomicalArray, cellNo)
                    controlArray = np.random.choice(anatomicalArray, len(idx), replace = False)
                    
                    # Random picked, size matched number of place cell's anatomical distances to the reference cell, mean per shuffle
                    expandingControl_ex[key][cellNo][controlNo, incrementNo] = np.nanmean(controlArray)    
                    
        for x in distances: expanding_distance_ex[key][str(x)] = np.concatenate(expanding_distance_ex[key][str(x)])            

    return expanding_distance_ex, expandingControl_ex

#%% Plot the results from the excluding circle   

def plot_expanding_circle_ex(distances, expanding_distance_ex, session_dict):
    
    nSessions = len(session_dict['dfNAT'])
    sessionName = ['A','B','A\'']
    
    # Plot the line plot   
    fig, ax = plt.subplots(sharex = True, sharey = True)
    ax.set_title('Stepwise anatomical distance of place field neighbours')
    
    for sessionNo in range(nSessions):
        key = 'NAT'+str(sessionNo)
        data = np.full([len(distances),1], np.nan)
        ci = np.full([len(distances),1], np.nan)
        
        for incrementNo in range(len(distances)):
            increment = str(distances[incrementNo])
            data[incrementNo] = np.nanmean(expanding_distance_ex[key][increment])
            
            # Calculate confidence interval
            std = np.nanstd(expanding_distance_ex[key][increment], axis = 0)
            lengths = np.sum(~np.isnan(expanding_distance_ex[key][increment]), axis = 0)
            ci[incrementNo] = 1.96 * std/np.sqrt(lengths)
            y1, y2 = np.array(data - ci), np.array(data + ci)
            
        ax.plot(distances, data, label = sessionName[sessionNo], color = palette[(1+sessionNo)*75])
        ax.fill_between(distances[:], y1[:,0], y2[:,0], color = palette[(1+sessionNo)*75], alpha = 0.3, edgecolor = None) 
    
    ax.set_ylabel('Anatomical distance (µm)')
    ax.set_xlabel('Place field distance (cm)')
    ax.set_xlim(min(distances), max(distances))
    ax.set_ylim([200, 220])
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon = False)    
    
    plt.tight_layout()
    
    plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg')


    # Plot as violin plot
    fig, ax = plt.subplots(3,1, figsize = (10,6), sharex = True)
    plt.suptitle('Anatomical distance for PCs with close by place fields (excluding)')
    fig.supxlabel('Place field distance (cm)')
    fig.supylabel('Anatomical distance (µm)')
    
    for sessionNo in range(nSessions):
        key = 'NAT'+str(sessionNo)
        
        df = pd.DataFrame()
    
        for keys in expanding_distance_ex[key].keys():
            df_new = pd.DataFrame({keys: expanding_distance_ex[key][keys]})
            df = pd.concat([df, df_new], axis = 1)
        
        sns.violinplot(data=df, ax = ax[sessionNo], palette = 'viridis')
        ax[sessionNo].spines[['top', 'right']].set_visible(False)
        
    plt.tight_layout()
    
    plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg')
    
    # Similar as above, but sessions concatenated
    all_data_ex = {}
    for keys in distances:
        str(keys)
        n = np.array([])
    
        for sessionNo in range(nSessions):
            key = 'NAT'+str(sessionNo)
            
            n = np.concatenate([n, expanding_distance_ex[key][str(keys)]])
       
        all_data_ex[str(keys)] = n
    
    df = pd.DataFrame.from_dict(all_data_ex, orient = 'index').T
    
    fig, ax = plt.subplots(figsize = (12,4))
    ax.set_title('Stepwise anatomical distance for place cells with close by place fields')
    ax.set_xlabel('Place field distance (cm)')
    ax.set_ylabel('Anatomical distance (µm)')
    
    sns.violinplot(data=df, ax = ax, palette = 'viridis')
    ax.spines[['top', 'right']].set_visible(False)
        
    plt.tight_layout()
    
    plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg')

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

field_neighbours_dict = {}

if __name__ == "__main__":
    
    for session in sessions: 
        
        # Load session_dict
        session_dict = pickle.load(open(session+'\session_dict.pickle','rb'))
        print('Successfully loaded session_dict from: '+str(session))
        
        key = session_dict['Animal_ID'] + '-' + session_dict['Date']
        
        maxproj, centre_dist_dict, centre_dist = calc_centre_dist(session_dict)
      
        placefield = get_placefields(session_dict)
        
        centroid_coords, cell_centres, field_distance, anatomical_distance = calc_distances(placefield, session_dict)
        
        distances, expanding_distance, expandingControl = calc_close_distances(field_distance, anatomical_distance, session_dict)
        
        expanding_distance_ex, expandingControl_ex = calc_close_distances_ex(field_distance, distances, anatomical_distance, session_dict)
        
        # Plot the results
        # plot_expanding_circle(distances, expanding_distance, session_dict)
        # plot_expanding_circle_ex(distances, expanding_distance_ex, session_dict)
        
        # Put the results in a dict
        field_neighbours_dict[key] = {'maxproj': maxproj,
                                      'centre_dist_dict': centre_dist_dict,
                                      'centre_dist': centre_dist,
                                      'placefield': placefield,
                                      'cell_centres': cell_centres,
                                      'field_distance': field_distance,
                                      'placefield': placefield,
                                      'field_distance': field_distance,
                                      'anatomical_distance': anatomical_distance,
                                      'distances': distances,
                                      'expanding_distance': expanding_distance,
                                      'expandingControl': expandingControl,
                                      'expanding_distance_ex': expanding_distance_ex, 
                                      'expandingControl_ex': expandingControl_ex}
        
        # Store the output
        with open(results_folder+'/field_neighbours_dict.pickle','wb') as handle:
            pickle.dump(field_neighbours_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Successfully saved results_dict in '+ results_folder)
   