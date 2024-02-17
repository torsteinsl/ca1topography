# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:56:35 2022

@author: torstsl

 --- TOPOGRAPHY ANALYSIS ---

This is a primary analysis of topogrgaphy of CA1 place cells. 

The scripts loads a session, in which place cells has been determined by the 
place_cell_analysis (hence placecell_dict is stored within session_dict), and
finds two elements per place cell:
    
    Place field location: Defined by the place field, which is detected using
                          opexebo and sep and stored within session_dict. 
                          From the place field, the centroid coordinates are 
                          stored. Centroid coordinates are better then peak, as
                          peak leads to binning issues (returns the peak bin, 
                          not a float of coordinates for the peak). Scaled by
                          binning at  2.5 cm/bin.
    Anatomical location:  The centre of the ROI loaded from the suite2p output, 
                          which in turn is stored in ExperimentInformation. 
                          The anatomical location is defined as the centre of 
                          the cells' ROI. Coodinates are motion corrected and
                          fits the maximum intensity projection. Scaled by 
                          resolution of the scope (1.1801 pixels/µm).
                                    
Further, the script calculated the pairwise distance in the box (place field) 
and in the FOV (anatomical distance) for all pairs of place cells. The results
are presented as a scatter plot for the pairwise distance is a session-wise 
matter as well as a 2D density heat map. In addition, it plots the correlation
matrix between place field distance and anatomical distance for place cells 
per session (which is kind of hard to interpret).

Then, it makes a strict addition to which cells to analyse in order to look at 
a potential effect of remapping on topography. Only cells that are classified 
as place cells  in all three session (A-B-A') are concidered for further 
analysis. Then, only PCs that are stable from A-A' (intersession_stability)
are accepted. A subset of stable and the non-stable cells are shown for check. 
For the stable cells, the place field distance is compared to their anatomical 
distance.

Secondly, looking at the strict cells, this script looks at how far apart cells
with similar firing fields are anatomically. For all place cell pairs, the 
correlation of place fields (Pearson) are compared to their anatomical distance. 
This is presented as a scatter plot, together with a linear regression of the
results - both normalized and "raw".

The last add on to this script is a visualization of how peak and centroid 
match the place fields of place cells. Just as a working tool to determine 
how good the field detection is. 

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import concurrent.futures as cf
import pickle
import scipy.stats as st
import opexebo as op
from tqdm import tqdm
from src.cell_activity_analysis import intersession_stability, plot_2d_density, ttest2, ttest_paired, shuffle_ratemap
from scipy.stats import pearsonr, linregress 
# from math import comb

#%% Define the analyses to be performed
def calc_field_distance(session_dict, **kwargs):
    
    plotter = kwargs.get('plotter', False)
    
    # Define constants and variables
    nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)
    nSessions = len(session_dict['dfNAT'])
    sessionName = ['A','B','A\'']
    
    scale = 1.1801 # Pixels/µm --> 1 pixel = 0.847.. µm 
    binning = 2.5
    
    placecell_dict = session_dict['Placecell']

    # Create variable with just one prominent place field per cell (all cells)
    placefield = {}
    
    for sessionNo in range(nSessions):
        placefield['NAT'+str(sessionNo)]=[]
        
        for cellNo in range(1, nCells +1): # NATEX index
            # placecell_dict['NAT'+str(sessionNo)][0]-1: # NATEX index converted to pythonic by -1 
            
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
    
    # This is the correlation between the two 1D vectors that are matrices above. Makes more sense
    corr_r = [[]]*nSessions
    
    for sessionNo in range(nSessions):
        maskNan = ~np.isnan(f_dist[sessionNo])
        corr_r[sessionNo] = pearsonr(f_dist[sessionNo][maskNan], a_dist[sessionNo][maskNan]) # Returns Pearson R and p-value
    
    if plotter == True:
        
        fig, ax = plt.subplots(1,nSessions, sharey = True, sharex = True)
        fig.suptitle('Pairwise distance between place field and ROI (centroid)')
        for sessionNo in range(nSessions):
            ax[sessionNo].scatter(f_dist[sessionNo],a_dist[sessionNo], s = 0.05, alpha = 0.5)
            ax[sessionNo].set_xlabel('Field distance (cm)')
            ax[sessionNo].set_title(str(sessionName[sessionNo])+(': ')+str(len(f_dist[sessionNo]))+' pairs')
            ax[sessionNo].set_xlim([0,max(f_dist[sessionNo])])
            ax[sessionNo].set_ylim([0,max(a_dist[sessionNo])])
            
            if sessionNo > 0:
                ax[sessionNo].tick_params(
                    axis='y',          # Changes apply to this axis
                    which='both',      # Both major and minor ticks are affected
                    left=False,        # Ticks along this edge are off
                    right=False,       # Ticks along this edge are off
                    labelleft=False    # Labels along this edge are off
                    )
            
        ax[0].set_ylabel('Anatomical distance (\u03BCm)')    
        plt.tight_layout()
        
        fig, ax = plt.subplots(figsize=(15,15))
        plt.scatter(f_dist[1],a_dist[1], s = 1, alpha = 0.8)
        ax.set_xlabel('Field distance (cm)')
        ax.set_ylabel('Anatomical distance (\u03BCm)')
        ax.set_title('Pairwise distance, B')
        
        # This is the correlation between the two 1D vectors that are matrices above. Makes more sense
        for sessionNo in range(nSessions):
            maskNan = ~np.isnan(f_dist[sessionNo])
            plot_2d_density(a_dist[sessionNo][maskNan], f_dist[sessionNo][maskNan])
        
    return centroid_coords, cell_centre, anatomical_distance, field_distance, f_dist, a_dist, corr_r

#%% Calculate the correlation between all place cells' rate maps

def corr_ratemaps(session_dict):
    
    # Define constants and variables
    nSessions = len(session_dict['dfNAT'])
    placecell_dict = session_dict['Placecell']

    corrs = {}
    
    for sessionNo in range(nSessions): 
         key = 'NAT'+str(sessionNo)
         ratemaps = session_dict['Ratemaps']['df'+key]
         nPC = len(placecell_dict[key][0])
         corrs[key] = np.full([nPC, nPC], np.nan)
         
         for row in range(nPC):
             PC1 = placecell_dict[key][0][row] 
             rmap = ratemaps['N'+str(PC1)]
             
             for col in range(nPC):
                 PC2 = placecell_dict[key][0][col] 
                 testmap = ratemaps['N'+str(PC2)]
                 
                 if PC1 >= PC2: continue
                 elif PC1 < PC2:
                     corrs[key][row,col] = pearsonr(rmap.flatten(), testmap.flatten())[0]
                     if np.isnan(corrs[key][row, col]): print(key,row,col)
    
    return corrs

#%% Plot heatmaps + histograms with rate map correlation and anatomical distance

def corr_anatDist(anatomical_distance, field_distance, corrs, **kwargs):
    
    plotter = kwargs.get('plotter', False)
    nSessions = len(corrs.keys())

    #Plot for sessions separately
    dfs = {}    
    
    for sessionNo in range(nSessions): 
        key = 'NAT'+str(sessionNo)
        
        mask = np.triu(np.ones_like(anatomical_distance[key], dtype=bool),1)
        
        d = {'a_dist': anatomical_distance[key][mask],
             'r': corrs[key][mask]}
        dfs[key] = pd.DataFrame(d)
        
        if plotter == True:
            s = sns.jointplot(x='a_dist', y='r', data=dfs[key], kind='hex', cmap = 'OrRd',
                              marginal_kws = {'bins': 25, 'color': '#d83221', 'alpha': 0.9},
                              xlim = (0,450))
            s.ax_joint.grid(False)
            s.ax_marg_y.grid(False)
            s.set_axis_labels('Anatomical distance (µm)', 'Rate map correlation (r)')
            s.fig.suptitle(sessionName[sessionNo], y=1.02)
        
            plt.savefig('N:/axon2pmini/Illustrations/'+key+'_img.svg', format = 'svg')            
        
    # Plot for all sessions together    
    all_corr_df = pd.concat([dfs['NAT0'], dfs['NAT1'], dfs['NAT2']])
    all_corr_df.reset_index(inplace = True, drop = True)
    
    # Add a mask to remove instances where the firing field was not found (hence, the cell_centre is NaN, and the distance NaN too)
    nanMask = ~np.isnan(all_corr_df['a_dist'])
    
    all_corr_df = all_corr_df.loc[nanMask,:]
    
    if plotter == True:
        s = sns.jointplot(x='a_dist', y='r', data=all_corr_df, kind='hex', cmap = 'OrRd',
                          marginal_kws = {'bins': 25, 'color': '#d83221', 'alpha': 0.9},
                          xlim = (0,450))
        s.ax_joint.grid(False)
        s.ax_marg_y.grid(False)
        s.set_axis_labels('Anatomical distance (µm)', 'Rate map correlation (r)')
        s.fig.suptitle('All sessions', y=1.02)
        
        plt.savefig('N:/axon2pmini/Illustrations/sessions_heatmap_img.svg', format = 'svg')    
    
    # Plot the binned data with reg line
    rVal = linregress(all_corr_df)[2]
    
    if plotter == True:
        fig, ax = plt.subplots()
        g = sns.regplot(x="a_dist", y="r", data=all_corr_df, ax = ax, x_bins = 35, x_ci = 95, ci = 95, color = '#33638d', 
                        scatter_kws = {'s': 0.1, 'alpha': 0.75}, line_kws = {'color': '#d83221', 'alpha': 1}) 
        g.set(xlabel=None, ylabel=None) 
        plt.suptitle('Mean of binned data for anatomical vs. rate map correlatiton')
        ax.set_title('R\u00B2 = '+str(round(rVal**2,5)))
        ax.set_xlabel('Anatomical distance (µm)')
        ax.set_ylabel('Correlation coefficient (r)') 
        ax.set_ylim([0,0.1])
        ax.spines[['top', 'right']].set_visible(False)
    
        plt.savefig('N:/axon2pmini/Illustrations/sessions_linreg_img.svg', format = 'svg')    

    if plotter == True:  
    
        #Plot correlation matrix between field distance and anatomical distance
         
        # This is a matrix of the correlation between two matrices - doesn't make too much sense
        fig, ax = plt.subplots(1,nSessions)  
        fig.suptitle('Correlation matrix of place field and anatomical distance for place cells')
         
        for sessionNo in range(nSessions):
             
            # Create a mask for all NaN values (are similar for field and anatomical distance)
            nanMask = np.isnan(field_distance['NAT'+str(sessionNo)])
            
            field_array = np.ma.array(field_distance['NAT'+str(sessionNo)], mask = nanMask)
            anatomical_array = np.ma.array(anatomical_distance['NAT'+str(sessionNo)], mask = nanMask)
            
            # Does this make sense?
            corr_matrix = np.ma.corrcoef(field_array, anatomical_array)
        
        if sessionNo == nSessions-1:
            sns.heatmap(corr_matrix, ax=ax[sessionNo],cmap='viridis', cbar=True)
            ax[sessionNo].set_title(str(sessionName[sessionNo]))
        else: 
            sns.heatmap(corr_matrix, ax=ax[sessionNo],cmap='viridis',cbar=False)
            ax[sessionNo].set_title(str(sessionName[sessionNo]))
                
        plt.tight_layout()
    
    return dfs, all_corr_df, rVal

#%% Strict criteria: Intersession stabiliy (A-A')

def inter_stability(session_dict, centroid_coords, **kwargs):
    
    plotter = kwargs.get('plotter', False)
    placecell_dict = session_dict['Placecell']
    nSessions = len(session_dict['dfNAT'])
    
    # Get only A-A' stable PCs, and those of these that are PCs in B as well
    
    # Find predefined place cells without a place field (=data lost)
    dataLoss = {}
    for sessionNo in range(nSessions): 
        key = 'NAT'+str(sessionNo)
        dataLoss[key] = np.where(np.isnan(centroid_coords[key][:,0]))[0]
        
    # Get NATEX index for predefined place cells
    A = placecell_dict['NAT0'][0]
    B = placecell_dict['NAT1'][0]
    A2 = placecell_dict['NAT2'][0]
    
    # Remove the instances where there is no place field placefield['NATx'][dataLoss['NATx']] should be NaN, NaN
    A = np.delete(A, dataLoss['NAT0'])
    B = np.delete(B, dataLoss['NAT1'])
    A2 = np.delete(A2, dataLoss['NAT2'])
    
    # Get cells in A that are also in A', and cells that are in both A and A' that are in B
    stableA = A[np.isin(A,A2)]
    stableAinB = stableA[np.isin(stableA,B)]
    
    # Predefine variables
    corr_stats = np.full([len(stableAinB),2], np.nan)
    shuffle_stats = np.full([len(stableAinB),2], np.nan)
    
    # Find stable PC in A-A', initiate this through a parallell prosess
    numWorkers = 15
    futures = []
    nShuffle = 100
    
    with cf.ProcessPoolExecutor(max_workers=numWorkers) as pool:
        for cellNo in range(len(stableAinB)):
            PC = stableAinB[cellNo]
            
            futures.append(pool.submit(
                intersession_stability, # The function
                session_dict,           # Everything
                PC,                     # Place cell index (NATEX)
                cellNo,                 # Cell number, an iterable
                nShuffle,               # The number of shuffles to perform
            ))     
    
        for future in tqdm(cf.as_completed(futures), total=len(stableAinB)):
            jobNo = future.result()[2] 
            corr_stats[jobNo] = future.result()[0]
            shuffle_stats[jobNo] = future.result()[1]
                
    # Grab cells that are more stable than the 95th percentile of a shuffled distribution between A and A'
    strictPC = stableAinB[corr_stats[:,0] > shuffle_stats[:,1]]
    nonStable = stableAinB[corr_stats[:,0] <= shuffle_stats[:,1]]
    
    # Check the stable and non-stable cells
    if plotter == True:

        # Plot the non-stable cells
        for x in nonStable:
            fig, ax = plt.subplots(1,2, sharex = True, sharey = True)
            fig.suptitle('N'+str(x))
            ax[0].imshow(session_dict['Ratemaps']['dfNAT0']['N'+str(x)])
            ax[0].axis('off')
            ax[1].imshow(session_dict['Ratemaps']['dfNAT2']['N'+str(x)])
            ax[1].axis('off')
            plt.tight_layout()
           
        # Plot some stable cells    
        for x in strictPC:
            if x < 10:
                fig, ax = plt.subplots(1,2, sharex = True, sharey = True)
                fig.suptitle('N'+str(x))
                ax[0].imshow(session_dict['Ratemaps']['dfNAT0']['N'+str(x)])
                ax[0].axis('off')
                ax[1].imshow(session_dict['Ratemaps']['dfNAT2']['N'+str(x)])
                ax[1].axis('off')
                plt.tight_layout()  
      
    return strictPC
 
#%%
def strict_PC_calc(session_dict, centroid_coords, cell_centre, strictPC, binning, scale, **kwargs):    
    
    plotter = kwargs.get('plotter', False)
    nSessions = len(session_dict['dfNAT'])
    placecell_dict = session_dict['Placecell']
    
    # For the strict cells, get the anatomical and field distance 
    strict_fieldCoords, strict_anatCoords = {}, {}
    strict_field_distance, strict_anatomical_distance = {}, {}
    strict_f_dist, strict_a_dist = [[]]*nSessions, [[]]*nSessions
    
    for sessionNo in range(nSessions): 
        key = 'NAT'+str(sessionNo)
        PCs = placecell_dict[key][0]
        
        strict_fieldCoords[key] = centroid_coords[key][np.isin(PCs,strictPC)]
        strict_anatCoords[key] = cell_centre[key][np.isin(PCs,strictPC)]
    
        # For the strict cells, see if the field distance changes for cell pairs in A, B and A'
        strict_field_distance[key] = (np.linalg.norm(strict_fieldCoords[key] - strict_fieldCoords[key][:,None], axis=-1))*binning
        strict_anatomical_distance[key] = (np.linalg.norm(strict_anatCoords[key] - strict_anatCoords[key][:,None], axis=-1))/scale
    
        mask = np.triu(np.ones_like(strict_field_distance[key], dtype=bool),1)
    
        strict_f_dist[sessionNo] = strict_field_distance[key][mask==True]
        strict_a_dist[sessionNo] = strict_anatomical_distance[key][mask==True]
        
        # Plot the anatomical distance against the field distance
        if plotter == True: 
            plot_2d_density(strict_a_dist[sessionNo], strict_f_dist[sessionNo])
    
    # Get the difference in field distance between cell pairs in A-A' and A-B
    diff_A = strict_f_dist[0]-strict_f_dist[2]
    diff_B = strict_f_dist[0]-strict_f_dist[1]   
    
    diff_A_ci = st.t.interval(alpha=0.95, df=len(diff_A)-1, loc=np.mean(diff_A), scale=st.sem(diff_A)) 
    diff_B_ci = st.t.interval(alpha=0.95, df=len(diff_B)-1, loc=np.mean(diff_B), scale=st.sem(diff_B)) 
    
    diffs = {'Diff_A': diff_A, 'Diff_B': diff_B, 'ci_A': diff_A_ci, 'ci_B': diff_B_ci}
    
    # Plot the results
    if plotter == True:
    
        legends = [sessionName[0]+'-'+sessionName[2], sessionName[0]+'-'+sessionName[1]]
        
        # Histogram
        fig, ax = plt.subplots(1,1, figsize = (6,6))
        ax = plt.hist(diff_A, 100, alpha = 0.75, color = palette[100], label = legends[0])
        ax = plt.hist(diff_B, 100, alpha = 0.75, color = palette[200], label = legends[1])
        plt.legend()
        plt.xlabel('Difference (cm)')
        plt.ylabel('Count')
        plt.title('Pairwise field distance of '+str(len(strictPC))+' cells')
        
        plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg')   
        
        # Violin
        a = pd.DataFrame({ 'Group': np.repeat(legends[0], len(diff_A)), 'Difference (cm)': diff_A })
        b = pd.DataFrame({ 'Group': np.repeat(legends[1], len(diff_B)), 'Difference (cm)': diff_B })
        dfDiff = pd.concat([a,b], ignore_index = True)
        
        fig, ax = plt.subplots(1,2, sharey = True)
        plt.suptitle('Difference in place fields for pairs of PCs')             
        sns.violinplot(x = 'Group', y = 'Difference (cm)', data=dfDiff, ax=ax[0], palette = 'viridis' )
        sns.boxplot(x='Group', y = 'Difference (cm)', data=dfDiff,  ax=ax[1], fliersize = 0.25, palette = 'viridis')
        ax[1].set_ylabel(None)
        ax[1].tick_params(
             axis='y',          # Changes apply to this axis
             which='both',      # Both major and minor ticks are affected
             left=False,        # Ticks along this edge are off
             right=False,       # Ticks along this edge are off
             labelleft=False    # Labels along this edge are off
             )
        
        fig, ax = plt.subplots(1,1)
        ax.scatter(diff_A, diff_B, marker = '.', s = 1, alpha = 0.75, color = palette[80])
        ax.set_title('Pairwise place field distances for remapping')
        ax.set_xlabel(legends[0]+' (cm)')
        ax.set_ylabel(legends[1]+' (cm)')
        ax.set_aspect('equal', 'box')
    
        # Plot the distribution of place field centroid for the three sessions
        fig, ax = plt.subplots(figsize = (5,5))
        for sessionNo in range(nSessions):
            x, y = strict_fieldCoords['NAT'+str(sessionNo)][:,0]*2.5, strict_fieldCoords['NAT'+str(sessionNo)][:,1]*2.5
            ax.scatter(x,y, color = palette[65+75*sessionNo], label = sessionName[sessionNo])
            ax.set_aspect('equal', 'box')
            plt.legend(bbox_to_anchor=(1.04, 1), loc = 'upper left')
        plt.suptitle('Distribution of place field centres')
    
        # Plot the distribution of place field centroid for the three sessions per cell
        fig, ax = plt.subplots(figsize = (5,5)) 
        for cellNo in range(10):# range(len(strictPC)):
            for sessionNo in range(nSessions):
                x, y = strict_fieldCoords['NAT'+str(sessionNo)][cellNo,0]*2.5, strict_fieldCoords['NAT'+str(sessionNo)][cellNo,1]*2.5
                ax.scatter(x,y, color = palette[cellNo*28], label = sessionName[sessionNo])
                ax.set_aspect('equal', 'box')
        plt.suptitle('Centroid for 10 cells (color) in A-B-A\'') 
 
    return diffs, strict_anatomical_distance, strict_field_distance, strict_fieldCoords, strict_f_dist

#%% 

def calc_diffShuffle(strict_fieldCoords, strict_f_dist):

    binning = 2.5 # cm/bin
    nShuffles = 100
    
    shuffleCoords = strict_fieldCoords['NAT2'].copy()
    diff_Shuffle = np.full([len(strict_f_dist[0]), nShuffles], np.nan)
    
    for shuffleNo in range(nShuffles): 
        np.random.shuffle(shuffleCoords)
 
        shuffleDist = (np.linalg.norm(shuffleCoords - shuffleCoords[:,None], axis=-1))*binning
        
        mask = np.triu(np.ones_like(shuffleDist, dtype=bool),1)
        shuffleDist = shuffleDist[mask==True]  
        
        diff_Shuffle[:,shuffleNo] = strict_f_dist[0] - shuffleDist
        
    return diff_Shuffle
   
# #%%  For parallelling shuffling of place fields in A'

# # See shuffle_fields for usage

# def calc_distShuffle(dfNAT, strictPC, coordsAStrict, shuffleNo):
    
#     minPeak = 0.2
#     binning = 2.5 # cm/bin
    
#     jobNo = shuffleNo
    
#     coordsShuffle = np.full([len(strictPC), 2], np.nan)
    
#     for ii in range(len(strictPC)):
    
#         cellNo = strictPC[ii]
#         rmap_shuffle = shuffle_ratemap(dfNAT, cellNo)
    
#         fields, fields_map = op.analysis.place_field(rmap_shuffle, search_method = 'sep', min_peak = minPeak)
             
#         if len(fields) == 1: # Only 1 detected field
#             coordsShuffle[ii] = fields[0]['centroid_coords']
        
#         elif len(fields) > 1: # More than 1 detected field, take the most prominent one (highest mean rate)
#             field_mean_rates = []
            
#             for fieldNo in range(len(fields)): field_mean_rates.append(fields[fieldNo]['mean_rate'])
            
#             field_idx = np.where(field_mean_rates==max(field_mean_rates))[0][0]
           
#             coordsShuffle[ii] = fields[field_idx]['centroid_coords']
    
#     # Calculate the distance from the field in A to the field in shuffled A'       
#     resultDistance = np.linalg.norm(coordsAStrict - coordsShuffle[:,None], axis=-1)/binning
    
#     mask = np.triu(np.ones_like(resultDistance, dtype=bool),1)
#     resultDistance =  resultDistance[mask==True]
    
#     return  jobNo, resultDistance

# #%% Simulate random n points in 80x80 grid, get distance between points and plot results

# def shuffle_fields(session_dict, strictPC, strict_field_distance, centroid_coords, diffs, **kwargs):
    
#     plotter = kwargs.get('plotter', False)
    
#     PC_A = session_dict['Placecell']['NAT0'][0]
#     idxA  = np.isin(PC_A, strictPC)
#     coordsAStrict = centroid_coords['NAT0'][idxA,:] # Is in NATEX idx from PCs in placecell_dict['NAT0'][0]
    
#     dfNAT = session_dict['dfNAT']['dfNAT2']
    
#     mask = np.triu(np.ones_like(strict_field_distance['NAT0'], dtype=bool),1)
#     strict_f_dist = strict_field_distance['NAT0'][mask==True]
    
#     # Parallell for A-A'
#     numWorkers = 18
#     futures = []
    
#     nShuffles = 100
#     dfNAT = session_dict['dfNAT']['dfNAT2']
#     distShuffle = np.full([len(strict_f_dist), nShuffles], np.nan)
#     diffShuffle = np.full([len(strict_f_dist), nShuffles], np.nan)
    
#     with cf.ProcessPoolExecutor(max_workers=numWorkers) as pool:
#         for shuffleNo in range(nShuffles):
#             futures.append(pool.submit(
#                 calc_distShuffle,   # The function
#                 dfNAT,              # Timestamps, headpos, speed, cell activity
#                 strictPC,           # NATEX indices of stable place cells (A-B-A')
#                 coordsAStrict,      # The field coordinates of the stable place cells
#                 shuffleNo           # Shuffle number, the iterable (NATEX idx)
#             ))     
    
#         for future in tqdm(cf.as_completed(futures), total=(nShuffles)):
#             jobNo = future.result()[0]
#             distShuffle[:,jobNo] = future.result()[1]
#             diffShuffle[:,jobNo] = strict_f_dist - distShuffle[:,jobNo]

#     if plotter == True:
        
#         nSim = 100
#         diff_A, diff_B = diffs['Diff_A'], diffs['Diff_B']
        
#         sim_diff = np.full([comb(len(strictPC),2), nSim], np.nan)
#         legends = [sessionName[0]+'-'+sessionName[2], sessionName[0]+'-'+sessionName[1]]
        
#         for sim in range(nSim):    
#             dataA = np.random.uniform(0,80,[len(strictPC),2]) 
#             dataB = np.random.uniform(0,80,[len(strictPC),2])
            
#             #fig, ax = plt.subplots()
#             #ax.scatter(dataA[:,0],dataA[:,1], color = palette[80])
#             #ax.scatter(dataB[:,0],dataB[:,1], color = palette[180])
#             #ax.set_xlim([0,80])
#             #ax.set_ylim([0,80])
#             #ax.set_aspect('equal', 'box')
            
#             data_distA = np.linalg.norm(dataA - dataA[:,None], axis=-1)
#             data_distB = np.linalg.norm(dataB - dataB[:,None], axis=-1)
            
#             mask = np.triu(np.ones_like(data_distA, dtype=bool),1)
            
#             sim_distA = data_distA[mask==True]
#             sim_distB = data_distB[mask==True]
            
#             sim_diff[:,sim] = sim_distA - sim_distB
    
#         # Histogram simulation, version 1
#         fig, ax = plt.subplots(1,1, figsize = (5,5))
#         ax.hist(diff_A, bins = 75, alpha = 0.75, color = '#26828E', label = legends[0])
#         ax.hist(diff_B, bins = 75, alpha = 0.75, color = '#6CCD5A', label = legends[1])
#         ax.hist(np.mean(sim_diff, axis=1), bins = 75, alpha = 0.5, color = '#d83221', label = 'Simulation')
#         ax.set_xlabel('Difference (cm)')
#         ax.set_ylabel('Count')
#         ax.set_title('Place field distance for PC pairs in A-A\' and A-B, mean of sim')
#         plt.legend()
        
#         # Histogram simulation, version 2
#         fig, ax = plt.subplots(1,1, figsize = (5,5))
#         ax.hist(diff_A, bins = 100, alpha = 0.75,  color = '#31688E', label = legends[0])
#         ax.hist(diff_B, bins = 100, alpha = 0.75, color = '#35B779', label = legends[1])
#         sim_choice = np.random.choice(len(sim_diff.flatten()), len(diff_A), replace = False)
#         ax.hist(sim_diff.flatten()[sim_choice], bins = 100, alpha = 0.75, color = '#B5DE2B', label = 'Simulation')
#         ax.set_title('Place field distance for PC pairs in A-A\' and A-B')
#         ax.set_xlabel('Difference (cm)')
#         ax.set_ylabel('Count')
#         plt.legend()
        
#         plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg')   
    
#         # Two-sample t-test
#         tstats_sim = ttest2(diff_B, sim_diff.flatten()[sim_choice])
#         print(tstats_sim)
    
#         # Plot the data as violins
#         df_A = pd.DataFrame({ 'Group': np.repeat(legends[0], len(diff_A)), 'Difference (cm)': diff_A })
#         df_B = pd.DataFrame({ 'Group': np.repeat(legends[1], len(diff_B)), 'Difference (cm)': diff_B })
#         df_Sim = pd.DataFrame({ 'Group': np.repeat('Simulation', len(sim_diff.flatten()[sim_choice])), 'Difference (cm)': sim_diff.flatten()[sim_choice]})
#         dfData = pd.concat([df_A, df_B, df_Sim])
    
#         fig, ax = plt.subplots(sharey = True, figsize = (5,5))
#         sns.violinplot(x = 'Group', y = 'Difference (cm)', data=dfData, ax=ax, palette = 'viridis' )
#         ax.set_title('Place field distance for place cell pairs')   
#         ax.set_xlabel(None)
        
#         plt.tight_layout()
        
#         plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg')     
    
    # return diffShuffle, distShuffle

#%% Correlate rate maps pairwise and see the anatomical distance between them

def ratemap_correlation_strictPC(session_dict, strictPC, strict_anatomical_distance, **kwargs):
    
    plotter = kwargs.get('plotter', False) 
    nSessions = len(session_dict['dfNAT'])
    
    dfReg = {}
    
    # Iterate over sessions
    for sessionNo in range(nSessions):
        key = 'NAT'+str(sessionNo) 
    
        # Get test cell and initiate variable
        for testCell in range(1): #range(len(strictPC)):
            
            rd = np.full([len(strictPC),3], np.nan)
            rm1 = session_dict['Ratemaps']['df'+key]['N'+str(strictPC[testCell])].flatten()
            
            # Test this cell up to the other cells (including self)
            for cellNo in range(len(strictPC)): 
                if not testCell == cellNo:
                    rd[cellNo,0] = strictPC[cellNo] # NATEX index
                    rm2 = session_dict['Ratemaps']['df'+key]['N'+str(strictPC[cellNo])].flatten()
                    rd[cellNo,1] = pearsonr(rm1,rm2)[0] # Correlation coefficient
                    rd[cellNo,2] = strict_anatomical_distance[key][testCell,cellNo] # Distance is already scaled
            
            # Sort by the correlation and plot the distribution of Rs
            sortIdx = np.argsort(rd[:,1])[::-1] # Indexes of sorted data
            
            sort_r = rd[:,1][sortIdx] 
            sort_d = rd[:,2][sortIdx]
            
            # Remove NaN from autocorrelation 
            sort_r = np.delete(sort_r, np.isnan(sort_r))
            sort_d = np.delete(sort_d, np.isnan(sort_d))
            sortIdx = np.delete(sortIdx, np.where(sortIdx == 0))-1
            
            """ PLOTTING TO SEE IF THINGS ARE CORRECT
            plt.figure()
            plt.plot(sort_r) 
            plt.title('Correlation of cell '+str(strictPC[testCell])+' to the rest')
            plt.xlabel('Cell number')
            plt.ylabel('Pearson R')
            plt.xlim([0,len(sort_r)])
            
            # Plot the values in a scatter, the latter should be similar if sorted correctly
            plt.figure()
            plt.title('Rate map correlation of different PCs')
            plt.scatter(sort_r, sort_d)
            plt.xlabel('Pearson R')
            plt.ylabel('Anatomical distance (microns)')
            
            plt.figure()
            plt.title('Rate map correlation of different PCs')
            plt.scatter(rd[:,1], rd[:,2])
            plt.xlabel('Pearson R')
            plt.ylabel('Anatomical distance (microns)')
            """
            
            # Do a linear regressin on the data (normalize first?)
            r_norm = sort_r/max(sort_r)
            d_norm = sort_d/max(sort_d)
    
            # Use the sorted because the NaN values are removed (doesn not need to be sorted)
            regStats = linregress(sort_r, sort_d)
            regStatsNorm = linregress(r_norm, d_norm)
            CI = regStatsNorm.stderr*1.96
            
            # Print some info (+ a little message if the reg.line is significant)  
            print('Session '+sessionName[sessionNo]+':')
            print('Slope (CI 95): '+str(round(regStatsNorm.slope,3))+' ('+str(round(regStatsNorm.slope-CI,2))+
                      ', '+str(round(regStatsNorm.slope+CI,2))+')') 
            print('p-value: '+str(round(regStatsNorm[3],2)))
            
                    
            if regStatsNorm.slope - CI > 0 or regStatsNorm.slope + CI < 0:
                print('Significantly correlated!')      
            
            print('\n')
            
            # Plot scatter with regression line
            if plotter == True:
                plt.figure()
                plt.title(sessionName[sessionNo]+': Rate map correlation of different PCs')
                plt.scatter(r_norm, d_norm, color = palette[80])
                plt.xlabel('Pearson R')
                plt.ylabel('Anatomical distance (norm)')      
                regLine = regStatsNorm[0]*r_norm+regStatsNorm[1]       
                plt.plot(r_norm, regLine, color = contrast[190])
            
                # Plot scatter with confidence interval, sns.regplot() calculates this automatically
                res = regStatsNorm
            
                y1 = (res.slope-res.stderr*1.96)*np.abs(r_norm) + (res.intercept-res.intercept_stderr*1.96)
                y2 = (res.slope+res.stderr*1.96)*np.abs(r_norm) + (res.intercept+res.intercept_stderr*1.96)
                
                data = {'r': r_norm, 'dist': d_norm}
                df = pd.DataFrame(data)
                
                # The actual plot
                fig, ax = plt.subplots(2,1, sharex = True, sharey = True)
                plt.suptitle(sessionName[sessionNo]+': Rate map correlation vs. anatomical distance')
                fig.supylabel('Anatomical distance (norm)')
                fig.supxlabel('Correlation coefficient (r)')
                   
                ax[0].scatter(r_norm, d_norm, color = palette[80], alpha = 0.8)
                ax[0].plot(r_norm, res.intercept + res.slope*r_norm, color = 'crimson')
                ax[0].fill_between(r_norm, y1, y2, alpha = 0.5, color = 'crimson')
                   
                g = sns.regplot(x="r", y="dist", data=df, ax= ax[1], color = palette[80], line_kws ={'color': contrast[190], 'alpha': 1})
                g.set(xlabel=None, ylabel=None) 
                
                plt.tight_layout()
        
        # Do it again, but this time for all cells within a session (not just one by one)
        
        # Initate variable 
        r = np.full([2,len(strictPC),len(strictPC)], np.nan) 
        
        # Get test cell
        for testCell in range(len(strictPC)):
            
            rm1 = session_dict['Ratemaps']['df'+key]['N'+str(strictPC[testCell])].flatten()
            
            # Test this cell up to the other cells (including self)
            for cellNo in range(len(strictPC)):
                if testCell < cellNo: # To not recompute already computed cell pairs
                    r[1,testCell, cellNo] = strictPC[cellNo] #dim1 is the NATEX index
                    rm2 = session_dict['Ratemaps']['df'+key]['N'+str(strictPC[cellNo])].flatten()
                    r[0,testCell, cellNo] = pearsonr(rm1,rm2)[0] # dim0 is the r-values
        
        mask = np.triu(np.ones_like(r[0], dtype=bool),1)
        
        # Sort by the correlation and plot the distribution of Rs
        r_all = r[0][mask]
        sort_r = np.sort(r_all)[::-1] # Sort data
        sortIdx = np.argsort(r_all)[::-1] # Indexes of sorted data
        
        # Get the anatomical distance between the cells
        d_all = strict_anatomical_distance['NAT0'][mask]
        sort_d = d_all[sortIdx] # Data sorted by sorting indecies in r
        
        """ PLOTTING TO SEE IF THINGS ARE CORRECT
        plt.figure()
        plt.plot(sort_r, color = '#31668e') # Should be a descending curve
        plt.title('Decending R for place cell pairs (n = '+str(len(strictPC))+' cells)')
        
        plt.figure()
        plt.title('Rate map correlation of different PCs')
        plt.scatter(sort_r, sort_d, s = 1.5, alpha = 0.75, color = '#31668e')
        plt.xlabel('Pearson R')
        plt.ylabel('Anatomical distance (\u03BCm)')
        
        # Sorting does not affect the scatter? If not, then it is correctly sorted
        plt.figure()
        plt.title('Rate map correlation of different PCs')
        plt.scatter(r_all, d_all, s = 1.5, alpha = 0.75, color = '#31668e')
        plt.xlabel('Pearson R')
        plt.ylabel('Anatomical distance (\u03BCm)')
        """
        
        # Do a linear regression on the data (normalize first)
        r_norm = r_all/max(r_all)
        d_norm = d_all/max(d_all)
        
        regStats = linregress(r_all,d_all)
        regStatsNorm = linregress(r_norm, d_norm)
        
        rsquare = round(regStats.rvalue**2,5)
        CI = regStatsNorm.stderr*1.96
        
        print('Normalized: \nSlope (CI 95): '+str(round(regStatsNorm[0],3))+' ('
              +str(round(regStatsNorm.slope-CI,2))+', '+str(round(regStatsNorm.slope+CI,2))+')'
              '\n'+'p-value: '+str(round(regStatsNorm[3],3))+'\n')
        
        # Plot scatter with regression line
        dfReg[key] = pd.DataFrame({'r': r_all, 'dist': d_all})
        
        if plotter == True:
            
            fig, ax = plt.subplots()
            g = sns.regplot(x="dist", y="r", data=dfReg, ax= ax, color = palette[80], 
                            scatter_kws = {'s': 0.1, 'alpha': 0.5}, line_kws = {'color': contrast[190], 'alpha': 1}) 
            g.set(xlabel=None, ylabel=None) 
            plt.suptitle(sessionName[sessionNo]+': Rate map correlation vs. anatomical distance')
            ax.set_ylabel('Correlation coefficient (r)')
            ax.set_xlabel('Anatomical distance (µm)')
            
            fig, ax = plt.subplots(figsize = (5,5))
            g = sns.regplot(x="dist", y="r", data=dfReg, x_bins = 25, x_ci = 95, ci = 95, ax= ax, color = palette[80], 
                            scatter_kws = {'s': 0.1, 'alpha': 0.5}, line_kws = {'color': contrast[190], 'alpha': 1}) 
            g.set(xlabel=None, ylabel=None) 
            plt.suptitle(sessionName[sessionNo]+': Rate map correlation vs. anatomical distance')
            ax.set_title('R\u00B2: '+str(rsquare))
            ax.set_ylabel('Correlation coefficient (r)')
            ax.set_xlabel('Anatomical distance (µm)')
            
            plt.savefig('N:/axon2pmini/Illustrations/'+key+'img.svg', format = 'svg')   

    return dfReg

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

topography_pairwise_distance_dict = {}

if __name__ == "__main__":
    
    for session in sessions: 
        
        # Load session_dict
        session_dict = pickle.load(open(session+'\session_dict.pickle','rb'))
        print('Successfully loaded session_dict from: '+str(session))
        
        key = session_dict['Animal_ID'] + '-' + session_dict['Date']
        
        # Calculate fields and distances
        centroid_coords, cell_centre, anatomical_distance, field_distance, f_dist, a_dist, corr_r = calc_field_distance(session_dict, plotter = False)
        
        # Correlate ratemaps between all place cells
        corrs = corr_ratemaps(session_dict)
        
        # Calculate the correlation and anatomical distance between place cells
        dfs, all_corr_df, rVal = corr_anatDist(anatomical_distance, field_distance, corrs, plotter = False)
        
        # Calculate which place cells are stabile across all three sessions
        strictPC = inter_stability(session_dict, centroid_coords, plotter = False)
        
        # Calculate the differences in distance and actual distances for the strict place cells
        diffs, strict_anatomical_distance, strict_field_distance, strict_fieldCoords, strict_f_dist = strict_PC_calc(session_dict, centroid_coords, cell_centre, strictPC, binning, scale, plotter = False)   

        tstats_paired = ttest_paired(diffs['Diff_A'], diffs['Diff_B'])
        
        # Shuffle the PC IDs in NAT2 and compare to the pairwise distances between A
        diff_Shuffle = calc_diffShuffle(strict_fieldCoords, strict_f_dist)
        
        # Shuffle the rate map in A' and calculate the difference to that place field from A (A-A' Shuffle)
        # diffShuffle, distShuffle = shuffle_fields(session_dict, strictPC, strict_field_distance, centroid_coords, diffs, plotter = False)
        
        # Correlate rate maps pairwise for strict PC and calculate the anatomical distance between them
        dfReg = ratemap_correlation_strictPC(session_dict, strictPC, strict_anatomical_distance, plotter = False)
        
        # Put the results in a dict
        topography_pairwise_distance_dict[key] = {'a_dist': a_dist,
                                                  'f_dist': f_dist,
                                                  'corrs': corrs,
                                                  'dfs': dfs,
                                                  'diffs': diffs,
                                                  'strict_anatomical_distance': strict_anatomical_distance,
                                                  'strict_field_distance': strict_field_distance,
                                                  'diff_Shuffle': diff_Shuffle, # NaN values occur if there is no field in the shuffled data
                                                  # 'distShuffle': distShuffle, # NaN values occur if there is no field in the shuffled data
                                                  'dfReg': dfReg}
        
        # Store the output
        with open(results_folder+'/topography_pairwise_distance_dict.pickle','wb') as handle:
            pickle.dump(topography_pairwise_distance_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Successfully saved results_dict in '+ results_folder)
        
#%% Make the heat maps for each session

for session in sessions: 
    
    # Load session_dict
    session_dict = pickle.load(open(session+'\session_dict.pickle','rb'))
    print('Successfully loaded session_dict from: '+str(session))
    
    key = session_dict['Animal_ID'] + '-' + session_dict['Date']
    
    # Calculate fields and distances
    centroid_coords, cell_centre, anatomical_distance, field_distance, f_dist, a_dist, corr_r = calc_field_distance(session_dict, plotter = False)
    
    # Correlate ratemaps between all place cells
    corrs = corr_ratemaps(session_dict)
    
    # Calculate the correlation and anatomical distance between place cells
    dfs, all_corr_df, rVal = corr_anatDist(anatomical_distance, field_distance, corrs, plotter = False)
    
    # Plot the field and anatomical distance
    aDist = np.concatenate([a_dist[0], a_dist[1]])
    fDist = np.concatenate([f_dist[0], f_dist[1]])
    
    all_dist = pd.DataFrame({'a_dist': aDist[~np.isnan(aDist)], 'f_dist': fDist[~np.isnan(aDist)]})

    # Plot all in one big plot (hex heat map + hist)
    s = sns.jointplot(x='a_dist', y='f_dist', data=all_dist, kind='hex', cmap = 'OrRd',
                      marginal_kws = {'bins': 25, 'color': '#d83221', 'alpha': 0.9},
                      xlim = (0,450), ylim = (0,80))
    s.ax_joint.grid(False)
    s.ax_marg_y.grid(False)
    s.set_axis_labels('Anatomical distance (µm)', 'Place field distance (cm)')
    s.fig.suptitle(key, y=1.02)

    plt.savefig(r'N:/axon2pmini/Article/Figures/Supplementary/pairwiseFdist_'+key+'.svg', format = 'svg')   
    
    # Plot the correlation and anatomical distance
    a_dist_corr = np.concatenate([dfs['NAT0']['a_dist'], dfs['NAT1']['a_dist']])
    r_dist_corr = np.concatenate([dfs['NAT0']['r'], dfs['NAT1']['r']])  
    
    nanMask = ~np.isnan(a_dist_corr)
    
    all_corrs = pd.DataFrame({'a_dist': a_dist_corr[nanMask], 'r': r_dist_corr[nanMask]})
    
    # Plot all in one big plot (hex heat map + hist)
    s = sns.jointplot(x='a_dist', y='r', data=all_corrs, kind='hex', cmap = 'OrRd',
                      marginal_kws = {'bins': 25, 'color': '#d83221', 'alpha': 0.9},
                      xlim = (0,450), ylim = (-0.4,0.6))
    s.ax_joint.grid(False)
    s.ax_marg_y.grid(False)
    s.set_axis_labels('Anatomical distance (µm)', 'Tuning map correlation (r)')
    s.fig.suptitle(key, y=1.02)

    plt.savefig(r'N:/axon2pmini/Article/Figures/Supplementary/pairwiseCorr_'+key+'.svg', format = 'svg')   
    
#%% Check the place field detection - IT IS OK!  
"""    
    for pc in placecell_dict['NAT0'][0]:
        ratemap = session_dict['Ratemaps']['dfNAT0']['N'+str(pc)]
        fields = session_dict['Placefields']['NAT0']['N'+str(pc)][-1]
        
        centroid = []
        peak = []
        
        for fieldNo in range(len(session_dict['Placefields']['NAT0']['N'+str(pc)][0])):
            centroid.append(session_dict['Placefields']['NAT0']['N'+str(pc)][0][fieldNo]['centroid_coords'])
            peak.append(session_dict['Placefields']['NAT0']['N'+str(pc)][0][fieldNo]['peak_coords'])
        
        fig, ax = plt.subplots(1,2, figsize = (5,3))
        ax[0].imshow(ratemap)
        ax[1].imshow(fields)
        for x in range(len(session_dict['Placefields']['NAT0']['N'+str(pc)][0])):
            ax[0].scatter(centroid[x][1], centroid[x][0], marker = 'o', s = 25, color = 'magenta')
            ax[0].scatter(peak[x][1], peak[x][0], marker = 'x', s = 25, color = 'cyan')
    
            ax[1].scatter(centroid[x][1], centroid[x][0], marker = 'o', s = 25, color = 'magenta')
            ax[1].scatter(peak[x][1], peak[x][0], marker = 'x', s = 25, color = 'cyan')
            
        plt.suptitle('N'+str(pc))    
        plt.tight_layout()
    
    # Check them again, but use the extracted centroid coords from the field with highest mean rate
    for cellNo in range(len(placecell_dict['NAT0'][0])): # Python index
        pc = placecell_dict['NAT0'][0][cellNo] # NATEX index
        ratemap = session_dict['Ratemaps']['dfNAT0']['N'+str(pc)]
        y, x = centroid_coords['NAT0'][cellNo]
        
        fig, ax = plt.subplots(1,1)
        ax.imshow(ratemap)
        ax.scatter(y, x, marker = 'o', s = 25, color = 'magenta')
        plt.suptitle('N'+str(pc))    
        plt.tight_layout()
"""    