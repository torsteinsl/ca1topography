# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:41:21 2024

@author: torstsl

Moran by ROI - NEAREST - DISTANCE: 
    As its brother, but uses only the correlation to the 1 nearest neighbour
    AND THIS USES FIELD DISTANCE RATHER THAN CORRELATION

- Load the list of sessions.
- Loop over each session to:
    - Load session-specific data.
    - Calculate projections and masks of the image.
    - Compute the distance matrix between cells.
    - Calculate correlations to neighboring place cells (PCs).
    - Prepare data for Moran's I calculation.
        - Features: The mean correaltion between neighbouring place cells
                    within at least minDist (25 um?) from each other based on
                    ROI centre. To be analyses, a place cell must have at least
                    two neighbours (that is, this is more than pairwise testing)
        - Weights: Handles as "islands", so that all cells are neighbours to all
                    other cells, but are weighted by the inverse as the distance
                    between then (see helper func calc_weights for details). 
    - Perform Moran's I calculation (global and local).
    - Store the results in a dictionary.
    - Save the results dictionary to a file.

"""

import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import pickle
import libpysal.weights
from esda.moran import Moran, Moran_Local
from src.cell_activity_analysis import get_placefields_distance

#%% Get the maximum intensitity projection image from the session and the place cell masks

def projection_masks(session_dict, **kwargs):
    
    plotter = kwargs.get('plotter', False)
    
    nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)
    placecell_dict = session_dict['Placecell']
    placecells = np.unique(np.concatenate([placecell_dict['NAT0'][0],placecell_dict['NAT1'][0],placecell_dict['NAT2'][0]]))
    
    maxproj = session_dict['ExperimentInformation']['ImagingOptions']['max_proj']
    meanimg = session_dict['ExperimentInformation']['ImagingOptions']['meanImg']
    yrange = session_dict['ExperimentInformation']['ImagingOptions']['yrange'] # Valid y-range used for cell detection
    xrange = session_dict['ExperimentInformation']['ImagingOptions']['xrange'] # Valid x-range used for cell detection
    
    # Get the masks for all cells in this experiment and the centre of the ROIs
    masks_max = np.full([nCells,maxproj.shape[0],maxproj.shape[1]], np.nan)
    masks_max_zero = np.zeros([nCells,maxproj.shape[0],maxproj.shape[1]])

    masks_mean = np.full([nCells,meanimg.shape[0],meanimg.shape[1]], np.nan)
    
    cell_centre = np.full([nCells,2],np.nan) 
    
    for cellNo in range(nCells):
        cell_centre[cellNo] = session_dict['ExperimentInformation']['CellStat'][cellNo]['med'][::-1] # Get cell centre, xy (anatomical)
        cell_centre[cellNo][0] -=  xrange[0]
        cell_centre[cellNo][1] -=  yrange[0]
        
        ypix_max = session_dict['ExperimentInformation']['CellStat'][cellNo]['ypix']-yrange[0] # List of all y-coordinates for this cell's mask
        xpix_max = session_dict['ExperimentInformation']['CellStat'][cellNo]['xpix']-xrange[0] # List of all x-coordinates for this cell's mask
        
        ypix_mean = session_dict['ExperimentInformation']['CellStat'][cellNo]['ypix'] # List of all y-coordinates for this cell's mask
        xpix_mean = session_dict['ExperimentInformation']['CellStat'][cellNo]['xpix'] # List of all x-coordinates for this cell's mask
        
        masks_max[cellNo][ypix_max, xpix_max] = 1 # Array corresponding to max proj with this cell's mask 
        masks_max_zero[cellNo][ypix_max, xpix_max] = 1 # Array corresponding to max proj with this cell's mask  
        masks_mean[cellNo][ypix_mean, xpix_mean] = 1 # Array corresponding to mean image with this cell's PCs mask  
    
    all_masks_max = np.nansum(masks_max,0) # Make one array with all masks for these cells
    all_masks_mean = np.nansum(masks_mean,0) # Make one array with all masks for these cells

    # Plot the max and mean images with masks from the cells
    if plotter == True: 
        
        fig, ax = plt.subplots()
        sns.heatmap(maxproj, cmap = 'gist_gray', square = True, cbar = False, ax = ax, xticklabels = False, yticklabels = False)
        
        # For loop over all masks of place cells to get them separately
        fig, ax = plt.subplots(1,2)    
        fig.suptitle('Place cell masks')
        ax[0].set_title('Max projection')
        ax[0].imshow(maxproj,cmap='gist_gray')
        ax[1].set_title('Mean image')
        ax[1].imshow(meanimg,cmap='gist_gray')
        ax[0].axis('off')  
        ax[1].axis('off')  
        for pc in placecells:
            ax[0].contour(masks_max[pc-1], colors = 'w', linewidths = 0.15, alpha = 0.75)
            ax[1].contour(masks_mean[pc-1], colors = 'w', linewidths = 0.15, alpha = 0.75)
        plt.tight_layout()   
            
        # Plot all cells' masks
        fig, ax = plt.subplots(1,2)    
        fig.suptitle('All cells masks')
        ax[0].set_title('Max projection')
        ax[0].imshow(maxproj,cmap='gist_gray')
        ax[0].imshow(all_masks_max, cmap='gist_gray', alpha = 0.5)
        ax[0].axis('off')    
        
        ax[1].set_title('Mean image')
        ax[1].imshow(meanimg,cmap='gist_gray')
        ax[1].imshow(all_masks_mean, cmap='gist_gray', alpha = 0.5)
        ax[1].axis('off')  
        
        plt.tight_layout()  
        
        # Plot each ROI separately with diverging colors
        colors = sns.color_palette('Paired', masks_max.shape[0]).as_hex()
        
        fig, ax = plt.subplots()    
        ax.set_title('Max projection with all cells\' masks')
        sns.heatmap(maxproj, cmap = 'gist_gray', square = True, cbar = False, ax = ax, xticklabels = False, yticklabels = False)
        ax.axis('off')  
        for ii in range(masks_max.shape[0]):
            ax.contourf(masks_max[ii], colors = colors[ii], alpha = 0.2)
        plt.tight_layout()   
        
        # Plot each ROI separately with diverging colors, only place cells
        fig, ax = plt.subplots()    
        ax.set_title('Max projection with all cells\' masks')
        sns.heatmap(maxproj, cmap = 'gist_gray', square = True, cbar = False, ax = ax, xticklabels = False, yticklabels = False)
        ax.axis('off')  
        for ii in (placecells-1):
            ax.contourf(masks_max[ii], colors = colors[ii], alpha = 0.2)
        plt.tight_layout()

    return maxproj, meanimg, cell_centre, masks_max, masks_mean, masks_max_zero

#%% Calculate the correlation to its nearest neighbour place cell
def calc_fielddist(session_dict, distanceMtx):

    placefield, placefield_coords, field_distance = get_placefields_distance(session_dict)

    nSessions = len(session_dict['dfNAT'])
    placecell_dict = session_dict['Placecell']
    
    near_dist = {}
    
    for sessionNo in range(nSessions):
        key = 'NAT'+str(sessionNo)
        
        pcs = placecell_dict[key][0] # NATEX ID
        near_dist[key] = np.full([len(pcs)], np.nan)
        
        # Find distance from each PC to its nearest neighbouring PC
        for no, pCell in enumerate(pcs): # NATEX ID  
            
            # Make submatrix of just place cells
            rows, cols = pcs-1, pcs-1 # From NATEX to Pythonic
            pcMtx = distanceMtx[np.ix_(rows, cols)]
            
            nearestDist = np.nanmin(pcMtx[no, :]) # Pythonic ID
            nearestPC = np.where(pcMtx[no,:] == nearestDist)[0][0] # Pythonic ID
            
            near_dist[key][no] = field_distance[key][no, nearestPC]

    return near_dist    

#%%

def prepare_moran(near_dist, distanceMtx, cell_centre):
         
    features = {}
    W = {}
    ROIs =  {}
    
    for NAT in near_dist.keys():
    
        features[NAT] = near_dist[NAT]
        ROIs[NAT] = cell_centre[np.where(~np.isnan(features[NAT]))[0], :]
        
        # Calculate the weights, set diagonal to 0 (wij, i==j --> 0).
        w = 1/distanceMtx
        w[np.diag_indices_from(w)] = 0 # Set NaN to 0 for weight of pair with itself
    
        # Remove NaNs
        nanMask = np.where(np.isnan(features[NAT]))
        
        features[NAT] = np.delete(features[NAT], nanMask)
        w = np.delete(w, nanMask, axis = 0)
        w = np.delete(w, nanMask, axis = 1)
        
        keys = []
        for i in range(len(features[NAT])): keys.append(i)
        keys = np.array(keys)
        
        # Create dict of all neighbours and the weight to all neighbours 
        # All are ROIs are "islands", and thus "neighbours to all" with different weights
        neighbours, weights = {}, {}
        for i in range(len(features[NAT])): 
            others = keys[~np.isin(keys,i)]
            neighbours[keys[i]] = others
            weights[keys[i]] = w[i, others]

        # Get the weights object
        W[NAT] = libpysal.weights.W(neighbours, weights)
        
    return features, W, ROIs

#%% Do Moran's I on the matrix
def calc_morans(features, W, maxproj, ROIs, **kwargs):

    plotter = kwargs.get('plotter', False)
    
    moran, moran_loc = {}, {}
    
    NATs = ['NAT0', 'NAT1']        
    
    for NAT in NATs:    
        
        # Calculate Moran's I 
        moran[NAT] = Moran(features[NAT], W[NAT]) 
        moran_loc[NAT] = Moran_Local(features[NAT], W[NAT])
        
        # Plot bootstrap permutations
        if plotter == True:
            fig, ax = plt.subplots()
            plt.suptitle('Moran\'s I')
            
            ax.hist(moran[NAT].sim, 25, color = 'darkgrey')
            ax.axvline(moran[NAT].I, color = 'crimson')
            ax.axvline(np.percentile(moran[NAT].sim,95), color = 'grey')
            ax.set_title('Random: I: '+str(round(moran[NAT].I,2))+'; p: '+str(round(moran[NAT].p_sim,2)))
            
            plt.tight_layout()
            
            plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg')
        
        print('Morans I: ' + str(round(moran[NAT].I,3)) + ' p-value: ' + str(round(moran[NAT].p_sim,3)))
        
        # Plot heatmap with moran histogram at the top
        if plotter == True:
           
            # vMin, vMax = -0.1, 0.1
            
            ext = abs(features[NAT].min()), features[NAT].max()
            
            vMin, vMax = -np.max(ext), np.max(ext)
            
            norm = plt.Normalize(vMin, vMax)
            sm = plt.cm.ScalarMappable(cmap="vlag", norm=norm)
            sm.set_array([])
            
            fig = plt.figure(figsize=(15,10))
            gs = fig.add_gridspec(2,2, height_ratios=(1,4), hspace = 0.15)
            
            ax = fig.add_subplot(gs[1, 0])
            ax_histy = fig.add_subplot(gs[0, 0])
            
            sns.heatmap(maxproj, ax = ax, cmap = 'gist_gray', square = True, 
                            cbar = False, yticklabels = False, xticklabels = False, alpha = 0.75, rasterized = True)
            
            ax.scatter(ROIs[NAT][:,0], ROIs[NAT][:,1], s = 50, c = features[NAT], cmap = 'vlag', vmin = vMin, vmax = vMax)
            ax.figure.colorbar(sm, ax = ax, shrink = 0.75, label = 'Correlation (r)')
            ax.set_aspect('equal')
                
            sns.histplot(data = moran[NAT].sim, ax = ax_histy, color = 'darkgrey', edgecolor = 'gray')
            ax_histy.axvline(moran[NAT].I, color = contrast[200], label = 'Moran\'s I')
            ax_histy.axvline(np.percentile(moran[NAT].sim,95), color = 'grey', label = '95th percentile')
            ax_histy.legend(frameon = False)
            ax_histy.set_xlabel('Moran\'s I')
            ax_histy.spines[['top', 'right']].set_visible(False)
            
            plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg')
        
    return moran, moran_loc

#%% Define constants
sessionName = ['A','B','A\'']

scale = 1.1801 # Pixels/µm --> 1 pixel = 0.847.. µm 
binning = 2.5
minPeak = 0.2

# minDist = 25 # Radius minimum minimum distance (um) from a cell centre to neighbours' (centres) to calculate "nearby tuning"

palette = list(sns.color_palette("viridis", 256).as_hex())
contrast = list(sns.color_palette('OrRd', 256).as_hex())

#%% Loop over the sessions

results_folder = r'C:\Users\torstsl\Projects\axon2pmini\results'

with open(results_folder+'/sessions_overview.txt') as f:
    sessions = f.read().splitlines() 
f.close()

moran_ROI_nearest_dist = {}

    
for session in sessions: 
    
    # Load session_dict
    session_dict = pickle.load(open(session+'\session_dict.pickle','rb'))
    print('Successfully loaded session_dict from: '+str(session))
    
    key = session_dict['Animal_ID'] + '-' + session_dict['Date']
    
    # Calculate the projection of the image
    maxproj, meanimg, cell_centre, masks_max, masks_mean, masks_max_zero = projection_masks(session_dict, plotter = False)
    
    # Calculate distance matix
    distanceMtx =  np.linalg.norm(cell_centre - cell_centre[:,None], axis = -1)
    distanceMtx[np.diag_indices_from(distanceMtx)] = np.nan # Put distance to self as NaN

    # Calculate the correlation to the nearest neighbour PC 
    near_dist = calc_fielddist(session_dict, distanceMtx)

    # Need to get the features and weights correct to calculate Moran's statistics
    features, weights, ROIs = prepare_moran(near_dist, distanceMtx, cell_centre)
 
    # Perform Moran's on the prepared data
    moran, moran_loc = calc_morans(features, weights, maxproj, ROIs, plotter = False)    
             
    # Put the results in a dict
    moran_ROI_nearest_dist[key] = {'maxproj': maxproj,
                               'cell_centre': cell_centre,
                               'masks_max': masks_max,
                               'masks_max_zero': masks_max_zero,
                               'moran': moran,
                               'moran_loc': moran_loc, 
                               'ROIs': ROIs, 
                               'features': features,
                               'type': 'Place field distance'}
    
    # Store the output
    with open(results_folder+'/moran_ROI_nearest_dist.pickle','wb') as handle:
        pickle.dump(moran_ROI_nearest_dist, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Successfully saved results_dict in '+ results_folder)

#%% Load results if already analysed

moran_ROI_nearest_dist = pickle.load(open(r'C:\Users\torstsl\Projects\axon2pmini\results\moran_ROI_nearest_dist.pickle','rb'))

#%%
results = np.full([24,5], np.nan)
NATs = ['NAT0', 'NAT1']

counter = 0
anySign = False

for key in moran_ROI_nearest_dist.keys():
    for NAT in NATs:
        
        results[counter,0] = moran_ROI_nearest_dist[key]['moran'][NAT].I
        results[counter,1] = moran_ROI_nearest_dist[key]['moran'][NAT].EI
        results[counter,2] = np.percentile(moran_ROI_nearest_dist[key]['moran'][NAT].sim, 95)
        results[counter,3] = moran_ROI_nearest_dist[key]['moran'][NAT].p_rand
        results[counter,4] = moran_ROI_nearest_dist[key]['moran'][NAT].p_sim
        
        if results[counter,0] > results[counter,2]: 
            print(key, NAT)
            anySign = True
        
        counter += 1
        
if anySign == False: print('No significant Moran I calculated')

#%% Plot Moran's
for key in moran_ROI_nearest_dist.keys():
    
    features = moran_ROI_nearest_dist[key]['features']
    maxproj = moran_ROI_nearest_dist[key]['maxproj']
    ROIs = moran_ROI_nearest_dist[key]['ROIs']
    moran = moran_ROI_nearest_dist[key]['moran'] 
    
    for NAT in NATs:
       
       vMin, vMax = 0, features[NAT].max()
       
       norm = plt.Normalize(vMin, vMax)
       sm = plt.cm.ScalarMappable(cmap="OrRd_r", norm=norm)
       sm.set_array([])
       
       fig = plt.figure(figsize=(15,10))
       gs = fig.add_gridspec(2,2, height_ratios=(1,4), hspace = 0.15)
       
       plt.suptitle(key+str(': ')+NAT, x = '0.3', y = '0.925')
       
       ax = fig.add_subplot(gs[1, 0])
       ax_histy = fig.add_subplot(gs[0, 0])
       
       sns.heatmap(maxproj, ax = ax, cmap = 'gist_gray', square = True, 
                       cbar = False, yticklabels = False, xticklabels = False, alpha = 0.75, rasterized = True)
       
       ax.scatter(ROIs[NAT][:,0], ROIs[NAT][:,1], s = 50, c = features[NAT], cmap = 'OrRd_r', vmin = vMin, vmax = vMax)
       ax.figure.colorbar(sm, ax = ax, shrink = 0.75, label = 'Field distance (cm)')
       ax.set_aspect('equal')
           
       sns.histplot(data = moran[NAT].sim, ax = ax_histy, color = 'darkgrey', edgecolor = 'gray')
       ax_histy.axvline(moran[NAT].I, color = contrast[200], label = 'Moran\'s I')
       ax_histy.axvline(np.percentile(moran[NAT].sim,95), color = 'grey', label = '95th percentile')
       ax_histy.legend(frameon = False)
       ax_histy.set_xlabel('Moran\'s I')
       ax_histy.spines[['top', 'right']].set_visible(False)
       
       plt.savefig('N:/axon2pmini/Article - Topography/Figures/Moran_ROI_nearest_dist/'+key+'_'+NAT+'.svg', format = 'svg') 

#%% Local Moran's I for significant session(s)

from statsmodels.stats.multitest import fdrcorrection

theKey, theNAT = '102124-201222', 'NAT1'

moranLoc = moran_ROI_nearest_dist[theKey]['moran_loc'][theNAT]

moranLoc.Is.max()

(moranLoc.p_sim)<0.05
(moranLoc.p_z_sim*2)<0.05

# Prepare values for plotting 
coords = moran_ROI_nearest_dist[theKey]['ROIs'][theNAT]
feature = moran_ROI_nearest_dist[theKey]['features'][theNAT]
maxproj = moran_ROI_nearest_dist[theKey]['maxproj']


#%% Plot the figure of bin correlations and local Moran's Is
fig, ax = plt.subplots(2,2, figsize = (8,5.4))
for axes in ax.flatten():
    sns.heatmap(maxproj, ax = axes, cmap = 'gist_gray', square = True, 
                    cbar = False, yticklabels = False, xticklabels = False, alpha = 0.65, rasterized = True)
    axes.set_aspect('equal')
  
# Scatter bin correlations
vEx = abs(feature).max()    
norm = plt.Normalize(0, vEx)
sm = plt.cm.ScalarMappable(cmap="OrRd_r", norm=norm)    
    
scatterSize = 10

ax[0,0].scatter(coords[:,0], coords[:,1], s = scatterSize, c = feature, cmap = 'OrRd_r', vmin = 0, vmax = vEx)
ax[0,0].figure.colorbar(sm, ax = ax[0,0], shrink = 0.95, label = 'Place field distance (cm)')
ax[0,0].set_title('Place field distance (cm)')

# Scatter local Moran's I
vEx = abs(moranLoc.Is).max()    
norm = plt.Normalize(-vEx, vEx)
sm = plt.cm.ScalarMappable(cmap="vlag", norm=norm)    
    
ax[0,1].scatter(coords[:,0], coords[:,1], s = scatterSize, c = moranLoc.Is, cmap = 'vlag', vmin = -vEx, vmax = vEx)
ax[0,1].figure.colorbar(sm, ax = ax[0,1], shrink = 0.95, label = 'Local Moran\'''s I')
ax[0,1].set_title('Local Moran\'''s I')

# Scatter p-values
localM_sig, pAdjust = fdrcorrection(moranLoc.p_sim, 0.05, 'indep', False)

ax[1,0].scatter(coords[:,0][localM_sig], coords[:,1][localM_sig], s = scatterSize, c = contrast[220], label = 'p < 0.05')
ax[1,0].scatter(coords[:,0][~localM_sig], coords[:,1][~localM_sig], s = scatterSize, c = 'darkgrey', label = 'p ≥ 0.05')
ax[1,0].legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
ax[1,0].set_title('p-value')

# Scatter of significant values
ax[1,1].scatter(coords[:,0][localM_sig], coords[:,1][localM_sig], s = scatterSize, c = moranLoc.Is[localM_sig], cmap = 'vlag', vmin = -vEx, vmax = vEx)
ax[1,1].set_title('Significant local Moran\'''s I')

plt.tight_layout()

plt.savefig(r'N:\axon2pmini\Article - Topography\Figures\NN/localM.svg', format = 'svg')   