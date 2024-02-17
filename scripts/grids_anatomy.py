# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 09:33:41 2022

@author: torstsl

GRID'S ANATOMY

This chunky boy does a bit. It is rather nested and complex. 
In short, this script does the following:
    - Loads the 2P image from session_dict (max projection and mean)
    - Plots the 2P image. Allows for plotting ROIs of single/groups/all cells
      on top.
    - Loads the activity data for the cells (deconvolved and dF/F)
    - Separates the max. intensitiy image in a grid. The size of the grid is
      determined by the manual input nRows
    - Place cells are grouped by which bin in the grid they belong to
    - The activity traces (deconvolved, dF/F and binned dF/F) per session of 
      place cells in the same bin are correlated, and the means of the 
      correlations are stored in the dicts group_decon/df/bin 
    - The activity traces (deconvolved, dF/F and binned dF/F) of place cells 
      are shuffled. Number of shuffles is set in the manual inpout nShuffles. 
      Group sizes are kept constant. The shuffled group acitivty is correlated 
      and the mean and 95th percentile of the shuffled distribution is 
      calculated. This is stored in the dict groups_shuffle
    - The correlations in the true anatomical cluster can thus be compared to
      the shuffled distributions
    - The bins/groups are then used to compare the corrleations between the 
      place fields of plkace cells within one bin. The Pearson Rs are store in 
      the dict bin_field_corr
     
"""

import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from src.cell_activity_analysis import bin_signal_trace, shuffle_bin_signal, calc_corr_mtx_mean, ttest_paired
import libpysal.weights
from esda.moran import Moran, Moran_Local

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

#%% Make the grid, put each cell into its coordinate in the grid and get the bin number per cell     

def calc_grids(maxproj, scale, masks_max, **kwargs):

    plotter = kwargs.get('plotter', False)
    
    placecell_dict = session_dict['Placecell']
    placecells = np.unique(np.concatenate([placecell_dict['NAT0'][0],placecell_dict['NAT1'][0],placecell_dict['NAT2'][0]]))
    
    nRows = 8 # Size of one bin is [maxproj.shape[0]/nRows, maxproj.shape[1]/nRows]
    
    x_edges = np.linspace(0,maxproj.shape[1],nRows+1) # Number of edges = number of bins +1   
    y_edges = np.linspace(0,maxproj.shape[0],nRows+1) # Number of edges = number of bins +1   
    
    binsize = np.diff(x_edges)[0]/scale, np.diff(y_edges)[0]/scale
    print('Bin size in grid: '+str(round(binsize[0],1))+' µm x ' +str(round(binsize[1],1))+' µm')
    
    if plotter ==  True: 
        # 1: Maxproj with grid
        plt.subplots()
        plt.suptitle('Max proj. with PCs')
        sns.heatmap(maxproj, cmap='gist_gray', square = True, xticklabels = False, yticklabels = False, cbar = False)
        for ii in range(1,len(x_edges)): 
            plt.vlines(x_edges[ii], 0, len(maxproj), color = 'white', linewidth = 0.5)
            plt.hlines(y_edges[ii], 0, len(maxproj), color = 'white', linewidth = 0.5)
        plt.xlim([0, maxproj.shape[1]])
        plt.ylim([maxproj.shape[0], 0])
        plt.axis('off')
        plt.axis('equal')
        plt.tight_layout()
        
        plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg')  
        
        # 2: Maxproj with mask outline from all place cells
        fig, ax = plt.subplots()
        sns.heatmap(maxproj, cmap='gist_gray', square = True, xticklabels = False, yticklabels = False, cbar = False)
        fig.suptitle('Max proj. with all PCs')
        for ii in placecells-1:
            ax.contour(masks_max[ii], colors = 'w', linewidths = 0.15, alpha = 0.75)
        for ii in range(1,len(x_edges)): 
            ax.vlines(x_edges[ii], 0, len(maxproj), color = 'white', linewidth = 0.5)
            ax.hlines(y_edges[ii], 0, len(maxproj), color = 'white', linewidth = 0.5)
        plt.xlim([0, maxproj.shape[1]])
        plt.ylim([maxproj.shape[0], 0])
        ax.axis('off')
        fig.tight_layout()

    return x_edges, y_edges, binsize

#%% Group the place cells by their bin in the grid

def group_by_grid(session_dict, cell_centre, x_edges, y_edges, **kwargs):

    plotter = kwargs.get('plotter', False)
    
    nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)
    placecell_dict = session_dict['Placecell']
    placecells = np.unique(np.concatenate([placecell_dict['NAT0'][0],placecell_dict['NAT1'][0],placecell_dict['NAT2'][0]]))
    
    nRows = 8 # Size of one bin is [maxproj.shape[0]/nRows, maxproj.shape[1]/nRows]
    
    # Cell bin number is in datapoint_idx, and the cell ID is in placecell_dict
    datapoint_coor = np.full([nCells,2], np.nan) 
    datapoint_idx = np.full([nCells,1], np.nan) 
    
    for pcNo in placecells-1:
        x_coor = cell_centre[pcNo][0]
        y_coor = cell_centre[pcNo][1]
        
        x_ind = np.where(x_coor <= x_edges)[0][0]-1
        y_ind = np.where(y_coor <= y_edges)[0][0]-1
        
        if x_ind == -1: x_ind = 0
        if y_ind == -1: y_ind = 0
            
        datapoint_coor[pcNo] = [x_ind,y_ind] # Coordinate in the grid
        datapoint_idx[pcNo] = x_ind+y_ind + ((nRows-1)*y_ind) # Bin number in the grid 
        
        if datapoint_idx[pcNo] >= nRows**2: print('Idx out of reach for cell: ' + str(pcNo))
    
    if plotter == True:
        # Alternative way of doing the same: They do not give the same output
        from scipy import stats
        x = cell_centre[:,0]
        y = cell_centre[:,1]
        
        ret = stats.binned_statistic_2d(x, y, None, 'count', bins=[x_edges, y_edges], expand_binnumbers = True)
        
        cellBin = np.full([nCells, 3], np.nan)
        cellBin[:,0], cellBin[:,1] = ret.binnumber[0]-1, ret.binnumber[1]-1
        cellBin[:,2] = (cellBin[:,1])*10+cellBin[:,0]
        
        # Plot a figure to check that the two ways are equivalent
        fig, ax = plt.subplots(1,2)
        
        bin2check = 90
        pcs = placecell_dict['NAT0'][0]
        ax[0].imshow(np.flipud(maxproj), cmap = 'gist_gray')
        for x in pcs[groups['NAT0'][str(bin2check)]]-1:
            ax[0].contour(masks_max[x], colors = '#E7533A', linewidths = 0.2, alpha = 0.75)
        for ii in range(1,len(x_edges)): 
            ax[0].vlines(x_edges[ii], 0, y_edges[-1]+y_edges[1]/2, color = 'darkgray', linewidth = 0.5)
            ax[0].hlines(y_edges[ii], 0, x_edges[-1]+x_edges[1]/2, color = 'darkgray', linewidth = 0.5)      
            ax[0].set_axis_off() 
            ax[0].set_xlim([0,maxproj.shape[0]])
            ax[0].set_ylim([0, maxproj.shape[1]])
            
        cellIdx = np.where([cellBin[:,2] == bin2check])[1] # Python index of cells
        ax[1].imshow(np.flipud(maxproj), cmap = 'gist_gray')
        for x in cellIdx[np.isin(cellIdx+1, pcs)]: # The python index where it is place cells (NATEX index)
            ax[1].contour(masks_max[x], colors = '#E7533A', linewidths = 0.2, alpha = 0.75)
        for ii in range(1,len(x_edges)): 
            ax[1].vlines(x_edges[ii], 0, y_edges[-1]+y_edges[1]/2, color = 'darkgray', linewidth = 0.5)
            ax[1].hlines(y_edges[ii], 0, x_edges[-1]+x_edges[1]/2, color = 'darkgray', linewidth = 0.5)     
            ax[1].set_axis_off() 
            ax[1].set_xlim([0, maxproj.shape[0]])
            ax[1].set_ylim([0, maxproj.shape[1]])
        plt.tight_layout()
        
        # Plot a figure to show that this method traverses in x,y correctly, and that the masks make sense
        fig, ax = plt.subplots(2,5, figsize = (24,10))
        for i, ax in enumerate(ax.flat):
            cellIdx = np.where([cellBin[:,2] == i])[1]
            ax.imshow(np.flipud(maxproj), cmap = 'gist_gray')
            for x in cellIdx:
                ax.contour(masks_max[x], colors = '#E7533A', linewidths = 0.2, alpha = 0.75)
            for ii in range(1,len(x_edges)): 
                ax.vlines(x_edges[ii], 0, y_edges[-1]+y_edges[1]/2, color = 'darkgray', linewidth = 0.5)
                ax.hlines(y_edges[ii], 0, x_edges[-1]+x_edges[1]/2, color = 'darkgray', linewidth = 0.5) 
            ax.set_xlim([0, maxproj.shape[0]])
            ax.set_ylim([0, maxproj.shape[1]])
            ax.set_axis_off() 
        plt.tight_layout()

    return datapoint_coor, datapoint_idx

#%% Get cell event traces that are to be correlated (both convolved and raw dF/F)

def get_traces(session_dict, **kwargs):
    
    plotter = kwargs.get('plotter', False)
   
    nSessions = len(session_dict['dfNAT'])
    placecell_dict = session_dict['Placecell']
    
    signal_decon = {}
    signal_df = {}
    bin_signal = {}
    
    decon_shift = 12+3 # Shift in NAT to get deconvolved spikes for cell n (column shift+n, n = [1,nCells])
    df_shift = 12 # Shift in NAT to get raw dF/F for cell n (column shift+n, n = [1,nCells])
        
    for sessionNo in range(nSessions):
            
        dfNAT = session_dict['dfNAT']['dfNAT'+str(sessionNo)]
        pcs = placecell_dict['NAT'+str(sessionNo)][0]
        
        signal_decon['NAT'+str(sessionNo)] = dfNAT.iloc[:,decon_shift + np.dot(pcs-1,4)] # Deconvolved
        signal_df['NAT'+str(sessionNo)] = dfNAT.iloc[:,df_shift + np.dot(pcs-1,4)] # dF/F, from Fcorr (raw)
        bin_signal['NAT'+str(sessionNo)]= bin_signal_trace(signal_df['NAT'+str(sessionNo)], 10, smoothing = False) # Binned signal

    # Plot and check the signals from above
    if plotter == True:
        
        # To plot 10 place cell traces
        pcs = placecell_dict['NAT0'][0]
        
        fig, ax = plt.subplots(10,1, sharey=True)
        plt.suptitle('Traces for 10 place cells, dF/F')
        for ii in range(10):
            ax[ii].plot(signal_df['NAT0'].iloc[:,ii], color='forestgreen')
            ax[ii].set_yticklabels([])
            ax[ii].set_yticks([])
            ax[ii].set_ylabel(str(pcs[ii]),rotation=0, ha = 'right', va = 'center')
            ax[9].set_xlabel('Time (2 s bins)')
        
        fig, ax = plt.subplots(10,1, sharey=True)
        plt.suptitle('Traces for 10 place cells, bin dF/F')
        for ii in range(10):
            ax[ii].plot(bin_signal['NAT0'][:,ii], color='forestgreen')
            ax[ii].set_yticklabels([])
            ax[ii].set_yticks([])
            ax[ii].set_ylabel(str(pcs[ii]),rotation=0, ha = 'right', va = 'center')
            ax[9].set_xlabel('Time (2 s bins)')
        
        fig, ax = plt.subplots(10,1, sharey=True)
        plt.suptitle('Traces for 10 place cells, dF/F')
        for ii in range(10):
            jj = pcs[ii]
            ax[ii].plot(session_dict['dfNAT']['dfNAT0']['dF raw, N'+str(jj)][0:1000], color='forestgreen')
            ax[ii].set_yticklabels([])
            ax[ii].set_yticks([])
            ax[ii].set_ylabel(str(pcs[ii]),rotation=0, ha = 'right', va = 'center')
            ax[9].set_xlabel('Time (2 s bins)')

    return signal_decon, signal_df, bin_signal    

#%% Do correlation of place cell activity   
def corr_PCs(signal_decon, signal_df, bin_signal, **kwargs):
    
    plotter = kwargs.get('plotter', False)
    
    nSessions = len(bin_signal)
    
    # Correlate all the activity from all place cells per session, plot the correlation matrices
    corrcoef_decon = [[]]*3
    corrcoef_df = [[]]*3
    corrcoef_bin = [[]]*3
    
    for sessionNo in range(nSessions):
    
        # These are similar to df.corr(), but here I use the same method on all rather than making bin_signal into a DataFrame
        corrcoef_decon[sessionNo] = np.corrcoef(signal_decon['NAT'+str(sessionNo)], rowvar = False)
        corrcoef_df[sessionNo] = np.corrcoef(signal_df['NAT'+str(sessionNo)], rowvar = False)
        corrcoef_bin[sessionNo] = np.corrcoef(bin_signal['NAT'+str(sessionNo)], rowvar = False)
        
        if plotter == True:
            fig, ax = plt.subplots(1,3, sharex = True, sharey = True, figsize=(13,4))
            exp_string = ['A', 'B', 'A\'']
            fig.suptitle('Correlation matrix of place cells, session '+exp_string[sessionNo])
            ax[0].imshow(corrcoef_decon[sessionNo], cmap='mako', vmin = 0, vmax = 0.3)
            ax[0].set_title('Deconvolved')
            ax[1].imshow(corrcoef_df[sessionNo], cmap='mako', vmin = 0, vmax = 0.3)
            ax[1].set_title('dF/F')
            ax[2].imshow(corrcoef_bin[sessionNo], cmap='mako', vmin = 0, vmax = 0.3)
            ax[2].set_title('Binned dF/F')
            im = ax[2].imshow(corrcoef_bin[sessionNo], cmap='mako', vmin = 0, vmax = 0.3)
            plt.colorbar(im)
            plt.tight_layout()

    return corrcoef_decon, corrcoef_df, corrcoef_bin  

#%% Separate all cells into groups or bins within the anatomical 2D grid
def group_by_bin(session_dict, datapoint_idx, signal_decon, signal_df, bin_signal, maxproj, masks_max, x_edges, y_edges, **kwargs):
    
    plotter = kwargs.get('plotter', False)
    nRows = 8
    session_string = ['A', 'B', 'A\'']
    
    nSessions = len(session_dict['dfNAT'])
    placecell_dict = session_dict['Placecell']
    groups = {} # PCs within each group (group is the 2D histbin the PCs belong to)
    
    data_decon = {}
    data_df = {}
    data_bin = {}
    
    group_decon = {}
    group_df = {}
    group_bin = {}
    
    # Bin numbers in the grid for all placecells (all sessions) with NaN removed
    binPC = datapoint_idx[np.invert(np.isnan(datapoint_idx))]
    
    for sessionNo in range(nSessions):
        key = 'NAT'+str(sessionNo)
        
        data_decon[key] = {}
        data_df[key] = {}
        data_bin[key] = {}   
        groups[key] = {}
        
        group_decon[key] = {}
        group_df[key] = {}
        group_bin[key] = {}
        
        for binCell in np.unique(binPC).astype(int): 
            
            # Define the group (from the bins)
            pcs = placecell_dict[key][0]
            
            # Get idx where place cells for this session (pcs-1) belong in this bin 
            group_idx = np.where(datapoint_idx[pcs-1] == binCell)[0] # Index of PCs for this session that are within this bin
           
            # binCell is for all PCs in all sessions. Hence, only proceed if there are PCs in this bin for this session
            if len(group_idx) > 1:
                groups[key][str(binCell)] = group_idx # Note that these are just indexes
           
                # TEST: Group idx of pcs should give the same cells (NATEX ID) as are stated in the DataFrame by slicing as below
                # pcs[group_idx] # This is the NATEX cell number
                # signal_decon['NAT0'].iloc[:,groups['NAT'+str(sessionNo)][str(binCell)]]
            
                # Find the traces for the cells within this group (temp. variables)
                data_decon[key][str(binCell)] = signal_decon[key].iloc[:,groups[key][str(binCell)]] 
                data_df[key][str(binCell)] = signal_df[key].iloc[:,groups[key][str(binCell)]] 
                data_bin[key][str(binCell)] = bin_signal[key][:,groups[key][str(binCell)]]
                
                # Transpose to compare across different cells, not timestamps
                group_decon[key][str(binCell)] = calc_corr_mtx_mean(data_decon[key][str(binCell)].T)
                group_df[key][str(binCell)] = calc_corr_mtx_mean(data_df[key][str(binCell)].T)
                group_bin[key][str(binCell)] = calc_corr_mtx_mean(data_bin[key][str(binCell)].T)
    
            # Correlate the traces within this group - the value in the dicts are the mean correlation value (mean R)
            # if len(group_idx) == 1:
            #     group_decon[key][str(binCell)] = np.nan
            #     group_df[key][str(binCell)] = np.nan
            #     group_bin[key][str(binCell)] = np.nan
    
    # Plot the grid with number in grid --> Can then add each single ROI (sanity check)
    if plotter == True:
        fig, ax = plt.subplots()
        ax.imshow(maxproj, cmap='gist_gray')
        fig.suptitle('Bin groups')
        s = 0
        for ii in range(1,len(x_edges)): 
            ax.vlines(x_edges[ii], 0, maxproj.shape[0], color = 'white', linewidth = 0.5)
            ax.hlines(y_edges[ii], 0, maxproj.shape[1], color = 'white', linewidth = 0.5)
        
        for x in x_edges[1:nRows+1]:
            for y in y_edges[0:nRows]:
                ax.text(y,x-5,str(s), color = 'w')
                s += 1
       
        plt.xlim([0, maxproj.shape[1]])
        plt.ylim([maxproj.shape[0], 0])
        ax.axis('off')
        fig.tight_layout()
    
        # Check cell(s) from masks (which is nCells long), hence pcs-1 cooresponds to PC index in masks_max
        bin2check = str(90)
        pcs = placecell_dict['NAT0'][0]
        
        fig, ax = plt.subplots()
        if bin2check in groups['NAT0']:
            for x in pcs[groups['NAT0'][bin2check]]-1:
                ax.contour(masks_max[x], colors = '#E7533A', linewidths = 0.2, alpha = 0.75)
            for ii in range(1,len(x_edges)): 
                ax.vlines(x_edges[ii], 0, y_edges[-1]+y_edges[1]/2, color = 'darkgray', linewidth = 0.5)
                ax.hlines(y_edges[ii], 0, x_edges[-1]+x_edges[1]/2, color = 'darkgray', linewidth = 0.5)      
            ax.imshow(maxproj, cmap = 'gist_gray')
    
        # Plot the distribution of cell numbers per bin
        fig, ax = plt.subplots(1, nSessions, sharey=True, sharex = True)
        plt.suptitle('Number of cells per bin')
        fig.supxlabel('Cells per grid bin')
        fig.supylabel('Count')
        
        for sessionNo in range(nSessions):
            lengths = [len(v) for v in groups['NAT'+str(sessionNo)].values()]
            ax[sessionNo].hist(lengths, 15, color = '#33638DFF')
            ax[sessionNo].set_title(session_string[sessionNo], rotation = 0)
            
        plt.tight_layout()
        
    return  groups, group_decon, group_df, group_bin

#%% Shuffle the binned trace of dF/F
def shuffle_signal(bin_signal, groups, group_bin):
    
    nShuffles = 200
    nSessions = len(bin_signal)
    groups_shuffle, shuffle_mean = {}, {}
    
    for sessionNo in range(nSessions):
        key = 'NAT'+str(sessionNo)
        groups_shuffle[key], shuffle_mean[key] = shuffle_bin_signal(bin_signal[key].T, groups[key], nShuffles)
    
        # See if the true data are above the suffled distribution
    
        # bin_signal is the binned dF/F, and group_bin are the true values from the data
        for binNo in groups[key].keys():
            if group_bin[key][binNo] > groups_shuffle[key][binNo][1]:
                print('Quite significant '+key+' bin '+binNo)

    return groups_shuffle, shuffle_mean

#%% Correlate the ratemaps within each group and shuffle the data
def shuffle_ratemap(session_dict, shuffle_bin_signal, groups, **kwargs): 
   
    plotter = kwargs.get('plotter', False)
       
    nSessions = len(session_dict['dfNAT'])
    placecell_dict = session_dict['Placecell'] 
    nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)
    session_string = ['A', 'B', 'A\'']
    
    nShufflesMap = 200
    bin_field_corr = {}
    groups_shuffleMap, shuffle_meanMap = {}, {}
    
    for sessionNo in range(nSessions):
        key = 'NAT'+str(sessionNo)
        
        bin_field_corr[key] = {}
        pcs = placecell_dict[key][0]
        
        # Make a 3D array with ratemaps for all cells, cells along z axis
        values = session_dict['Ratemaps']['df'+key].values()
        value_list = list(values)
        zmaps = np.ma.array(value_list)
        
        # From the 3D arrary, flatten all signals and put into an masked 2D array (rows are cells)
        flatmap = np.ma.zeros((nCells, len(zmaps[0])**2))
        
        for cellNo in range(nCells): flatmap[cellNo] = zmaps[cellNo].flatten()
        
        groups_shuffleMap[key], shuffle_meanMap[key] = shuffle_bin_signal(flatmap, groups[key], nShufflesMap)
       
        # Plot the correlation matrix for all place cells' place fields
        if plotter == True:
            plt.figure()
            plt.imshow(np.corrcoef(flatmap[pcs-1]), cmap = 'mako')
            plt.title(session_string[sessionNo]+': Correlation of all place cells\' rate maps')
            plt.colorbar()
            plt.tight_layout()
        
        for binNo in groups[key].keys():
                   
            binCells = pcs[groups[key][binNo]]-1 # Array of python idx of place cells in this binNo/group (per session)
            
            if len(binCells) == 0:
                bin_field_corr[key][str(binNo)] = np.nan
            elif len(binCells) > 1:
                bin_field_corr[key][str(binNo)] = calc_corr_mtx_mean(flatmap[binCells])
                  
                # Visualizing results from a spesific bin
                if plotter == True: 
                    if binNo == '101' and binNo in bin_field_corr[key].keys():
                        # Correlate the place fields for cells within this bin
                        plt.figure()
                        plt.imshow(np.corrcoef(flatmap[binCells]), cmap = 'mako')
                        plt.title('Place field correlations, bin ' + str(binNo) + ', R = ' + str(round(bin_field_corr[key][str(binNo)],2)))
                        plt.colorbar()
                        plt.tight_layout()
                        
                        # pcs[groups['NAT0']['77']] Going from the idx to the actual cells
                        fig, ax = plt.subplots(2,3)
                        ax = ax.flatten()
                        for ii in range(2*3):
                            if ii >= len(pcs[groups['NAT2'][str(binNo)]]):
                                ax[ii].axis('off') 
                            else: 
                                ax[ii].imshow(zmaps[pcs[groups['NAT0'][str(binNo)]][ii]])
                                ax[ii].axis('off')       
                        plt.tight_layout()
                
                # Print a message if the correlation within the bin is higher than expected (arbitrary set to R = 0.3)
                if  bin_field_corr[key][str(binNo)] > 0.4: 
                    print('Session '+session_string[sessionNo] + ': Bin '+str(binNo) + '\nPearson R > 0.4')
                
                # This does not make sense, as I here compare the trace correlation to the place field correlation    
                #if  bin_field_corr[key][str(binNo)] > groups_shuffle[key][str(binNo)][1]: 
                #    print('Session '+session_string[sessionNo] + ': Bin '+str(binNo) + 
                #          '\nPearson R: '+ str(round(bin_field_corr[key][str(binNo)],3)) +' (> 99th prc)')
                
    # Make an array of the place field correlation values 
    bin_field_corr_mtx = {}
    nRows = 8
    
    for sessionNo in range(nSessions):
        key = 'NAT'+str(sessionNo)
        data_vec = np.full(nRows**2, np.nan)
        
        for binNo in bin_field_corr[key].keys():
            data_vec[int(binNo)] = bin_field_corr[key][binNo]
        
        data_mtx = np.ma.array(np.reshape(data_vec,(-nRows, nRows)), mask = np.isnan(data_vec))    
        bin_field_corr_mtx[key] = data_mtx            
                
    return bin_field_corr, bin_field_corr_mtx, groups_shuffleMap, shuffle_meanMap
                
#%% Plots of results
def plot_results(session_dict, bin_field_corr_mtx, groups_shuffleMap, group_bin, groups_shuffle, shuffle_mean):
        
    nSessions = len(session_dict['dfNAT'])
    nRows = 8
    session_string = ['A', 'B', 'A\'']
    
    true_data_field = {}
    dfData = pd.DataFrame()
    for sessionNo in range(nSessions):
        key = 'NAT'+str(sessionNo)
        true_data_field[key] = bin_field_corr_mtx[key].data.flatten()
    
        dfTemp = pd.DataFrame({'Correlation': true_data_field[key], 'Group': np.repeat(session_string[sessionNo], len(true_data_field[key]))})
        dfData = pd.concat([dfData,dfTemp])
    
    dfData.reset_index(inplace = True, drop = True)
    
    fig, ax = plt.subplots()
    sns.kdeplot(data = dfData, ax = ax, x = 'Correlation', hue = 'Group', fill = True, alpha = 0.5, palette = 'viridis')
    ax.set_title('Correlation of rate maps for all bins per session')
    # ax.set_xlim([-1,1])
    ax.spines[['top', 'right']].set_visible(False)
    
    # Scatter plot bin tuning map correlation against 99th percentile
    controlField = np.full([nSessions, nRows*nRows], np.nan)
    
    for key in groups_shuffleMap.keys():
        for binNo in groups_shuffleMap[key].keys():
            controlField[int(key[-1]), int(binNo)] = groups_shuffleMap[key][binNo][1]
            
    dataField = np.array([true_data_field['NAT0'], true_data_field['NAT1'], true_data_field['NAT2']]).flatten()
    controlField = controlField.flatten()
    
    nanMask = ~np.isnan(dataField)
    
    dfField = pd.DataFrame({'X': controlField[nanMask], 'Y': dataField[nanMask]})
    
    minVal, maxVal = np.nanmin(dataField), np.nanmax(dataField)
    
    fig, ax = plt.subplots(figsize=(5,5))
    sns.scatterplot(data = dfField, ax = ax, x = 'X', y = 'Y', alpha = 0.75, color = 'black', linewidth = 0, label = ('n = '+str(dfField['X'].size)))
    ax.set_xlabel('Shuffle 99th percentile')
    ax.set_ylabel('Bin correlation')
    ax.plot([minVal-0.01, maxVal+0.01], [minVal-0.01, maxVal+0.01], color = contrast[200], linestyle = '-')
    ax.set_aspect('equal')
    ax.set_xlim([minVal-0.01, maxVal+0.01])
    ax.set_ylim([minVal-0.01, maxVal+0.01])
    ax.set_title('Correlation of tuning maps for all bins')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon = False)
    
    plt.tight_layout()
    
    plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg')
    
    # Compare the correlation of shuffled traces and true traces (binned signal)
    
    true_data_bin_signal = {}
    for key in group_bin.keys():
        true_data_bin_signal[key] = []
        for binNo in group_bin[key].keys():
            true_data_bin_signal[key].append(group_bin[key][binNo])
    
    for sessionNo in range(nSessions):
        key = 'NAT'+str(sessionNo)
        
        df1 = pd.DataFrame({'Correlation': true_data_bin_signal[key], 'Group': np.repeat('Data', len(true_data_bin_signal[key]))})
        df2 = pd.DataFrame({'Correlation': shuffle_mean[key].flatten(),'Group': np.repeat('Shuffle', len(shuffle_mean[key].flatten()))})
        
        idx = np.random.choice(len(df2), len(df1), replace = False)
        data = pd.concat([df1, df2.iloc[idx]], ignore_index = True)
        
        lbls = [['Data'], ['Shuffle'], ['Data', 'Shuffle']]
        
        fig, ax = plt.subplots(3,1, sharex = True, sharey = True)
        sns.kdeplot(data = df1, ax = ax[0], x = 'Correlation', hue = 'Group', fill = True, alpha = 0.75, palette = 'viridis')
        sns.kdeplot(data = df2, ax = ax[1], x = 'Correlation', hue = 'Group', fill = True, alpha = 0.75, palette = 'viridis')
        sns.kdeplot(data = data, ax = ax[2], x = 'Correlation', hue = 'Group', fill = True, alpha = 0.5, palette = 'viridis')
        for ii in range(3):
            ax[ii].spines[['top', 'right']].set_visible(False)
            ax[ii].legend(labels = lbls[ii], frameon = False)
        plt.tight_layout()
        
        fig, ax = plt.subplots()
        sns.kdeplot(data = data, ax = ax, x = 'Correlation', hue = 'Group', fill = True, alpha = 0.5, palette = 'viridis')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(labels = ['Data', 'Shuffle'], frameon = False)
        plt.tight_layout()
        
    # Check if the bins had any significant correlation of signal
    for sessionNo in range(nSessions):
         key = 'NAT'+str(sessionNo)
         
         for binNo in group_bin[key].keys():
             X = group_bin[key][binNo] # The correlation of the signals in a bin
             Y = groups_shuffle[key][binNo][1] # The shuffled value (95th/99th percentile) of the signals
             
             if X > Y: 
                 print(key+' '+binNo)
                 print(round(X,2), round(Y,2))
    
    shuf = {}
    for key in groups_shuffle.keys():
        shuf[key] = []
        for binNo in groups_shuffle[key].keys():
            shuf[key].append(groups_shuffle[key][binNo][1])
    
    
    fig, ax = plt.subplots(3,1, sharex = True, sharey = True)
    for sessionNo in range(nSessions):
        key = 'NAT'+str(sessionNo)
        
        df1 = pd.DataFrame({'Correlation': true_data_bin_signal[key], 'Group': np.repeat('Data', len(true_data_bin_signal[key]))})
        df2 = pd.DataFrame({'Correlation': shuf[key],'Group': np.repeat('Shuffle', len(shuf[key]))})
           
        data = pd.concat([df1, df2])
        data.reset_index(inplace = True, drop = True)
            
        sns.kdeplot(data = data, ax = ax[sessionNo], x = 'Correlation', hue = 'Group', fill = True, alpha = 0.5, palette = 'viridis')
        
        plt.tight_layout()
     
    data = true_data_bin_signal['NAT0'] + true_data_bin_signal['NAT1'] + true_data_bin_signal['NAT2'] 
    control = shuf['NAT0'] + shuf['NAT1'] + shuf['NAT2'] 
    
    df1 = pd.DataFrame({'Correlation': data, 'Group': np.repeat('Data', len(data))})
    df2 = pd.DataFrame({'Correlation': control, 'Group': np.repeat('Shuffle', len(control))})
    
    df = pd.concat([df1, df2])
    df.reset_index(inplace = True, drop = True)
    
    fig, ax = plt.subplots(figsize=(6,5))
    sns.kdeplot(data = df, ax = ax, x = 'Correlation', hue = 'Group', fill = True, alpha = 0.75, palette = 'viridis')
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Density')
    ax.set_title('Correlation of event trace for all bins')
    
    plt.tight_layout()
    
    data, control = np.array(data), np.array(control)
    
    tStats = ttest_paired(data[~np.isnan(data)], control[~np.isnan(data)])
    
    fig, ax = plt.subplots(figsize=(6,5))
    sns.kdeplot(data = df1, ax = ax, x = 'Correlation', hue = 'Group', fill = True, alpha = 0.75, palette = 'viridis', legend=False)
    ax.axvline(np.nanmean(control), color = contrast[200])
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Density')
    ax.set_title('Correlation of event trace for all bins')
    plt.tight_layout()
    
    # Scatter plot bin trace correlation against 99th percentile
    df = pd.DataFrame({'X': control, 'Y': data})
    
    fig, ax = plt.subplots(figsize=(5,5))
    sns.scatterplot(data = df, ax = ax, x = 'X', y = 'Y', alpha = 0.75, color = 'black', linewidth = 0, label = ('n = '+str(data.size)))
    ax.set_xlabel('Shuffle 99th percentile')
    ax.set_ylabel('Bin correlation')
    ax.plot([min(data)-0.01,max(data)+0.01], [min(data)-0.01, max(data)+0.01], color = contrast[200], linestyle = '-')
    ax.set_aspect('equal')
    ax.set_xlim([min(data)-0.01,max(data)+0.01])
    ax.set_ylim([min(data)-0.01,max(data)+0.01])
    ax.set_title('Correlation of event traces for all bins')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon = False)
    
    plt.tight_layout()
    
    plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg')

#%% Plot the masks of cells from one given bin
def plot_cellmasks(session_dict, groups, maxproj, masks_max, masks_max_zero):
    
    bin2check = str(45)
    placecell_dict = session_dict['Placecell']
    pcs = placecell_dict['NAT0'][0]
    
    fig, ax = plt.subplots()    
    sns.heatmap(maxproj, cmap = 'gist_gray', cbar = False, xticklabels = False, yticklabels = False, alpha = 1)
    for PC in pcs[groups['NAT0'][bin2check]]-1:
        ax.contour(masks_max_zero[PC], colors = 'lightgray', linewidths = 0.75, alpha = 0.75)
        ax.contourf(masks_max[PC], colors = '#E7533A', alpha = 0.5)
    ax.axis('off')  
    ax.set_aspect('equal')  
    plt.tight_layout() 
    
    plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg') 
    
    # Plot the trace of these cells
    traces = session_dict['dfNAT']['dfNAT0'].iloc[:,12::4]
    events = session_dict['dfNAT']['dfNAT0'].iloc[:,15::4]
        
    plotCells = pcs[groups['NAT0'][bin2check]] # NATEX index
    
    fig, ax = plt.subplots(len(plotCells),1, figsize = (8,5), sharex = True, sharey = False) 
    for ii in range(len(plotCells)):
        cell = plotCells[ii]
        g1 = sns.lineplot(ax = ax[ii], data=traces['dF raw, N'+str(cell)], color = 'forestgreen', alpha = 0.8, rasterized = True)  
        g2 = sns.lineplot(ax = ax[ii], data=events['Deconvolved, N'+str(cell)], color = 'crimson', alpha = 0.5, rasterized = True)  
        g1.set(ylabel = None)
        ax[ii].set_xlim([0,15000])
        ax[ii].set_frame_on(False)
        ax[ii].set_ylabel('N'+str(cell), rotation = 90)
        ax[ii].yaxis.label.set(rotation='horizontal', ha='right')
    plt.setp(ax, xticks = [], yticks = [])
    ax[len(plotCells)-1].set_xticks(np.linspace(0,15000,6, dtype = int))
    ax[len(plotCells)-1].set_xticklabels(np.linspace(0,15000/7.5,6, dtype = int))
    fig.supylabel('Amplitude (A.U.)')
    fig.supxlabel('Time (seconds)')
    plt.tight_layout()
    
    plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg')    
    
    ratemaps = session_dict['Ratemaps']['dfNAT0']
    
    fig, ax =  plt.subplots(1, plotCells.size, figsize=(10,2))
    for ii in range(len(plotCells)):
        cell = plotCells[ii]
        sns.heatmap(ratemaps['N'+str(cell)], mask = ratemaps['N'+str(cell)].mask, cmap = 'viridis', cbar = False, 
                    xticklabels = False, yticklabels = False, ax = ax[ii], square = True)
        ax[ii].set_title('N'+str(cell))
    plt.tight_layout()
    
    plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg')
            
#%% Do Moran's I on the matrix
def calc_morans(x_edges, y_edges, groups, group_bin, maxproj, **kwargs):
    
    plotter = kwargs.get('plotter', False)
    
    nRows = 8
    
    # Get the centre of the bins in the grid 
    x_point = np.linspace(y_edges[1]/2, y_edges[-1]-y_edges[1]/2,nRows)
    y_point = np.linspace(x_edges[1]/2, x_edges[-1]-x_edges[1]/2,nRows)
    
    mtx_centre = np.full([len(x_point)*len(y_point), 2], np.nan)
    
    counter = 0
    for x in x_point:
        for y in y_point:
            mtx_centre[counter] = (x, y)
            counter += 1
    
    moran, moran_loc = {}, {}
    coords, feature = {}, {}
            
    for sessionNo in range(2):    
        gs = []
        for g in groups['NAT'+str(sessionNo)].keys(): gs.append(int(g))       
        G = np.asarray(gs)
        
        maskG = np.full(nRows*nRows, False)
        maskG[G] = True
        
        centre_mtx = mtx_centre[maskG]
        
        # Calculate the weights matrix: Defined by 1/distance between points in scatter
        dist = np.linalg.norm(centre_mtx - centre_mtx[:,None], axis = -1)
        dist[np.diag_indices_from(dist)] = np.nan # To omit devide by 0 warning
        
        # Calculate the weights, set diagonal to 0 (wij, i==j --> 0).
        w = 1/dist
        w[np.diag_indices_from(w)] = 0
        
        # Because these are random point, they sometimes become the same, not doable
        if np.sum(np.isinf(w))>0:
            print('Divide by 0 in weights, infinite value occurs')
         
        # Get the feature we want to look at (variable of interest)
        feature['NAT'+str(sessionNo)] = np.asarray(list(group_bin['NAT'+str(sessionNo)].values())) # Correlation of binned signal trace
        # feature['NAT'+str(sessionNo)] = np.asarray(list(bin_field_corr[['NAT'+str(sessionNo)]].values())) # Correlation of rate maps
        
        # If the value is NaN, this bin only has one cell and thus the bin will be discarded
        if np.sum(np.isnan(feature['NAT'+str(sessionNo)]))>0: print('NaN in variable of interest, cannot calculate Morans I')
        
        nanMask = np.where(np.isnan(feature['NAT'+str(sessionNo)]))
        
        feature['NAT'+str(sessionNo)] = np.delete(feature['NAT'+str(sessionNo)], nanMask)
        w = np.delete(w, nanMask, axis = 0)
        w = np.delete(w, nanMask, axis = 1)
        
        # Plot a scatter of the points with the variable of interest
        coords['NAT'+str(sessionNo)] = np.delete(centre_mtx, nanMask, axis = 0)
        
        if plotter == True:
            fig, ax = plt.subplots(figsize=(5,5))
            #ax.set_title('Anatomical bins\' correlation')
            sns.heatmap(np.flipud(maxproj), ax = ax, cmap = 'gist_gray', square = True, cbar = False, yticklabels = False, xticklabels = False, alpha = 0.75, rasterized = True)
            ax.scatter(coords['NAT'+str(sessionNo)][:,1], coords['NAT'+str(sessionNo)][:,0], s = feature['NAT'+str(sessionNo)][:]*100, color = contrast[200])
            ax.set_aspect('equal')
            for ii in range(1,len(x_edges)): 
                # ax.vlines(x_edges[ii], 0, y_edges[-1]+y_edges[1]/2, color = 'darkgray', linewidth = 0.5)
                # ax.hlines(y_edges[ii], 0, x_edges[-1]+x_edges[1]/2, color = 'darkgray', linewidth = 0.5)
                ax.vlines(x_edges[ii], 0, y_edges[-1], color = 'darkgray', linewidth = 0.5)
                ax.hlines(y_edges[ii], 0, x_edges[-1], color = 'darkgray', linewidth = 0.5)
            ax.set_xlim(x_edges[0],x_edges[-1])    
            ax.set_ylim(y_edges[0],y_edges[-1])    
            plt.axis('off')
            
            plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg')
        
        #labels = np.array([100,200,300,400])
        #labelpos = labels*scale
        
        #ax.set_xlabel('µm')
        #ax.set_ylabel('µm')
        #ax.set_yticks(labelpos)
        #ax.set_yticks(labels)
        #ax.set_xticks(labelpos)
        #ax.set_xticks(labels)
        
        # Create the weights object needed for calculation, instances of i==j are not accepted
        keys = []
        for i in range(len(feature['NAT'+str(sessionNo)])): keys.append(i)
        keys = np.array(keys)
        
        neighbours, weights = {}, {}
        for i in range(len(feature['NAT'+str(sessionNo)])): 
            others = keys[~np.isin(keys,i)]
            neighbours[keys[i]] = others
            weights[keys[i]] = w[i, others]
        
        # Get the weights object
        W = libpysal.weights.W(neighbours, weights)
        
        # Calculate Moran's I 
        moran['NAT'+str(sessionNo)] = Moran(feature['NAT'+str(sessionNo)], W) 
        moran_loc['NAT'+str(sessionNo)] = Moran_Local(feature['NAT'+str(sessionNo)],W)
        
        # Plot bootstrap permutations
        if plotter == True:
            fig, ax = plt.subplots()
            plt.suptitle('Moran\'s I')
            
            ax.hist(moran['NAT'+str(sessionNo)].sim, 25, color = 'darkgrey')
            ax.axvline(moran['NAT'+str(sessionNo)].I, color = 'crimson')
            ax.axvline(np.percentile(moran['NAT'+str(sessionNo)].sim,95), color = 'grey')
            ax.set_title('Random: I: '+str(round(moran['NAT'+str(sessionNo)].I,2))+'; p: '+str(round(moran['NAT'+str(sessionNo)].p_sim,2)))
            
            plt.tight_layout()
            
            plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg')
        
        print('Morans I: ' + str(round(moran['NAT'+str(sessionNo)].I,3)) + ' p-value: ' + str(round(moran['NAT'+str(sessionNo)].p_sim,3)))
        
        # Plot heatmap with moran histogram at the top
        if plotter == True:
           
            # cd = pd.DataFrame({'X': coords['NAT'+str(sessionNo)][:,0], 'Y': coords['NAT'+str(sessionNo)][:,1]})
            vMin, vMax = -0.1, 0.1
            
            norm = plt.Normalize(vMin, vMax)
            sm = plt.cm.ScalarMappable(cmap="vlag", norm=norm)
            sm.set_array([])
            
            fig = plt.figure(figsize=(15,10))
            gs = fig.add_gridspec(2,2, height_ratios=(1,4), hspace = 0.15)
            
            ax = fig.add_subplot(gs[1, 0])
            ax_histy = fig.add_subplot(gs[0, 0])
            
            sns.heatmap(maxproj, ax = ax, cmap = 'gist_gray', square = True, 
                            cbar = False, yticklabels = False, xticklabels = False, alpha = 0.75, rasterized = True)
            # ax.scatter(coords['NAT'+str(sessionNo)][feature['NAT'+str(sessionNo)]>=0,0], coords['NAT'+str(sessionNo)][feature['NAT'+str(sessionNo)]>=0,1], s = (feature['NAT'+str(sessionNo)][feature['NAT'+str(sessionNo)]>=0])*750, color = contrast[190]) # Positive correlation
            # ax.scatter(coords['NAT'+str(sessionNo)][feature['NAT'+str(sessionNo)]<0,0], coords['NAT'+str(sessionNo)][feature['NAT'+str(sessionNo)]<0,1], s = abs(feature['NAT'+str(sessionNo)][feature['NAT'+str(sessionNo)]<0])*750, color = palette[110]) # Negative correlation
            # sns.scatterplot(data = cd, x = 'X', y = 'Y', ax = ax, hue = feature['NAT'+str(sessionNo)], palette = 'icefire', vmin = -1, vmax = 1,
            #                 edgecolors = None, legend = 'auto')
            
            ax.scatter(coords['NAT'+str(sessionNo)][:,1], coords['NAT'+str(sessionNo)][:,0], s = 50, c = feature['NAT'+str(sessionNo)], cmap = 'vlag', vmin = vMin, vmax = vMax)
            ax.figure.colorbar(sm, ax = ax, shrink = 0.75, label = 'Bin correlation (r)')
            ax.set_aspect('equal')
            
            for ii in range(1,len(x_edges)): 
                # ax.vlines(x_edges[ii], 0, y_edges[-1]+y_edges[1]/2, color = 'darkgray', linewidth = 1.5)
                # ax.hlines(y_edges[ii], 0, x_edges[-1]+x_edges[1]/2, color = 'darkgray', linewidth = 1.5)   
                ax.vlines(x_edges[ii], 0, y_edges[-1], color = 'darkgray', linewidth = 0.5)
                ax.hlines(y_edges[ii], 0, x_edges[-1], color = 'darkgray', linewidth = 0.5)
                
            sns.histplot(data = moran['NAT'+str(sessionNo)].sim, ax = ax_histy, color = 'darkgrey', edgecolor = 'gray')
            ax_histy.axvline(moran['NAT'+str(sessionNo)].I, color = contrast[200], label = 'Moran\'s I')
            ax_histy.axvline(np.percentile(moran['NAT'+str(sessionNo)].sim,95), color = 'grey', label = '95th percentile')
            ax_histy.legend(frameon = False)
            ax_histy.set_xlabel('Moran\'s I')
            ax_histy.spines[['top', 'right']].set_visible(False)
            
            plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg')
        
    return moran, moran_loc, coords, feature

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

grids_anatomy_dict = {}

if __name__ == "__main__":
    
    for session in sessions: 
        
        # Load session_dict
        session_dict = pickle.load(open(session+'\session_dict.pickle','rb'))
        print('Successfully loaded session_dict from: '+str(session))
        
        key = session_dict['Animal_ID'] + '-' + session_dict['Date']
        
        maxproj, meanimg, cell_centre, masks_max, masks_mean, masks_max_zero = projection_masks(session_dict, plotter = False)
            
        x_edges, y_edges, binsize = calc_grids(maxproj, scale, masks_max, plotter = False)
            
        datapoint_coor, datapoint_idx = group_by_grid(session_dict, cell_centre, x_edges, y_edges, plotter = False)
            
        signal_decon, signal_df, bin_signal = get_traces(session_dict, plotter = False)
            
        corrcoef_decon, corrcoef_df, corrcoef_bin = corr_PCs(signal_decon, signal_df, bin_signal, plotter = False)
            
        groups, group_decon, group_df, group_bin = group_by_bin(session_dict, datapoint_idx, signal_decon, signal_df, bin_signal, maxproj, masks_max, x_edges, y_edges, plotter = False)
            
        groups_shuffle, shuffle_mean = shuffle_signal(bin_signal, groups, group_bin)
            
        bin_field_corr, bin_field_corr_mtx, groups_shuffleMap, shuffle_meanMap = shuffle_ratemap(session_dict, shuffle_bin_signal, groups, plotter = False)
            
        # Plot the results
        # plot_results(session_dict, bin_field_corr_mtx, groups_shuffleMap, group_bin, groups_shuffle, shuffle_mean)
        # plot_cellmasks(session_dict, groups, maxproj, masks_max, masks_max_zero)

        moran, moran_loc, coords, feature = calc_morans(x_edges, y_edges, groups, group_bin, maxproj, plotter = False)    
        
        # Put the results in a dict
        grids_anatomy_dict[key] = {'maxproj': maxproj,
                                   'cell_centre': cell_centre,
                                   'masks_max': masks_max,
                                   'masks_max_zero': masks_max_zero,
                                   'x_edges': x_edges, 
                                   'y_edges': y_edges,
                                   'groups': groups,
                                   'group_bin': group_bin,
                                   'groups_shuffle': groups_shuffle,
                                   'groups_shuffleMap': groups_shuffleMap,
                                   'bin_field_corr_mtx': bin_field_corr_mtx,
                                   'shuffle_mean': shuffle_mean,
                                   'shuffle_meanMap': shuffle_meanMap,
                                   'moran': moran,
                                   'moran_loc': moran_loc, 
                                   'coords': coords, 
                                   'feature': feature}
        
        # Store the output
        with open(results_folder+'/grids_anatomy_dict.pickle','wb') as handle:
            pickle.dump(grids_anatomy_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Successfully saved results_dict in '+ results_folder)
        