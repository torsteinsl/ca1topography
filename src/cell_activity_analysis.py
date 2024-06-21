# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 17:01:11 2022

@author: torstsl

This file contains functions for exploring and analyzing neuronal data from 
2PMINI. 

"""
import numpy as np
import pandas as pd
import opexebo as op
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import libpysal.weights
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, kde, chi2_contingency, t, sem, ttest_rel, wilcoxon, f, mannwhitneyu, norm

#%%
def calc_tuningmap(occupancy, x_edges, y_edges, signaltracking, params = 2.5):
    '''
    Calculate tuningmap - ADAPTED FROM DATAJOINT
    
    @author: simonball, adapted by torstsl
    
    Parameters
    ----------
    occupancy : masked np.array
        Smoothed occupancy. Masked where occupancy low
    x_edges : np.array
        Bin edges in x 
    y_edges : np.array
        Bin edges in y
    signaltracking : DataFrame
        SignalTracking table entry. 3xM df with: 
            1) Cell signal (one cell, all timestamps or filtered by signal)
            2) Head position X (all timestamps, dimentions as signal)
            3) Head position Y (all timestamps, dimentions as signal)
    params : A constant value of 2.5 (float)
        MapParams table entry
    
    Returns
    -------
    tuningmap_dict : dict
        - binned_raw : np.array: Binned raw (unsmoothed) signal
        - tuningmap_raw: np masked array: Unsmoothed tuningmap (mask where occupancy low)
        - tuningmap    : np masked array: Smoothed tuningmap (mask where occupancy low)
        - bin_max    : tuple   : (x,y) coordinate of bin with maximum signal
        - max        : float : Max of signal 
    
    '''
    
    tuningmap_dict = {}
    
    # Add one at end to not miss signal at borders   
    edges_x = x_edges.copy()
    edges_y = y_edges.copy()
    
    # Look up signal per bin  
    
    # Old way from DataJoint, which is slow 
    # binned_signal = np.zeros_like(occupancy.data)
    # edges_x[-1] += 1
    # edges_y[-1] += 1
    # for no_x in range(len(edges_x)-1):
    #    for no_y in range(len(edges_y)-1):
    #        boolean_x = (signaltracking['Head_X'] >= edges_x[no_x]) & (signaltracking['Head_X'] < edges_x[no_x+1])
    #        boolean_y = (signaltracking['Head_Y'] >= edges_y[no_y]) & (signaltracking['Head_Y'] < edges_y[no_y+1])
    #        extracted_signal = signaltracking['Signal'][boolean_x & boolean_y]
    #        binned_signal[no_y, no_x] = np.nansum(extracted_signal)
    
    """
    PROBLEM:    The bin edges output from op.analysis.spatial_occupancy does  
    (fixed)     not match the bin edges calculated underneath. It looks like
                the op. way (when fed with box size) does not scale the 
                tracking, so that the scale of the two axis from the tracking
                are 1:1 cm/units. This doesn't necessarily make sense. 
                By the other method, in which I state number of bins (which I
                want to be 80 cm/ 2,5 cm = 32 bins), I get that 1 bin (2,5 cm)
                of tracking corresponds to 2,789x. In the op. edges, this 
                relationship is 1:1, starting at the lowest and just increasing 
                by 2,5. (For completion: The last instance is added +1 to not  
                lose data, but from what I understand - data is already lost..)
                                
                Plotting ratemaps from the two shows a clear difference, where
                signal_map (new edges) looks better and binned_signal (op. way)
                looks cropped (for high values of both x and y). 
                
                This would explain why I get an error when running this code
                for "edges_x" (op. output), where some of the tracking is 
                notably higher than the edges. This is not an issue with the 
                "old" method above (for no_x.. for no_y).
                
                If this is so, would my occupancy also be wrong? My occupancy
                map and my ratemap should have the same bin edges for the 
                2D histogram. Would have to spesify this in my input?
                
                I have been running:
                    op.analysis.spatial_occupancy(timestamps, np.transpose(headpos),boxSize)
                If I instead run:
                    op.analysis.spatial_occupancy(timestamps, np.transpose(headpos),boxSize,bin_number=32)
                I get almost the same edges as for the method below, np.diff()
                is 2.789 (bin_edges_x) vs. 2.790 (op..occupancy, bin_number=32).
                The op...occupancy uses op.general.accumulate_spatial, which 
                determines the edges by [np.min(x),np.max(x)*1.001]. Probably
                the reason for the slight difference.
                
                TO CONCLUDE: I think I might have used the occupancy calculation
                calculation wrongly, hence this error. I thought it would scale  
                by box size, but it seems like it needs the number of bins to 
                scale correctly. If I change my occupancy, I believe the "new"
                calculaton of ratemaps would also work (and be faster). It 
                would also mean that I have the same bin edges for occupancy 
                and ratemap, as I can load the occupancy bin_edges into this
                function (and thus not calculate them within this function).
                Might run into issues with two dimentions? Don't think so, it
                seems like the op.general.accumulate_spatial calculates x,y 
                separately (hence, they might not be 1:1, which they should in
                a perfect world).
                
                ===============================================================
                Add on 14.01.23: 
                The bin_edges from op.occupancy are made for the output 
                ratemap. This almost always conincides with the binning of the 
                headpos, but there are some instances wherethese don't align. 
                It will then throw an error for the lines:
                    x_ind = np.where(x <= edges_x)[0][0] - 1
                    y_ind = np.where(y <= edges_y)[0][0] - 1
                in which case, the solution is to provide the op.occupancy 
                with bin_edges instead of numbins, like this:
                    x_edges = np.linspace(headpos[:,0].min(), headpos[:,0].max(), 33)
                    y_edges = np.linspace(headpos[:,1].min(), headpos[:,1].max(), 33)
                    
                    op.analysis.spatial_occupancy(timestamps, headpos.T, 80, 
                                            bin_edges = (x_edges, y_edges))
                
                To maintain similarity, this should probably just be default 
                for all ratemap calculations.
             
    """
    # New way that is mostly the same, but computationally much faster (approx. 45 % decrease in running time)
    num_bins = len(edges_x)-1
    occupancy_map = np.zeros((num_bins, num_bins), np.int16)
    signal_map = np.zeros((num_bins, num_bins), np.float64)
    
    # Using imported bin_edges from occupancy calculation    
    for x, y, s in zip(signaltracking['Head_X'], signaltracking['Head_Y'], signaltracking['Signal']):
        x_ind = np.where(x <= edges_x)[0][0] - 1
        y_ind = np.where(y <= edges_y)[0][0] - 1
        if x_ind == -1: x_ind = 0
        if y_ind == -1: y_ind = 0
        occupancy_map[y_ind, x_ind] += 1
        signal_map[y_ind, x_ind] += s       
      
    # All rate maps are flipped after this, np.flipup corrects this    
    tuningmap_dict['binned_raw'] = np.flipud(signal_map)
    signal_map = np.ma.masked_where(occupancy.mask, signal_map)  # Masking. This step is probably unnecessary
    tuningmap_dict['tuningmap_raw'] = np.flipud(signal_map / occupancy)
    
    # Instead of smoothing the raw binned events, substitute those values that are masked in
    # occupancy map with nans.
    # Then use astropy.convolve to smooth padded version of the spikemap 
        
    signal_map[occupancy.mask] = np.nan
    
    kernel = Gaussian2DKernel(x_stddev=params)
  
    pad_width = int(5*params)
    signal_map_padded = np.pad(signal_map, pad_width=pad_width, mode='symmetric')  # as in BNT
    signal_map_smoothed = convolve(signal_map_padded, kernel, boundary='extend')[pad_width:-pad_width, pad_width:-pad_width]
    signal_map_smoothed = np.ma.masked_where(occupancy.mask, signal_map_smoothed)  # Masking. This step is probably unnecessary
    masked_tuningmap = signal_map_smoothed / occupancy

    tuningmap_dict['tuningmap']     = np.flipud(masked_tuningmap)
    tuningmap_dict['bin_max']       = np.flipud(np.unravel_index(masked_tuningmap.argmax(), masked_tuningmap.shape))
    tuningmap_dict['max']           = np.max(masked_tuningmap)
        
    return tuningmap_dict

#%%
def get_signaltracking(dfNAT, cell_no, **kwargs):
    
    """
    Created on Fri Jul  8 23:23:10 2022

    @author: torstsl
    
    signaltracking = get_signaltracking(dfNAT, cell_no, **kwargs)
    
        Generates signaltracking, which is used to calculate rate maps.
        Signal is the signal per cell, and tracking is the head position
        for when this cell is active (signal>0).
    
        OUTPUT: 
            signaltracking (df): Contains one cell's signal and the 
                                headposition for those signals. Type of signal 
                                is defined by input 'signal'.
                                
        INPUT: 
            dfNAT (df): DataFrame from session_dict with all cells' activity 
                        and tracking data.
            cell_no (int): Cell number for whose signal is extracted (NATEX ID)
            **kwargs: signal (str): Signal type, can be either 'deconvolved' or
                                  'dF'. Default is 'deconvolved'. 
                      speed_threshold (int): Filter for minimum speed. If not
                                              given, no threshold is used.
        
    """
     
    if len(kwargs) == 0 or kwargs['signal'] == 'deconvolved':
        signal_string = 'Deconvolved, N'
    elif kwargs['signal'] == 'dF':
        signal_string = 'dF filtered, N'
                  
    if len(kwargs) == 0:
        signaltracking = pd.DataFrame()
        signaltracking['Signal'] = dfNAT.loc[dfNAT[signal_string+str(cell_no)]>0,signal_string+str(cell_no)]
        signaltracking['Head_X'] = dfNAT.loc[dfNAT[signal_string+str(cell_no)]>0,'Head_X']
        signaltracking['Head_Y'] = dfNAT.loc[dfNAT[signal_string+str(cell_no)]>0,'Head_Y']
        
    if kwargs['speed_threshold'] != 0:
        speed_threshold = kwargs['speed_threshold']
        signaltracking = pd.DataFrame()
        signaltracking['Signal'] = dfNAT[(dfNAT[signal_string+str(cell_no)] > 0) & (dfNAT['Body_speed'] >= speed_threshold)][signal_string+str(cell_no)]
        signaltracking['Head_X'] = dfNAT[(dfNAT[signal_string+str(cell_no)] > 0) & (dfNAT['Body_speed'] >= speed_threshold)]['Head_X']
        signaltracking['Head_Y'] = dfNAT[(dfNAT[signal_string+str(cell_no)] > 0) & (dfNAT['Body_speed'] >= speed_threshold)]['Head_Y']
                
    return signaltracking

#%%

def get_signal(dfNAT, cell_no, **kwargs):
    
    """
    Created on Fri Jul  8 23:54:01 2022

    @author: torstsl
    
    cell_signal = get_signal(dfNAT, cell_no, **kwargs)
    
        Generates signal for one cell, spescified by signal type. Similar to 
        signaltracking, but less computational if tracking is abundant.
        
        OUTPUT: 
            signal (df): Contains one cell's signal. The type of signal 
                        is defined by input 'signal'.
                                
        INPUT: 
            dfNAT (df): DataFrame from session_dict with all cells' activity.
            cell_no (int): Cell number for whose signal is extracted
            kwargs: signal (str): Signal type, can be either 'deconvolved' or
                                  'dF'. Default is 'deconvolved'. 
        
    """
       
    if len(kwargs)==0 or kwargs['signal'] == 'deconvolved':
       signal_string = 'Deconvolved, N'
    elif kwargs['signal'] == 'dF':
       signal_string = 'dF filtered, N'
    
    signal = pd.DataFrame()
    signal['Signal'] = dfNAT.loc[dfNAT[signal_string+str(cell_no)]>0,signal_string+str(cell_no)]
    
    return signal

#%%

def plot_ratemap(session_dict, sessionNo, cellNo):
    
    """
    Created on Sat Jan  14 11:56:09 2023

    @author: torstsl
    
    plot_ratemap(session_dict, sessionNo, cellNo)
    
        Quick function to plot a ratemap from simple inputs. This function
        also calculates the ratemap, so it does not require session_dict['Ratemaps']
        to exist. Nice if one wants to check code or debug.
        
        OUTPUT: 
            Plots the ratemap for this cell
                                
        INPUT: 
            session_fict: From load_session_dict()
            sessionNo (int): 0-2, which session to be used
            cellNo (int): NATEX index of the cell to be plotted. 
    """
    
    session_string = ['A','B','A\'']
    dfNAT = session_dict['dfNAT']['dfNAT'+str(sessionNo)]
    timestamps  = dfNAT.Timestamps.to_numpy()
    headpos     = dfNAT[['Head_X','Head_Y']].to_numpy()
    
    x_edges = np.linspace(headpos[:,0].min(), headpos[:,0].max(), 33)
    y_edges = np.linspace(headpos[:,1].min(), headpos[:,1].max(), 33)
    
    occupancy, coverage_prc, bin_edges = op.analysis.spatial_occupancy(timestamps, headpos.T, 80, 
                                                                       bin_edges = (x_edges, y_edges))
    
    signaltracking = get_signaltracking(dfNAT, cellNo, signal = 'deconvolved', speed_threshold = 2.5)
    
    tuningmap_dict = calc_tuningmap(occupancy, bin_edges[0], bin_edges[1], signaltracking, 2.5)
    ratemap_smooth = gaussian_filter(tuningmap_dict['tuningmap'], 1.5)
    ratemap = np.ma.MaskedArray(ratemap_smooth, tuningmap_dict['tuningmap'].mask)
    
    fig, axes = plt.subplots(1,1)
    p1 = sns.heatmap(ratemap, ax = axes, square = True, cmap = 'viridis', cbar = True,
                     robust = True, xticklabels = False, yticklabels = False,
                     mask = tuningmap_dict['tuningmap'].mask)
    p1.set_title('Rate map, box '+session_string[sessionNo]+', N'+str(cellNo))
    
#%%

def shuffle_ratemap(dfNAT, cellNo):
    
    """
    rmap_shuffle = shuffle_ratemap(dfNAT)
    
    Simple funtion that calculates a ratemap, but shufffles the head positions
    by a random number so that the output ratemap is a shuffle of that cell's
    ratemap. 
    
    INPUT:  dfNAT (DataFrame): DataFrame of session info from session_dict
            cellNo (int): NATEX index of cell to check
            
    OUTPUT: ratemap_shuffle (np.ma.array): Tuning map masked by occupancy.
    
    """
    
    timestamps  = dfNAT.Timestamps.to_numpy()
    headpos     = dfNAT[['Head_X','Head_Y']].to_numpy()
   
    # Shuffle headpos
    roller = np.random.randint(30,timestamps.size-30)
    headpos_shuffle = np.roll(headpos, roller, axis = 0)
    
    dfNAT.Head_X = headpos_shuffle[:,0]
    dfNAT.Head_Y = headpos_shuffle[:,1]
    
    x_edges = np.linspace(headpos_shuffle[:,0].min(), headpos_shuffle[:,0].max(), 33)
    y_edges = np.linspace(headpos_shuffle[:,1].min(), headpos_shuffle[:,1].max(), 33)
    
    occupancy, coverage_prc, bin_edges = op.analysis.spatial_occupancy(timestamps, headpos_shuffle.T, 80, 
                                                                       bin_edges = (x_edges, y_edges))
    
    signaltracking = get_signaltracking(dfNAT, cellNo, signal = 'deconvolved', speed_threshold = 2.5)
    
    tuningmap_dict = calc_tuningmap(occupancy, bin_edges[0], bin_edges[1], signaltracking, 2.5)
    ratemap_smooth = gaussian_filter(tuningmap_dict['tuningmap'], 1.5)
    rmap_shuffle = np.ma.MaskedArray(ratemap_smooth, tuningmap_dict['tuningmap'].mask)
    
    return rmap_shuffle

#%% 

def get_cell_masks(ExperimentInfo):
    """
    Get the maximum intensitity projection image from the session and the place cell masks

    Parameters
    ----------
    ExperimentInfo : dict
        The experiment information from session_dict['ExperimentInformation'].

    Returns
    -------
    maxprojection : np.array (2D)
        The maximum intensity porojection from suite2p.
    maxprojection_mask : np.array (3D)
        3D array with cell masks from suite2p. Dimentions are (N, X, Y), 
        where N = number of cells, X and Y corresponds to  maxprojection.shape. 
        Only the cells' shape have value, the other coordinates are masked. 

    """
    
    maxprojection = ExperimentInfo['ImagingOptions']['max_proj']

    nCells = ExperimentInfo['TotalCell'].astype(int)
    yrange = ExperimentInfo['ImagingOptions']['yrange'] # Valid y-range used for cell detection
    xrange = ExperimentInfo['ImagingOptions']['xrange'] # Valid x-range used for cell detection

    # Get the masks for all cells in this experiment and the centre of the ROIs
    masked_max = np.full([nCells,maxprojection.shape[0],maxprojection.shape[1]], False)
    
    for cellNo in range(nCells):   
        ypix_max = ExperimentInfo['CellStat'][cellNo]['ypix']-yrange[0] # List of all y-coordinates for this cell's mask
        xpix_max = ExperimentInfo['CellStat'][cellNo]['xpix']-xrange[0] # List of all x-coordinates for this cell's mask
        
        masked_max[cellNo][ypix_max, xpix_max] = True
        
    maxprojection_mask = np.ma.array(masked_max, mask = masked_max==False)

    return maxprojection, maxprojection_mask  
  
#%%

def plot_maxproj_feature(maxproj, masks, feature, title_str, **kwargs):
    """
    Created on Wed Jan 25 15:55:39 2023

    @author: torstsl
    
    Plots the max intensity projection and overlays the filled contours 
    of each cell. Fill color correspondd to value of some feature.
    
    Inputs:
        maxproj (array): 2D array, meant to be the maximum intensity projection 
                         from suite2p (motion corected).
        masks (ma. array): 3D array with cell masks from suite2p. Dimentions are
                       (N, X, Y), where N = number of cells, X and Y 
                       corresponds to the array maxproj. Only the cells' shape
                       have value, the other coordinates are masked. 
        feature(array-like): 1D array-like of length == N. This is the feature 
                        that decides the color of each cells' mask. The color
                        is chosen by binning the feature between its min and  
                        max in 100 steps, each bin with a corresponding color. 
                        The colors are based on the cmap 'OrRd'. If there is no
                        feature for this cell (for some reason, i.e. no 
                        activity) the contour is filled with a blue color 
                        (hex: #35a1ab).                    
        title_str (str): String with title for the plot
        
        **kwargs
              background (str): Either 'maxint' (default) or 'black'. 
                          If background = 'maxint', the maxproj is used as the 
                          background with more transparent cells on top. If 
                          background = 'black', less transparent cells are 
                          plotted on a completely black background.
              vmin, vmax (float): To set limits of the color map for the feature. 
                          Else, this is normalized to the feature input. 
              palette (str): String of cmap to use, default is 'OrRd'.
    """
    background = kwargs.get('background', 'maxint')
    
    nCells = len(masks)
    
    if 'palette' in kwargs.keys():
        palette_c = sns.color_palette(kwargs['palette'], 100).as_hex()
        palette_str = kwargs['palette']
    else: 
        palette_c = sns.color_palette('OrRd', 100).as_hex()
        palette_str = 'OrRd'
        
    if ('vmin' and 'vmax') in kwargs.keys():
        norm = mpl.colors.Normalize(vmin=kwargs['vmin'], vmax=kwargs['vmax'])
        sm = plt.cm.ScalarMappable(cmap=palette_str, norm=norm)
        edges = np.linspace(kwargs['vmin'], kwargs['vmax'], 100)
        
    else:
        norm = mpl.colors.Normalize(vmin=min(feature), vmax=max(feature))
        sm = plt.cm.ScalarMappable(cmap=palette_str, norm=norm)
        edges = np.linspace(min(feature), max(feature), 100)
    
    feature_color= []
    
    for c in feature:
        if not np.isnan(c):
            c_ind = np.where(c <= edges)[0][0] - 1
            feature_color.append(palette_c[c_ind])
        else: feature_color.append(np.nan)
    
    if background == 'maxint':       
        fig, ax = plt.subplots(figsize = (8,8))
        fig.patch.set_alpha(0)
        ax.set_title(title_str+', '+str(nCells)+' cells', color = 'w')
        ax.axis('off')
        ax.imshow(maxproj,cmap='gist_gray')
        ax.set_aspect('equal')
        for cellNo in range(nCells):
            if type(feature_color[cellNo]) == str: 
                ax.contourf(masks[cellNo], colors=feature_color[cellNo], alpha = 0.75)
            elif  np.isnan(feature_color[cellNo]) == True: # If there is no value for the cell in this session, make in a dim cyan
                ax.contourf(masks[cellNo], colors='#35a1ab', alpha = 0.3)                   
        cb = plt.colorbar(sm, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color='w', labelcolor = 'w')
        cb.outline.set_edgecolor('w')
        
    elif background == 'black':     
        fig, ax = plt.subplots(figsize=(8,8))
        fig.patch.set_alpha(0)
        ax.set_title(title_str+', '+str(nCells)+' cells', color = 'w')
        ax.set_xlabel(None)  
        ax.set_xticks([])
        ax.set_ylabel(None)  
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.set_facecolor('black')
        ax.set_aspect('equal')
        for cellNo in range(nCells):
            if type(feature_color[cellNo]) == str: 
                ax.contourf(masks[cellNo], colors=feature_color[cellNo], alpha = 0.5)
            elif  np.isnan(feature_color[cellNo]) == True: # If there is no value for the cell in this session, make in a dim cyan
                ax.contourf(masks[cellNo], colors='#35a1ab', alpha = 0.5)                   
        cb = plt.colorbar(sm, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color='w', labelcolor = 'w')
        cb.outline.set_edgecolor('w')

#%%
"""
Created on Wed Jan 18 17:53:41 2023

@author: torstsl

Plots a 2D density heatmap from scattered data base on input x, y (coords).

"""
def plot_2d_density(x,y):
    #data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 3]], 200) 
    
    if len(x) != len(y): TypeError('Inputs x and y must have same size')
    
    fig, ax = plt.subplots(1,2, sharex = True, sharey = True)
    ax[0].scatter(x,y, s = 0.025, alpha = 0.25)
    ax[0].set_title('Scatter')
    ax[0].set_ylabel('Field distance (cm)')
    ax[0].set_xlabel('Anatomical distance (\u03BCm)')
    ax[0].set_xlim([min(x),max(x)])
    ax[0].set_ylim([min(y),max(y)])

    nbins = 50
   
    data = np.zeros([len(x),2])
    data[:,0], data[:,1] = x, y
    
    k = kde.gaussian_kde(data.T)
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    ax[1].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap='OrRd')
    ax[1].contour(xi, yi, zi.reshape(xi.shape), cmap='gist_gray', alpha = 0.75, linewidths = 0.5)
    ax[1].set_title('2D density')
    ax[1].set_xlabel('Anatomical distance (\u03BCm)')
    ax[1].set_xlim([min(x),max(x)])
    ax[1].set_ylim([min(y),max(y)])
    ax[1].tick_params(
        axis='y',          # Changes apply to this axis
        which='both',      # Both major and minor ticks are affected
        left=False,        # Ticks along this edge are off
        right=False,       # Ticks along this edge are off
        labelleft=False    # Labels along this edge are off
        )
      
    plt.tight_layout()

#%%

def bin_signal_trace(signal, bin_size, smoothing):
    
    """
    Created on Fri Nov  25 11:10:41 2022

    @author: torstsl
    
    bin_signal = bin_signal_trace(signal, bin_size)
    
        Bins the signal into bins ordered by the input bin_size. The input
        signal is binned and the mean value of the bin is returned. The signal 
        may be smoothened by a gaussian filter (sigma = 1.5) before returned.
        
        OUTPUT: 
            bin_signal (array): The binned signal. 1D array of length 
                                   = len(signal)/bin_size
                                or 2D array of shape signal.shape
                                
        INPUT: 
            signal (array like): 1D trace (dF/F or deconvolved or anything)
                                or 2D DataFrame with signals (cells in columns).
            bin_size (int): How many datapoints to put into one bin.
            smoothing (boolean): If True, the mean signal is smoothened with a 
                                gaussian filter (sigma = 1.5) before returned.
    """
    
    if type(signal) == list:
        raise TypeError('Input signal is a list, must be array like')
    
    bins = int(len(signal)/bin_size)
    slices = np.linspace(0, len(signal), int(bins)+1, True).astype(int)
    counts = np.diff(slices)
    
    if signal.ndim == 1:
        mean_signal = np.add.reduceat(np.array(signal), slices[:-1]) / counts
        bin_signal = gaussian_filter(mean_signal, 1.5)
        
    elif signal.ndim > 1:
        cells = signal.shape[1] # Number of cells with signals
        bin_signal = np.full([bins, cells], np.nan)
        
        for x in range(cells):
            mean_signal = np.add.reduceat(np.array(signal.iloc[:,x]), slices[:-1]) / counts
            if smoothing == True: bin_signal[:,x] = gaussian_filter(mean_signal, 1.5)
            else: bin_signal[:,x] = mean_signal
            
        #mean = np.add.reduceat(np.array(signal), slices[:-1]) / np.array(cells*[counts]).T
        #bin_signal = np.full(mean.shape, np.nan)
        
        # Do the filtering 1D instead of directly (which becomes 2D)
        #for x in range(cells): bin_signal[:,x] = gaussian_filter(mean[:,x], 1.5) 
        
    #fig, ax = plt.subplots(3,1)
    #ax[0].plot(signal)
    #ax[1].plot(mean)
    #ax[2].plot(gaussian_filter(mean, 1.5))
    #plt.tight_layout()

    return bin_signal
#%%
def calc_corr_mtx_mean(data):
    
    """
    Created on Wed Dec  7 11:58:31 2022

    @author: torstsl
    
    corr_mtx_mean = corr_mtx_mean(data):
    
        Correlates the data provided using np.corrcoef. It then takes the mean
        of all the (unique) correlations and returns this value.
        
        OUTPUT: 
            corr_mtx_mean (float): The mean of the correlations in the
                                    correlation matrix for the data.
                                
        INPUT: 
            data (array-like): The data to be correlated. Should be array-like
                                or a list of data. Data are analysed as observations
                                per row, with variables per observation per column.
                                Hence, np.corrcoef(data, rowvar = True)
    """
    
    if len(data) == 1: print('Data only has one value')
    
    corrMtx = np.corrcoef(data, rowvar = True)
    mask = np.triu(np.ones_like(corrMtx, dtype=bool), 1)
    corr_mtx_mean = np.nanmean(corrMtx[mask])
    
    return corr_mtx_mean

#%%
def ratemap_correlation(mainArray, testArray):
    
    """
    Created on Wed Jan  4 12:05:38 2023

    @author: torstsl
    
    stats = ratemap_correlation(mainArray, testArray):
    
        Used in the script neighbourhood. It takes the ratemap of one cell 
        and calculated the correlation to a set of test cells. The returned
        value is the mean correlation and mean p-value of these correlations. 
        The test cells are not correlated to eachother. 
        
        OUTPUT: 
            stats (tuple): The mean of the correlations (mean Pearson R) and 
                           the mean p-value from the correlations.
                                
        INPUT: 
            mainArray(array-like): The cell who's ratemap the others will be
                                    correlated to. Should be 2D (dimX, dimY).
            testArray(array-like): The cell(s) that will be correlated to the 
                                    mainArray. Could be several cells, in which 
                                    the array should be 3D (nCells, dimX, dimY)
    
    """
    
    values = np.full([len(testArray), 2], np.nan)
    
    for x in range(len(testArray)): values[x,:] = pearsonr(mainArray.flatten(), testArray[x].flatten())
        
    stats = np.nanmean(values[:,0]), np.nanmean(values[:,1])
       
    return stats


#%%
def shuffle_bin_signal(bin_signal, groups, nShuffles):
    
    """
    Created on Wed Dec  7 12:21:48 2022

    @author: torstsl
    
    groups_shuffle = shuffle_bin_signal(bin_signal, groups, nShuffles):
    
        Used in grids_anatomy.py for shufflings groups of place cells that are
        anatomically organized within the same 2D hisogram bin. The number of 
        cells within one bin is constant, but the corresponding signal of these
        cells are shuffled randomly. The signals per shuffle (per group) are
        correlated, and the mean of the correlation is calculated as this 
        groups shuffled mean correlation. The true correlation can then be  
        compared to the shuffled distribution.
        
        OUTPUT: 
            groups_shuffle (dict): The mean of the correlations in the
                                    correlation matrix for the data, per group.
                                    First value = mean (groups_shuffle[x][0])
                                    Second = 99th percentile (groups_shuffle[x][1])
                                    If there's only one cell per bin, the 
                                    returned values are [NaN, NaN].
            shuffle_mean (dict): Dict of np.arrays that corresponds to the mean 
                                correlation per shuffle of the signal. Rows are
                                shuffle number, columns are group number.
                                
        INPUT: 
            bin_signal (array-like): The data to be correlated. Should be array-like
                                or a list of data. 
            groups (dict): The groups that the data are binned into. To be kept in shuffling.
            nShuffles (int): Number of shuffles to be performed.
    """
    
    # Pre define variable
    shuffle_mean = np.zeros([nShuffles,len(groups.keys())])
    bin_signal_array = np.asarray(bin_signal) # All signals binned and put into an np.array 
    
    # Run shuffling, get the binned data per shuffle (per group)
    for shuffleNo in range(nShuffles):
        counter = 0
        for x in groups.keys():
            
            # This shuffles random signal, new seed for each group
            randoms = np.random.randint(0, len(bin_signal), len(groups[x]))
            
            group_data = bin_signal_array[randoms,:] 
    
            # Get the mean of the correlation matrix per group
            if len(group_data) == 1: shuffle_mean[shuffleNo,counter] = np.nan
            
            elif len(group_data) > 1:
                shuffle_mean[shuffleNo,counter] = calc_corr_mtx_mean(group_data)
   
            counter += 1
            
    mean_shuffles = np.nanmean(shuffle_mean, axis = 0)    
    prc_shuffles = np.nanpercentile(shuffle_mean,99,axis=0)    
    groups_shuffle = {}
    
    # x is the bin group, y is the mean of the shuffles, z is the 99th percentile of the shuffles
    for x, y, z in zip(groups.keys(), mean_shuffles, prc_shuffles): groups_shuffle[str(x)] = [y, z]
   
    return groups_shuffle, shuffle_mean

#%%
def calc_independency(X, Y, RxC):
    """
    chiStats = calc_independency(X, Y, RxC)
    
    Calculates wethger or not the separate continuous variables X and Y are 
    independent using chi-square testing.
    
    First, it calculates a contengency table based on histcounts by dimentions
    given by RcX (shape of the contengency table). The table is then used to 
    test whether or not the two variabels are independent. 
    
    The output is the chiStats, a tuple with:
        chiStats[0]: X^2 value
        chiStats[1]: p-value
        chiStats[2]: Degrees of freedom (dof)
        chiStats[3]: Expected frequencies
    
    If categorical data, put the in a pd.DataFrame, use: 
    cTable = pd.crosstab(df["X"], df["Y"])
    chi2_contingency(cTable)
    
    """
    
    countsX, edgesX = np.histogram(X,RxC[0])
    countsY, edgesY = np.histogram(Y,RxC[1])
    
    cTable = np.array([countsX, countsY])
    chiStats = chi2_contingency(cTable)

    return chiStats 
#%%
def calc_blb(data, subsamples, bootstraps):
    """
    Bag of little bootstraps: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5041595/
     1) Draw s subsamples of size m from the original data of size n
     2) For each s, draw r bootstraps of size n, and obtain the CI (or point estimate) for each r sample
     3) The s estimates are averaged to the overall estimate
    
    """
    
    s = subsamples
    n = data.size
    m = round(n**0.5)
    r = bootstraps
    
    means = np.full(s, np.nan)
    cis = np.full([s,2], np.nan)
    stds = np.full(s, np.nan)
    
    for sample in range(s):
    
        temp = np.random.choice(data, m, replace = True)
        
        mean_temp, std_temp, ci_temp = np.full([r], np.nan), np.full([r], np.nan), np.full([r,2], np.nan)
        
        for resample in range(r):
            temper = np.random.choice(temp, n, replace = True)
            mean_temp[resample] = np.mean(temper)
            std_temp[resample] = np.std(temper)
            ci_temp[resample] = np.mean(temper) - 1.96*np.std(temper), np.mean(temper) + 1.96*np.std(temper)
        
        means[sample] = np.mean(mean_temp)
        stds[sample] = np.mean(std_temp)
        cis[sample,:] = np.mean(ci_temp, axis = 0)
    
    return means, stds, cis
#%%
def ttest_paired(x1, x2):
     
     """
     tstats_pair = ttest_paired(x1, x2)
     
     Perform paired t-test. Calculates t-statistics, p-value and confidence intervals
     
     INPUT: 
         x1,x2 (np.array): Array of data. x1 and x2 must be related and sorted, of same size
     
     OUTPUT: 
         tstats_pair (pd.DataFrame): DataFrame with test statistics. Contains t,
             degrees of freedom, p-value, difference in means, confidence interval.
     
     """
     
     # Calculate mean difference and standard error of mean
     delta = np.nanmean(x1)-np.nanmean(x2)
     stderr = sem(x1-x2)
     
     # Calculate test statistics and p-value with confidence intervals
     tstat, pval = ttest_rel(x1, x2)
     
    
     ci = t.interval(0.95, x1.size-1, loc = delta, scale = stderr)
     
     return pd.DataFrame(np.array([tstat,x1.size-1,pval,delta,ci[0],ci[1]]).reshape(1,-1),
                          columns=['t_stat','df','pvalue','delta','ci_low','ci_up'])
#%%     
def ttest2(x1, x2): 
     
     """
     tstats_two = ttest2(x1, x2)
     
     Perform twp-sample t-test. Calculates t-statistics, p-value and confidence intervals.
     Assumes unequal variance, and calculates a pooled standard error. From StackExchange
     
     INPUT: 
         x1,x2 (np.array): Array of data. x1 and x2 must be related and sorted, of same size
     
     OUTPUT: 
         tstats_pair (pd.DataFrame): DataFrame with test statistics. Contains t,
             degrees of freedom, p-value, difference in means, confidence interval.
     
     """
     
     # Calculate pooled standard error 
     n1 = np.sum(~np.isnan(x1))
     n2 = np.sum(~np.isnan(x2))
     m1 = np.nanmean(x1)
     m2 = np.nanmean(x2)
     
     v1 = np.nanvar(x1, ddof=1)
     v2 = np.nanvar(x2, ddof=1)
     
     pooled_se = np.sqrt(v1 / n1 + v2 / n2)
     delta = m1-m2
     
     tstat = delta /  pooled_se
     df = (v1 / n1 + v2 / n2)**2 / (v1**2 / (n1**2 * (n1 - 1)) + v2**2 / (n2**2 * (n2 - 1)))
     
     # Calculate p-value
     p = 2 * t.cdf(-abs(tstat), df)
     
     # Calculate the confidence intervals
     lb = delta - t.ppf(0.975,df)*pooled_se 
     ub = delta + t.ppf(0.975,df)*pooled_se
   
     return pd.DataFrame(np.array([tstat,df,p,delta,lb,ub]).reshape(1,-1),
                          columns=['t_stat','df','pvalue','delta','ci_low','ci_up'])
#%%

def signedrank(data1, data2, **kwargs):
    
    """
    Calculates confidence intervals for the Wilcoxon signed-rank test using bootstrapping.

    Parameters:
        data1 (array-like): The first paired sample.
        data2 (array-like): The second paired sample.
        **kwargs:
            n_iterations (int): The number of bootstrap iterations (default: 1000).
            alpha (float): The desired confidence level (default: 0.95).

    Returns:
        tuple: A tuple containing the test statistic, p-value, and confidence interval.

    Example:
        data1 = np.array([1, 2, 3, 4, 5, 6, 7])
        data2 = np.array([2, 3, 4, 5, 6, 8, 10])
        statistic, p_value, confidence_interval = signedrank(data1, data2)
        
    """

    n_iterations = kwargs.get('n_iterations', 1000)
    alpha = kwargs.get('alpha', 0.95)
    
    # Perform the Wilcoxon signed rank test
    statistic, p_value = wilcoxon(data1, data2)
    
    # Number of bootstrap samples (for confidence interval of median)
    n_iterations = 1000
    
    # Array to store the bootstrap statistics
    bootstrap_stats = np.zeros(n_iterations)
    
    # Perform bootstrapping
    for i in range(n_iterations):
        # Create a bootstrap sample by resampling with replacement
        bootstrap_data1 = np.random.choice(data1, size=len(data1), replace=True)
        bootstrap_data2 = np.random.choice(data2, size=len(data2), replace=True)
        
        # Perform the Wilcoxon signed rank test on the bootstrap sample
        bootstrap_stats[i], _ = wilcoxon(bootstrap_data1, bootstrap_data2)
    
    # Calculate the confidence intervals
    lower = np.percentile(bootstrap_stats, (1 - alpha) / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 + alpha) / 2 * 100)
    
    return statistic, p_value, (lower, upper)        

#%%

def calc_mannwhitneyu(x, y, alpha=0.05):
    """
    Calculate the Mann-Whitney U test statistic and confidence interval.
    
    Parameters:
        x (array-like): First sample.
        y (array-like): Second sample.
        alpha (float): Significance level for the confidence interval.
        
    Returns:
        dict: Dictionary containing 'statistic' (test statistic),
              'pvalue' (p-value), and 'confidence_interval' (confidence interval).
    """
    # Perform Mann-Whitney U test
    statistic, pvalue = mannwhitneyu(x, y, alternative='two-sided')
    
    # Calculate sample sizes
    n1 = len(x)
    n2 = len(y)
    
    # Calculate standard error for the test statistic
    se = np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
    
    # Calculate z-value for the confidence interval
    z = norm.ppf(1 - alpha / 2)
    
    # Calculate rank-based effect size (r)
    r = (statistic - (n1 * n2 / 2)) / se
    
    # Calculate confidence interval
    lower_bound = r - z * se
    upper_bound = r + z * se
    
    # Create dictionary with results
    result = {
        'statistic': statistic,
        'pvalue': pvalue,
        'confidence_interval': (lower_bound, upper_bound)
    }
    
    return result
#%%

def f_test(group1, group2):
    
    """
    Perform F-test and calculate confidence intervals.
    Note that this is very sensitive to non-normality.
    
    Parameters:
        group1 (array-like): Array of datapoints.
        group2 (array-like): Array of datapoints.
        
    Returns:
        tuple: Confidence interval as a tuple (lower_bound, upper_bound).
    """
    
    if np.var(group1) < np.var(group2):
        g1 = group2
        g2 = group1        
        
    else:
        g1 = group1
        g2 = group2
        
    F = np.var(g1, ddof=1)/np.var(g2, ddof=1)
    pval = 1 - f.cdf(F, g1.size-1, g2.size-1)
    
    # Define degrees of freedom for the numenator and denominator
    dfn, dfd = len(group1)-1, len(group2)-1
    alpha = 0.05
    
    # Calculate the critical values
    lower_critical = f.ppf(alpha / 2, dfn, dfd)
    upper_critical = f.ppf(1 - alpha / 2, dfn, dfd)
    
    # Calculate the confidence interval
    ci = (1 / upper_critical, 1 / lower_critical)
    
    return F, pval, ci
    
#%% 

def cohen_d(data1, data2):

    """
    Calculate Cohen´s D as a measure of effect size. Must provide two samples, 
    that don´t need to be of similar size, and calculates a pooled std for the 
    samples. The effect size if defined as the number of standard deviations 
    the difference between the sample means are, and ranges from 0 to infinity. 
    Low Cohen´s D (i.e. <0.2) may indicate low practical significance, even if 
    hypothesis testing are significant.
    
    	data1: Array of values for sample 1, must be one-dimentional
    	data2: Array of values for sample 2, must be one-dimentional
     
    	cd: Cohen´s D, float
        
    """

    # Get the size of each sample
    n1, n2 = len(data1), len(data2)

    # Calculate the mean and variance of the samples
    m1, m2 = np.mean(data1), np.mean(data2)
    s1, s2 = np.var(data1, ddof=1), np.var(data2, ddof=1)

    # Calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    
    # Return the effect size (Cohen´s D)
    d = (abs(m1 - m2)) / s

    return d

#%%

def get_placefields_distance(session_dict):
    
    """    
    Calculate the distance between place fields for place cells. 
    If there are several place fields, picks out the most prominent one. 
    
    Input: Session dictionary
    Returns: 
        placefield: dict of NATs with the place field data. 
        placefield_coords: dict of centroid coordinates for the place fields
        field_distance: dict of the pairwise distances between place fields (cm)
            
    """
    
    nSessions = len(session_dict['dfNAT'])
    placecell_dict = session_dict['Placecell']
    
    placefield = {}
    placefield_coords = {}
    field_distance = {}
    
    binSize = 2.5
    
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
        
        # Calculate the pairwise distance between place fields of place cells, multiply by binning to get answer in cm          
        field_distance['NAT'+str(sessionNo)] = (np.linalg.norm(placefield_coords['NAT'+str(sessionNo)]            
           - placefield_coords['NAT'+str(sessionNo)][:,None], axis=-1))*binSize

    return placefield, placefield_coords, field_distance

#%% 

def calc_weights(x, y, feature):
    
    """
    calc_weights(x, y, feature)
    
    This function calculates the weights object to be used in esda.Moran.
    That function is meant for directly neighbouring data, but can be used
    for scatter data (islands) without direct boarders if the weights are
    adjusted accordingly. Here, this is achieved by saying that each datapoint 
    in the scatter is neighbouring all other datapoints. For each pair of data,
    the weight between them is calculated as the inverse of the distance
    between them. 
    
    This function takes the coordinates (x,y) of the datapoints and their 
    variable of interest (feature) as inputs, and returns the weight object
    that can then be used further in esda.Moran.
    
    The function does not handle NaN-values within the features. The input 
    must have datapoints that have an actual feature value to it.
    
    INPUT:
        x,y (array-like): 1D vector of datapoints' x- and y-coordinates,
                            makes out the position for a scatter. Must have 
                            equal length.
        feature (array-like): 1D vector of the variable of interest for each
                            datapoint corresponding to x,y.
        
    OUTPUT:
        W (obj): Weights object from libpysal that describes the weigh between
                all neighbours. The weight between two pairs of datapoints are
                calculated as the inverse of the distance between the points.
                W(A<-->B) = 1/distance(A<-->B)
                The weights object contains a dictionary (like) that for one 
                key notes the neighbours for each single datapoint, and for 
                another one the weights value between these datapoints. 
    """
    if not len(x) == len(y) == len(feature): 
        KeyError(print('Inputs are not of same size'))
        
    if np.sum(np.isnan(feature))>0: 
        KeyError(print('NaN in feature (variable of interest), cannot calculate Morans I'))
    
    # Calculate distance between datapoints
    coords = np.full([len(x),2], np.nan)
    coords[:,0], coords[:,1] = x, y
    dist = np.linalg.norm(coords - coords[:,None], axis = -1)
    
    # Calculate the weights, set diagonal to 0 (wij, i==j --> 0).
    dist[np.diag_indices_from(dist)] = np.nan # To omit devide by 0 warning
    w = 1/dist
    w[np.diag_indices_from(w)] = 0 # Set NaN to 0 for weight of pair with itself
    
    keys = []
    for i in range(len(feature)): keys.append(i)
    keys = np.array(keys)

    neighbours, weights = {}, {}
    for i in range(len(feature)): 
        others = keys[~np.isin(keys,i)]
        neighbours[keys[i]] = others
        weights[keys[i]] = w[i, others]

    # Get the weights object
    W = libpysal.weights.W(neighbours, weights)
    
    return W

#%%

def calc_session_ratemaps(session_dict):

    """
    Created on Mon Aug 22 09:47:53 2022

    @author: torstsl

    calc_session_ratemaps(session_dict)

        Calculates ratemaps (smoothened) for all cells in the session. 
        Iterates over trials/subsessions and over all cells. Only needs 
        the session_dict input, returns a dict with all ratemaps. 
        
        OUTPUT:
            sessions_ratemap (dict): Dictionary of size nCells of ratemaps as 
                                     arrays. If there are subsessions, it has
                                     nSessions dicts each with nCells arrays.
                                
        INPUT:
            session_dict (dict): Session_dict as generated from this package. 
                                 Accepts both single sessions and subsessions.                       

    """
    
    # LOAD DATASET
    dfNAT = session_dict['dfNAT']
    
    if type(session_dict['dfNAT']) == dict:
        nSessions = len(session_dict['dfNAT'])
    else: 
        nSessions = 1
        
    nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)

    minSpeed = 2.5 # Speed threshold for filtering out signal 
    boxsize = 80 # Box size for square open field experiment
    nBins = int(boxsize/2.5)
     
    print('Found '+str(nSessions)+' (sub)session(s) with '+str(nCells)+' cells')
    
    # GET POSITION DATA AND CALCULATE RATE MAPS

    if nSessions > 1:
        timestamps = {}
        headpos = {}
        session_ratemaps = {}
    
        # Iterate through all trials within this session_dict
        for k in range(nSessions):
            key = 'dfNAT'+str(k)
            timestamps[key] = session_dict['dfNAT'][key].Timestamps.to_numpy()
            headpos[key] = session_dict['dfNAT'][key][['Head_X','Head_Y']].to_numpy()
            
            x_edges = np.linspace(headpos[key][:,0].min(), headpos[key][:,0].max(), nBins+1)
            y_edges = np.linspace(headpos[key][:,1].min(), headpos[key][:,1].max(), nBins+1)
            
            sub_session_ratemaps = {}
            
            # Calculate occupancy for this session
            occupancy, coverage_prc, bin_edges = op.analysis.spatial_occupancy(timestamps[key], headpos[key].T,
                                                                               boxsize, bin_edges = (x_edges, y_edges))
        
            # Get the signal and coherent tracking per cell, then calculate the rate map
            for cellNo in range(1,nCells+1): 
                signaltracking = get_signaltracking(dfNAT[key], cellNo,signal='deconvolved',speed_threshold = minSpeed)

                tuningmap_dict = calc_tuningmap(occupancy,bin_edges[0],bin_edges[1], signaltracking, 2.5)
                ratemap_smooth = gaussian_filter(tuningmap_dict['tuningmap'], 1.5)
                ratemap_smooth = np.ma.masked_array(ratemap_smooth,tuningmap_dict['tuningmap'].mask)

                sub_session_ratemaps['N'+str(cellNo)] = ratemap_smooth
        
            session_ratemaps[key] = sub_session_ratemaps
        
    elif nSessions == 1:
        session_ratemaps = {}

        timestamps = session_dict['dfNAT'].Timestamps.to_numpy()
        headpos    = session_dict['dfNAT'][['Head_X','Head_Y']].to_numpy()
        
        x_edges = np.linspace(headpos[:,0].min(), headpos[:,0].max(), nBins+1)
        y_edges = np.linspace(headpos[:,1].min(), headpos[:,1].max(), nBins+1)
        
        # Calculate occupancy for this session
        occupancy, coverage_prc, bin_edges = op.analysis.spatial_occupancy(timestamps, headpos.T,
                                                                           boxsize, bin_edges = (x_edges, y_edges))
   
        # Get the signal and coherent tracking per cell, then calculate the rate map
        for cellNo in range(1,nCells+1):
            signaltracking = get_signaltracking(dfNAT, cellNo, signal='deconvolved', speed_threshold = minSpeed)

            tuningmap_dict = calc_tuningmap(occupancy, bin_edges[0], bin_edges[1], signaltracking, 2.5)
            ratemap_smooth = gaussian_filter(tuningmap_dict['tuningmap'], 1.5)
            
            session_ratemaps['N'+str(cellNo)] = ratemap_smooth

    return session_ratemaps

#%%

def intersession_stability(session_dict, cellNo, testCellNo, nShuffle):
    
    """
    Created on Wed Jan 11 12:11:33 2023

    @author: torstsl

    This function checks whether or not cells are stable across sessions within 
    the same environment. It does so by calculating the correlation between the
    ratemaps in box A in two exposures. The correlation should be high if the cell 
    is a true stable place cell. The function runs a shuffling to determine if the
    level of correlation is significant. The first exposure is kept constant, 
    whilst the data for the second exposure is shuffled. New ratemaps are 
    calculated for the shuffled distributions, and these are compared to the 
    true ratemap in the first exposure. The true correlation should be higher than 
    that of the 95th percentile from the shuffled distribution.

    intersession_stability(session_dict, cellNo, nShuffle):
        INPUTS: 
            session_dict (dict): From load_session_dict()
            cellNo (int): NATEX index of cell to be checked
            nShuffle (int): Number of shuffles to perform
        OUTPUTS:
            corr_stats (tuple): Pearson R-value and its corresponding p-value 
            shuffle_stats (tuple): Mean of the shuffled distribution's R-values 
                                    and the 95th percentile of this distribution
            jobNo (int): To be used for parallelization of the function

"""
    jobNo = testCellNo    
    
    ratemapA = session_dict['Ratemaps']['dfNAT0']['N'+str(cellNo)]
    ratemapA2 = session_dict['Ratemaps']['dfNAT2']['N'+str(cellNo)]
    
    # Correlate the two ratemaps
    corr_stats = pearsonr(ratemapA.flatten(), ratemapA2.flatten())
    
    # Calculate occupancy and bin_edges
    dfNAT2 = session_dict['dfNAT']['dfNAT2']
    timestamps = dfNAT2.Timestamps.to_numpy()
    headpos = dfNAT2[['Head_X','Head_Y']].to_numpy()
    
    x_edges = np.linspace(headpos[:,0].min(), headpos[:,0].max(), 33)
    y_edges = np.linspace(headpos[:,1].min(), headpos[:,1].max(), 33)
    
    occupancy, coverage_prc, bin_edges = op.analysis.spatial_occupancy(timestamps, headpos.T, 80, 
                                                bin_edges = (x_edges, y_edges))

    # Define a DataFrame for this cell
    dfCell = pd.DataFrame()
    dfCell['Head_X'], dfCell['Head_Y'] = headpos[:,0], headpos[:,1]
    dfCell['Signal'] = dfNAT2['Deconvolved, N'+str(cellNo)]
    
    # Prepare for shuffling
    iShuffle = np.random.randint(30, len(timestamps)-30, nShuffle) # Random shift
    dfShuffle = dfCell.copy()
    signal = np.array(dfCell['Signal'])
    
    # Shuffle the signal
    signalShuffle = np.full([len(timestamps), nShuffle], np.nan)          
    for ii in range(nShuffle): signalShuffle[:,ii] = np.roll(signal,iShuffle[ii]) # Shuffle event train 
    
    # Calculate ratemap for each shuffle and correlate it with the first ratemap
    shuffle_r = []
    
    for shuffleNo in range(nShuffle):
        dfShuffle.Signal = signalShuffle[:,shuffleNo]
        shuffle_tuningmap_dict = calc_tuningmap(occupancy, bin_edges[0], bin_edges[1], dfShuffle, 2.5)
        shuffle_ratemap_smooth = gaussian_filter(shuffle_tuningmap_dict['tuningmap'], 1.5)
        shuffle_ratemapA2 = np.ma.MaskedArray(shuffle_ratemap_smooth, shuffle_tuningmap_dict['tuningmap'].mask)
        
        shuffle_r.append(pearsonr(ratemapA.flatten(), shuffle_ratemapA2.flatten())[0])
    
    # Get the mean and 95th percentile of the shuffled correlation values    
    shuffle_stats = np.nanmean(shuffle_r), np.percentile(shuffle_r, 95)
    
    return corr_stats, shuffle_stats, jobNo

#%%

def place_cell_stability(signaltracking, firstOccupancy, secondOccupancy, 
                         bin_edges, nFramesSession):
        
    """
    Created on Thurs Oct 06 14:17:29 2022

    @author: torstsl

    place_cell_stability:

        Helper function for classification of place cells, that calculates the 
        stability of the cell by splitting the data in first and second halves,
        before calculating the correlation between the two halves.
            
        OUTPUT:
            correlation_r (float):  R-value from a Pearson correlation of the 
                                    flattened ratemaps.
                            
        INPUT:
            signaltracking (df):    DataFrame with signal and headpositions.
            firstOccupancy:         Occupancy from opexebo for the first and        
            secondOccupancy:        second halves of the session
            bin_edges:              Bin edges from the occupancy map (opexebo)
            nFramesSession (imt):   Number of frames/datapoints in this session.

    """
                                            
    # Split trial in halves, calculate ratemaps per half
    firstIdx   = np.where(signaltracking.index < nFramesSession/2)
    secondIdx  = np.where(signaltracking.index >= nFramesSession/2)
    firstHalf  = signaltracking.iloc[firstIdx]
    secondHalf = signaltracking.iloc[secondIdx]
    
    firstHalf_tuningmap  = calc_tuningmap(firstOccupancy,bin_edges[0],bin_edges[1],firstHalf,2.5)['tuningmap']
    firstRatemap_smooth  = gaussian_filter(firstHalf_tuningmap, 1.5)
    firstRatemap_smooth  = np.ma.MaskedArray(firstRatemap_smooth, firstHalf_tuningmap.mask)
                
    secondHalf_tuningmap = calc_tuningmap(secondOccupancy,bin_edges[0],bin_edges[1],secondHalf,2.5)['tuningmap']
    secondRatemap_smooth = gaussian_filter(secondHalf_tuningmap, 1.5)
    secondRatemap_smooth = np.ma.MaskedArray(secondRatemap_smooth, secondHalf_tuningmap.mask)
    
    """
    # Plot the ratemap with the halves, testing purpose
    
    fig, (ax2,ax3) = plt.subplots(1,2)
    im2 = ax2.imshow(firstRatemap_smooth)  
    ax2.set_title('First half')
    im3 = ax3.imshow(secondRatemap_smooth)  
    ax3.set_title('Second half')
    fig.suptitle('Session '+str(sessionNo)+' Cell '+str(cellNo)+ ' r='+str(round(r,2)))
    plt.tight_layout()
    """
        
    # Get stability (correlate halves) 
    correlation_r = pearsonr(firstRatemap_smooth.flatten(), secondRatemap_smooth.flatten())[0]
    
    return correlation_r

#%%

def place_cell_classifier(dfNAT, dfCell, nCells, SNR_Cells, 
                          occupancy, firstOccupancy, secondOccupancy, bin_edges, 
                          nFramesSession, param_dict, cellNo):

    """
    Created on Thurs Oct 06 13:09:01 2022

    @author: torstsl

    place_cell_classifier:

        This function is used to classify place cells. In short, it:
            - Filters out cells with sufficient SNR and number of events (deconvolved)
            - Calculates spatial information for a cell
            - Calculates stability of the cell (splitting the session in halves
                                                and correlate the two)
            - Finds place field(s) in the cells' ratemaps
            - Calls in a shuffling for each cell that calculates SI and stability
            
        OUTPUT:
            jobNo (int):            Job number to keep track of things when paralleled
            SI (float):             Spatial information for the cell
            corr_r (float):         Stabiliuty (correlation) for the cell
            placefields (list):     Place fields for this cell
            shuffleSI (arr):        SI for nShuffles (1xnShuffles)
            shuffleCorr (arr):      Stability (correlation) for nShuffles (1xnShuffles)
            candidatePC (int):      NATEX ID for cell with sufficient signal to be analysed
                         
        INPUT:
            dfNAT (df):             DataFrame with all tracking and cell information.
            dfCell (df):            DataFrame with headpos and signal for one cell
            nCells (int):           Total number of cells
            SNR_Cells (arr):        SNR for all cells
            occupancy (arr):        Occupancy from opexebo for the cell,
            firstOccupancy (arr):   the first half of the session,
            secondOccupancy (arr):  and the second half of the session
            bin_edges (list):       Bin edges from the occupancy map (opexebo)
            nFramesSession (int):   Number of frames/datapoints for this session
            param_dict (dict):      Dictionary of the following parameters:
                .nShuffle (int)         Number of times to perform shuffling
                .minShuffleInt (int)    Minimum number for which the cell's signal data is shufffled
                .speedThreshold (float) Minimum speed (cm/s) at time of cell activity to be concidered "true" activity
                .minEvents (int)        Minumim number of events to be analysed 
                .minPeakField (float)   Minimum "firing rate" (Hz) to be accepted as a local field peak
            cellNo (int):           Current cell being analysed (NATEX ID), an iterable

    """
    # Initiate variables      
    signalShuffle = np.full([nFramesSession,param_dict['nShuffle']],np.nan)
    shuffleSI     = np.full([param_dict['nShuffle']],np.nan)
    shuffleCorr   = np.full([param_dict['nShuffle']],np.nan)
    
    jobNo = cellNo-1
    
    # Only proceed if sufficient SNR
    if (cellNo in SNR_Cells) == True:
        signaltracking = get_signaltracking(dfNAT, cellNo, signal='deconvolved', speed_threshold = param_dict['speedThreshold'])
              
        # Only proceed if sufficient events
        if len(signaltracking) >= param_dict['minEvents']: 
            
            candidatePC = cellNo
            notCandidate = False
            
            # Add signal to cell dataframe, filter out events above speed threshold
            dfCell['Signal'] = dfNAT['Deconvolved, N'+str(cellNo)]
            dfCell['Signal'].loc[(dfNAT['Body_speed'])<=param_dict['speedThreshold']] = 0 
            
            # Could get it directly from session_dict['Ratemaps'], but it has to be provided into the function
            tuningmap_dict = calc_tuningmap(occupancy,bin_edges[0],bin_edges[1],signaltracking,2.5)
            ratemap_smooth = gaussian_filter(tuningmap_dict['tuningmap'], 1.5)
            ratemap_smooth = np.ma.MaskedArray(ratemap_smooth, tuningmap_dict['tuningmap'].mask)
                            
            # Get spatial information (SI), place field(s) and stability for this cell              
            SI = op.analysis.rate_map_stats(ratemap_smooth, occupancy)['spatial_information_content']
            placefields = op.analysis.place_field(ratemap_smooth, search_method='sep', min_peak=param_dict['minPeakField'])[0] 
            corr_r = place_cell_stability(signaltracking, firstOccupancy, secondOccupancy, bin_edges, nFramesSession)
                                        
            # Prepare for shuffling: Calculate the shuffled signal beforehand                                                                   
            dfShuffle = dfCell.copy()
            
            iShuffle = np.random.randint(param_dict['minShufflingInt'],nFramesSession-param_dict['minShufflingInt'],param_dict['nShuffle']) # Random shift
            signal = np.array(dfCell['Signal'])
                       
            for ii in range(len(iShuffle)): signalShuffle[:,ii] = np.roll(signal,iShuffle[ii]) # Shuffle event train
           
            # Perform the shuffling for this cell, get the SI and stability per shuffle 
            for shuffleNo in range(0, param_dict['nShuffle']):
                shuffleSI[shuffleNo], shuffleCorr[shuffleNo] = place_cell_shuffle(dfShuffle, signalShuffle, 
                                                                                  occupancy, firstOccupancy, secondOccupancy, 
                                                                                  bin_edges,  nFramesSession, shuffleNo)
                
        elif len(signaltracking) < param_dict['minEvents']:      
            notCandidate = True
            
    elif (cellNo in SNR_Cells) == False:
        notCandidate = True
    
    # Make "fill" values for the output in case the cell is not of sufficient quality so that there are no local errors    
    if notCandidate == True:
        SI, corr_r, placefields, candidatePC = np.nan, np.nan, np.nan, np.nan         
                
    return jobNo, SI, corr_r, placefields, shuffleSI, shuffleCorr, candidatePC

#%%

def place_cell_shuffle(dfShuffle, signalShuffle, occupancy, firstOccupancy, secondOccupancy, bin_edges, nFramesSession, shuffleNo):
           
    """
    Created on Thurs Oct 06 14:33:43 2022

    @author: torstsl

    place_cell_shuffle:

        Helper function for classification of place cells, does shuffling of 
        the cell signal and calculates: 
            - The SI for this shuffle
            - Stability (correlation of first and second halves) for this shuffle
            
        OUTPUT:
            shuffleSI (float):  The spatial information content for this shuffle
            shuffleCorr (float):The R from a Pearson correlation from ratemaps 
                                for the first and second halves of this session
                            
        INPUT:
            dfShuffle (df):         DataFrame with signal and headpositions
            signalShuffle (arr):    Array of shuffled signal trains
            occupancy:              Occupancy from opexebo for the entire session, 
            firstOccupancy:         first,        
            secondOccupancy:        and second halves of the session
            bin_edges:              Bin edges from the occupancy map (opexebo)
            nFramesSession (imt):   Number of frames/datapoints in this session
            shuffleNo (int):        Number of current shuffle, the iterable

    """
    
    # Calculate ratemaps for shuffled distributions
    dfShuffle.Signal = signalShuffle[:,shuffleNo]
                       
    shuffle_tuningmap_dict = calc_tuningmap(occupancy,bin_edges[0],bin_edges[1],dfShuffle,2.5)
    shuffle_ratemap_smooth = gaussian_filter(shuffle_tuningmap_dict['tuningmap'], 1.5)
    shuffle_ratemap_smooth = np.ma.MaskedArray(shuffle_ratemap_smooth, shuffle_tuningmap_dict['tuningmap'].mask)
    
    # Spatial information for this shuffle
    shuffleSI = op.analysis.rate_map_stats(shuffle_ratemap_smooth, occupancy)['spatial_information_content'] 

    # Calculate stability for this shuffle
    shuffleCorr = place_cell_stability(dfShuffle, firstOccupancy, secondOccupancy, bin_edges, nFramesSession)
    
    return shuffleSI, shuffleCorr

#%% Define helper functions to do Bayesian decoding

# Calculate the activity matrix
def calc_activityMatrix(testMatrix, binFrame):
    
    """
    For Bayesian decoder: 
        Bin the data from the data into an activity matrix.
        Uses the sum of the frames binned (not the mean).
    
    INPUT:
        testMatrix: Activity fir all cells for all timestamps
        binFrame:   The binning to be performed on the testMatrix. Number of
                    frames to put in each bin.
                    
    OUTPUT:
        activityMatrix: The binned activity from the test matrix, to be used 
                        to test the decoding from the train matrix.
                        NxM array: N = number of time bins; M = number of cells
    """
    
    slices = np.linspace(0, testMatrix.shape[0], int(testMatrix.shape[0]/binFrame)+1, True).astype(int)

    activityMatrix = np.add.reduceat(np.array(testMatrix), slices[:-1], axis = 0)

    return activityMatrix

# Do the decoding using a Poisson noice model
def poisson_model(trainData, activityMatrix):
    
    """
    
    Noice model for Bayesian decoding of position based on a Poisson distribution.
    
    Probability = (rate**n) * (e** (-rate))
    
        where:    rate = the acitivty from all rate maps for all cells in given 
                         bin (i.j) from the training data
                  n = the activity vector (from the matrix) for all cells in 
                      that timebin from the test data 
                  e = eulers number (np.e)
                  
        The denominator (factorial of rate) is omitted as the logarithmic 
        likelihood is computed later on. This bypasses the need of a interger 
        rate, necessary since the rate is the sum of deconvolved activity for 
        a given timebin.
    
    Parameters
    ----------
    trainData : 3D np.array
        Array of data from training. Ratemaps for each cell. nCells x nBins x nBins
    activityMatrix : 2D np.array
        Array of with row = timebins and column = cell activity in that timebin.
        Rowwise vectors of the summed activity for all cells at given timebins.
        From the test data.

    Returns
    -------
    likelihoodMatrix : 3D np.array
        Array for all timebins of size nBins x nBins, where the values indicate 
        the decoded position on the test data from the predictions made on the 
        train data. 

    """
    
    nBins = trainData.shape[1]
    
    likelihoodMatrix = np.zeros([activityMatrix.shape[0], nBins, nBins])
    
    for timeBin in range(activityMatrix.shape[0]):
        n = activityMatrix[timeBin] # Vector of activity for all cells in this time bin
        
        for i in range(nBins):
            for j in range(nBins):
                rate = trainData[:, i, j] # The rate in the first bin in all the cells' ratemaps             
                
                #Poisson model            
                num = (rate ** n) * (np.e ** -rate)       
                likelihood = np.nanprod(num)
                likelihoodMatrix[timeBin][i][j] = likelihood
                
    return likelihoodMatrix   

def poisson_model_vector(training_data, activity_matrix):
    
    """    Vectorised model for Bayesian decoding of position based on a Poisson distribution.    
    
        Probability = (rate**n) * (e** (-rate))        
        where:    
            rate = the acitivty from all rate maps for all cells in given                          
                    (i.j) from the training data                  
            n = the activity vector (from the matrix) for all cells in                  
                    that timebin from the test data                   
            e = eulers number (np.e)        
        The denominator (factorial of rate) is omitted as the logarithmic         
        likelihood is computed later on. This bypasses the need of a interger         
        rate, necessary since the rate is the sum of deconvolved activity for         
        a given timebin.    
        
        Parameters    
        ----------    
        training_data : 3D np.array        
            Array of data from training. Ratemaps for each cell. nCells x nBins x nBins    
            
        activity_matrix : 2D np.array        
            Array of with row = timebins and column = cell activity in that timebin.        
            Rowwise vectors of the summed activity for all cells at given timebins.        
            From the test data.    
            
        Returns    
        -------    
        likelihood_matrix : 3D np.array        
            Array for all timebins of size nBins x nBins, where the values indicate         
            the decoded position on the test data from the predictions made on the train data.       
           
    """
  
    tn = activity_matrix.shape[0]
    z, x, y = training_data.shape
    likelihood_matrix = np.zeros((tn, x, y))
        
    def tile3d(n, x, y):
        """        
        Given a 1D vector, repeat it multiple times to generate a 3D vector of the shape (n.size, x, y), where each
        element `out[:, j, k]` is identical to the input `n`. THis requires Fortran ordering, rather than C ordering
        """
        return np.tile(n, x*y).reshape((n.size, x, y), order="F")
        
    for tt, n in enumerate(activity_matrix):
        full_scale_n = tile3d(n, x, y)
        val = np.power(training_data, full_scale_n) * np.exp(-1 * training_data)
        likelihood_matrix[tt] = np.nanprod(val, axis=0)
            
    return likelihood_matrix 
     
def poisson_model_lg(trainData, activityMatrix): 
    
    nBins = trainData.shape[1]
    
    likelihoodMatrix = np.zeros([activityMatrix.shape[0], nBins, nBins])
    
    for timeBin in range(activityMatrix.shape[0]):
        n = activityMatrix[timeBin] # Vector of activity for all cells in this time bin
        
        for i in range(nBins):
            for j in range(nBins):
                rate = trainData[:, i, j] # The rate in the first bin in all the cells' ratemaps             
                
                #Poisson model   
                num = n * np.log(rate) - rate
                likelihood = np.nansum(num)
                likelihoodMatrix[timeBin][i][j] = likelihood
                
    return likelihoodMatrix  

def poisson_model_lg_vector(training_data, activity_matrix):
    
    """    
    Vectorised model for Bayesian decoding of position based on a Poisson distribution.  
    
    Probability = n * np.log(rate) - rate   
    
        where:    
            rate = the acitivty from all rate maps for all cells in given                          
                    (i.j) from the training data                  
            n = the activity vector (from the matrix) for all cells in                  
                    that timebin from the test data                   
       
        The equation is the log off the Poisson distribution. The denominator 
        (factorial of rate) is omitted as it is the logarithm of a constant.         
        This bypasses the need of a interger rate, necessary since the rate is 
        the sum of deconvolved activity for a given timebin.    
        
        Compared to the non-logaritmic metod, this uses the sum of the results
        across cells rather than the product.
        
        Parameters    
        ----------    
        training_data : 3D np.array        
            Array of data from training. Ratemaps for each cell. nCells x nBins x nBins    
            
        activity_matrix : 2D np.array        
            Array of with row = timebins and column = cell activity in that timebin.        
            Rowwise vectors of the summed activity for all cells at given timebins.        
            From the test data.    
            
        Returns    
        -------    
        likelihood_matrix : 3D np.array        
            Array for all timebins of size nBins x nBins, where the values indicate         
            the decoded position on the test data from the predictions made on the train data.       
           
        """
  
    tn = activity_matrix.shape[0]
    z, x, y = training_data.shape
    likelihood_matrix = np.zeros((tn, x, y))
        
    def tile3d(n, x, y):
        """        
        Given a 1D vector, repeat it multiple times to generate a 3D vector of the shape (n.size, x, y), where each
        element `out[:, j, k]` is identical to the input `n`. THis requires Fortran ordering, rather than C ordering
        """
        return np.tile(n, x*y).reshape((n.size, x, y), order="F")
        
    for tt, n in enumerate(activity_matrix):
        full_scale_n = tile3d(n, x, y)
        val = full_scale_n * np.log(training_data) - training_data
        likelihood_matrix[tt] = np.nansum(val, axis=0)

    return likelihood_matrix 
 
# Calculate the true and decoded positions from the decoder, and the decoding error
def calc_decodingPos(dfHD, activityMatrix, likelihoodMatrix, binFrame):
    """
    
    Calculate the true position and the decoded position from Bayesian decoding.
    Also, calculate the decoding error based on this.

    Parameters
    ----------
    dfHD : pd.DataFrame
        The dfNAT from the test session slices for head directions (dfTest[['Head_X', 'Head_Y']]). 
    activityMatrix : 2D np.array
        Array of with row = timebins and column = cell activity in that timebin.
        Rowwise vectors of the summed activity for all cells at given timebins.
        From the test data.
    likelihoodMatrix : 3D np.array
         Array for all timebins of size nBins x nBins, where the values indicate 
         the decoded position on the test data from the predictions made on the 
         train data. From poisson_model_lg_vector
    binFrame : int
        Number of frames to put into each time bin.

    Returns
    -------
    truePos : 2D np.array
        Coordinates (x, y) of true position of the animal for given time bin.
    decodingPos : 2D np.array
        Coordinates (x, y) of decoded position of the animal for given time bin.
    decodingError : 1D np.array
        Array of eucledean distances from the true position to the decoded position (in cm)

    """
    
    nBins = likelihoodMatrix.shape[1]
    
    # Calculate the true position
    if binFrame%2 == 0: # binFrame is an even number
        slicer = np.arange(len(dfHD)).reshape((len(dfHD)//binFrame), binFrame)[:, int(binFrame/2-1):int(binFrame/2+1)]
        midPos = dfHD.to_numpy()[slicer].reshape(slicer.shape[0]*2,2)
        truePos = np.add.reduceat(midPos, np.arange(0, len(midPos),2)) / 2 # The mean of the two middle poinst per data bin
        
    elif binFrame%2 > 0: # binFrame is an odd number     
        truePos = dfHD[binFrame//2::binFrame].to_numpy() # The middle point per data bin
    
    # Calculate the headpos for the test session, set bin edges from these
    headpos = dfHD.to_numpy()
    x_edges = np.linspace(headpos[:,0].min(), headpos[:,0].max(), nBins+1)
    y_edges = np.linspace(headpos[:,1].min(), headpos[:,1].max(), nBins+1)
    
    decodingPos = np.full([activityMatrix.shape[0], 2], np.nan)
    x_shift, y_shift = np.diff(x_edges).mean()/2, np.diff(y_edges).mean()/2 # Correction to plot the middle point of the bin
    
    for timeBinNo in range(activityMatrix.shape[0]):
    
        # Get the decoded bin position into the tracking dimentions for this time bin
        y_ind, x_ind = np.unravel_index(np.argmax(likelihoodMatrix[timeBinNo]), (32,32)) # Row, column
        x, y = x_edges[x_ind] + x_shift, y_edges[::-1][y_ind] + y_shift # The x, y coordinate of the middle point in the decoded position bin
        decodingPos[timeBinNo] = x, y
    
    # The distance from the true position to the decoded position 
    decodingError = np.linalg.norm(truePos - decodingPos, axis=-1)
    
    return truePos, decodingPos, decodingError 

def run_decoder(trainData_dict, activityMatrix_dict, dfHD, placecell_dict, anatomicalDist, speedIdx, parameters, cellNo):
    
    jobNo = cellNo
    
    nPC, sessionNo, binFrame, nShuffles, minDist = parameters['nPC'], parameters['sessionNo'], parameters['binFrame'], parameters['nShuffles'], parameters['minDist']

    # Get a random cell each time this loop runs
    refPC = np.random.choice(nPC) # Python index for only session PCs
    refPCNAT = placecell_dict['NAT'+str(sessionNo)][0][refPC] # NATEX index (all cells)
    
    distVector = anatomicalDist[refPCNAT-1, placecell_dict['NAT'+str(sessionNo)][0]-1]
    distBool = distVector < minDist
    nNeighbours = np.nansum(distBool)-1 # Minus the reference cell itself
    cohort = int(nNeighbours + 1) # Number of cells used to decode
    
    trainData = trainData_dict['NAT'+str(sessionNo)][placecell_dict['NAT'+str(sessionNo)][0][distBool]]
    
    activityMatrix = activityMatrix_dict['NAT'+str(sessionNo)][:, placecell_dict['NAT'+str(sessionNo)][0][distBool]]
    
    # Do decoding for the true distribution 
    likelihoodMatrix = poisson_model_lg_vector(trainData, activityMatrix)
    
    truePos, decodingPos, decodingError = calc_decodingPos(dfHD, activityMatrix, likelihoodMatrix, binFrame)
    
    # Decoding accuracy with speed filter   
    res_medians = np.median(decodingError[speedIdx])
    res_means = np.nanmean(decodingError[speedIdx])
    
    res_mediansShuffle = np.full([nShuffles], np.nan)
    res_meansShuffle = np.full([nShuffles], np.nan)
    
    for shuffleNo in range(nShuffles):
        shuffleCells = np.zeros(nNeighbours+1, dtype = int)
        shuffleCells[0] = refPC
        shuffleCells[1:] = np.random.choice(nPC, nNeighbours, replace = False)
        
        trainData = trainData_dict['NAT'+str(sessionNo)][placecell_dict['NAT'+str(sessionNo)][0][shuffleCells]]
        activityMatrix = activityMatrix_dict['NAT'+str(sessionNo)][:, placecell_dict['NAT'+str(sessionNo)][0][shuffleCells]]
        
        # Do decoding for the true distribution 
        likelihoodMatrix = poisson_model_lg_vector(trainData, activityMatrix)
        
        truePos, decodingPos, decodingError = calc_decodingPos(dfHD, activityMatrix, likelihoodMatrix, binFrame)
        
        # Decoding accuracy with speed filter        
        res_mediansShuffle[shuffleNo] = np.median(decodingError[speedIdx])
        res_meansShuffle[shuffleNo] = np.nanmean(decodingError[speedIdx])
            
    return jobNo, res_medians, res_means, res_mediansShuffle, res_meansShuffle, cohort
#%%

def plot_decodingError(decodingErrorSpeed, titleString, **kwargs):
    
    """
    Plot the decoding error as a histogram with the mean and median noted.
    
    Inputs: decodingError (np.array): From decoding. Data that will be plotted
            titleString (str): A string that will become the figure title
    
            kwargs: save (bool): If True, the figure is saves as an .svg in 
                                 the standard folder under name .img. 
                                 Default is False. 
    """
    
    # Plot the distribution of the decoding error
    histbins = 50 
    contrast = sns.color_palette('OrRd', 256).as_hex()
    chanceLevel = 52
    
    fig, ax = plt.subplots(figsize = (5,5))
    sns.histplot(data = decodingErrorSpeed, bins = histbins, ax = ax, 
                 color = 'gray', fill = True, kde = False, edgecolor='gray')
    ax.vlines(np.median(decodingErrorSpeed), 0, np.histogram(decodingErrorSpeed,histbins)[0].max(), 
              color = contrast[210], label = 'Median: ' + str(round(np.median(decodingErrorSpeed),1)) + ' cm')
    ax.vlines(np.nanmean(decodingErrorSpeed), 0, np.histogram(decodingErrorSpeed,histbins)[0].max(), 
              color = contrast[150], label = 'Mean: ' + str(round(np.nanmean(decodingErrorSpeed),1)) + ' cm')
    ax.vlines(chanceLevel, 0,np.histogram(decodingErrorSpeed,histbins)[0].max(),
              color = 'darkgray', linestyles = 'dashed', label = 'Chance')
    ax.legend(frameon = False)
    ax.set_xlim(0)
    ax.set_title(titleString)
    ax.set_xlabel('Distance (cm)')
    ax.set_ylabel('Count (time bins)')
    ax.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout() 
    
    if kwargs.get('save', True):
        plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg')     
    
#%%

def plot_decodingAccuracy(truePos, decodingPos, speedIdx, start, stop, timeBin, titleString, **kwargs):
    
    """
    Plot the decoding accuracy from the true position and the decoded position
    as two separate lines for X and Y, respectively. 
    
    Inputs: truePos (np.array): From decoding. The true positions from test data.
            decodingPos (np.array): From decoding. Decoded position.
            speedIdx (np.array): Array of booleans. Time bins at which the speed
                                is above given threshold.
            start (int): At what time bin to start plotting
            stop (int): At what time bin to stop plotting
            timeBin (float): The time in seconds for one time bin (nBins*frameRate).                    
            titleString (str): A string that will become the figure title
    
            kwargs: save (bool): If True, the figure is saves as an .svg in 
                                 the standard folder under name .img. 
                                 Default is False. 
    """
    
    # Plot the decoding accuracy of x and y separately
    xlabels = np.linspace(start*timeBin, stop*timeBin,7, dtype = int)    
    palette = sns.color_palette("viridis", 256).as_hex()
    contrast = sns.color_palette('OrRd', 256).as_hex()
    
    fig, ax = plt.subplots(2,1, figsize = (10,4), sharex = True, sharey = True)
    
    ax[0].set_title(titleString)
    sns.lineplot(data = truePos[speedIdx][start:stop,0], ax = ax[0], color = palette[100], label = 'True')
    sns.lineplot(data = decodingPos[speedIdx][start:stop,0], ax = ax[0], color = contrast[190], label = 'Decoded')
    ax[0].set_ylabel('X', rotation = 'horizontal')
    ax[0].set_xlim([0, np.diff([start,stop])])
    ax[0].legend(ncol = 2, loc='best', bbox_to_anchor=(0.5, 0.75, 0.5, 0.5), frameon = False)
    ax[0].spines[['top', 'right']].set_visible(False)
    
    sns.lineplot(data = truePos[speedIdx][start:stop,1], ax = ax[1], color = palette[100], label = 'True', legend = False)
    sns.lineplot(data = decodingPos[speedIdx][start:stop,1], ax = ax[1], color = contrast[190], label = 'Decoded', legend = False)
    ax[1].set_ylabel('Y', rotation = 'horizontal')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_xlim([0, np.diff([start,stop])])
    ax[1].spines[['top', 'right']].set_visible(False)
    ax[1].xaxis.set_ticks(xlabels/timeBin-start)
    ax[1].set_xticklabels(xlabels)
    
    plt.tight_layout()
    
    if kwargs.get('save', True):
        plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg')        
