#%%
"""
Created on Tue Aug 23 08:06:09 2022

@author: torstsl

-------------------------------------------------------------------------------
DESCRIPTION

This script loads session_dicts from a recording and gets the corresponding
ratemaps for all the cells. It singles out which cells are place cells (to be
further implemented), and does correlation analyses on them.

First, it calculated a 2D correlation of A to B and of A to A'. This can be 
plotted on a cell-by-cell basis with A-B-A' ratemaps and correlaation matrices. 
Secondly, it can plot the distribution of the flattened correlation between 
all ratemaps (hence, Pearson r) for place cells and non-place cells, for 
AB and AA', and plot it with 95 % CI error bars. This may show if there is 
remapping, and if there are differences in the PC and non-PC population.

-------------------------------------------------------------------------------
                                                  
op.analysis.place_field(firing_map, kwargs)
op.analysis.rate_map_stats(rate_map, time_map)
op.analysis.rate_map_coherence(rate_map_raw)

===============================================================================
||                                                                           ||
||  Use scipy.signal.correlate2d to get the correlation matrix (Nx2-1XMx2-1) ||
||                                                                           ||
||  Use either np.corrcoef(A.flatten(),B.flatten())[0,1] or                  ||
||  r = op.general.spatial_cross_correlation(arr_0, arr_1, kwargs)[0] or     ||
||  r,pval = scipy.stats.pearsonr(A.flatten(),B.flatten())                   ||
||  to get the pearson r-value for the correlation between the two matrices  ||
||                                                                           || 
===============================================================================

-------------------------------------------------------------------------------

The script is made into a function. In addition, the script "centre_of_mass" is
incorporated into this function. There is an addition, in which the correlations
between rate maps are compared to that of a shuffled distribution. It is always
the ratemap in A' that is shuffled, and comparisons are made from A and B to 
the shuffled A'. This is also the case for calculations on centre of mass and 
the eucledian distance between centres of mass.

"""
import numpy as np
import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import concurrent.futures as cf
from tqdm import tqdm
from scipy.signal import correlate2d
from scipy.stats import pearsonr 
from scipy.ndimage import center_of_mass
# from src.loading_data import load_session_dict
from src.cell_activity_analysis import calc_session_ratemaps, shuffle_ratemap

#%% For paralellising the shuffling procedure
def ratemap_shuffler(ratemaps, dfNAT, nShuffles, cellNo):
    
    """
    A function for utilizing the shuffle_ratemap function in a paralell loop 
    
    """
    
    jobNo = cellNo - 1
    
    resultsA = np.full([nShuffles], np.nan)
    resultsB = np.full([nShuffles], np.nan)
    resultsCOM = np.full([nShuffles, 2], np.nan)
    
    # Keep the cell in A
    rmapA = ratemaps['dfNAT0']['N'+str(cellNo)]
    rmapB = ratemaps['dfNAT1']['N'+str(cellNo)]
    
    # Shuffle the cell in A'
    rmap_shuffle = np.full([nShuffles, rmapA.shape[0], rmapA.shape[1]], np.nan)
    
    for shuffleNo in range(nShuffles):
        rmap_shuffle[shuffleNo] = shuffle_ratemap(dfNAT, cellNo)
        resultsA[shuffleNo] = pearsonr(rmapA.flatten(), rmap_shuffle[shuffleNo].flatten())[0]
        resultsB[shuffleNo] = pearsonr(rmapB.flatten(), rmap_shuffle[shuffleNo].flatten())[0]
        
        resultsCOM[shuffleNo] = center_of_mass(rmap_shuffle[shuffleNo])
        
    return jobNo, resultsA, resultsB, resultsCOM

#%%

def shuffle_ratemap_by_idx(ratemaps, placecell_dict, nShuffles, binning):
    
    rA = np.full([nShuffles, len(placecell_dict['NAT0'][0])], np.nan)
    rB = np.full([nShuffles, len(placecell_dict['NAT0'][0])], np.nan)
    
    com_distA = np.full([nShuffles, len(placecell_dict['NAT0'][0])], np.nan)
    com_distB = np.full([nShuffles, len(placecell_dict['NAT0'][0])], np.nan)

    for cellNo in range(len(placecell_dict['NAT0'][0])):
       PC = placecell_dict['NAT0'][0][cellNo]
       test_map = ratemaps['dfNAT0']['N'+str(PC)]
       
       test_com = np.array(center_of_mass(test_map))
       
       for shuffleNo in range(nShuffles):
           
           # Remove self
           pc_dict = placecell_dict['NAT0'][0].copy()
           pc_dict = np.delete(pc_dict, cellNo)
           
           idx = np.random.randint(len(pc_dict))
           sPC = pc_dict[idx]
           shuffle_mapA = ratemaps['dfNAT0']['N'+str(sPC)]
       
           rA[shuffleNo, cellNo],_ = pearsonr(test_map.flatten(), shuffle_mapA.flatten())
           
           shuffle_comA = np.array(center_of_mass(shuffle_mapA))
           com_distA[shuffleNo, cellNo] = np.linalg.norm(test_com-shuffle_comA) * binning
           # ((test_com[0]-shuffle_com[0])**2 + (test_com[1]-shuffle_com[1])**2)**0.5
          
           shuffle_mapB = ratemaps['dfNAT1']['N'+str(sPC)]
           rB[shuffleNo, cellNo],_ = pearsonr(test_map.flatten(), shuffle_mapB.flatten())
           
           shuffle_comB = np.array(center_of_mass(shuffle_mapB))
           com_distB[shuffleNo, cellNo] = np.linalg.norm(test_com-shuffle_comB) * binning

            
    return rA, com_distA, rB, com_distB

#%% Define the analyses to be performed
def func_correlate_ratemaps(session_dict, **kwargs):
    
    plotter = kwargs.get('plotter', False)
    
    # Load data, generate ratemaps (if needed), initiate variables
    
    binning = 2.5 # cm/bin in ratemap
    
    ratemaps = session_dict['Ratemaps']
    nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)
    
    placecell_dict = session_dict['Placecell']
    
    # Define place cells    
    PC_A = np.full([nCells], False)
    PC_A[placecell_dict['NAT0'][0]-1] = True
    
    PC_B = np.full([nCells], False)
    PC_B[placecell_dict['NAT1'][0]-1] = True
    
    # Correlate all cells to one another
    corrAA2 = np.full([nCells], np.nan)
    corrAB = np.full([nCells], np.nan)
    
    com = {'NAT0': np.full([nCells, 2], np.nan),
           'NAT1': np.full([nCells, 2], np.nan),
           'NAT2': np.full([nCells, 2], np.nan)}
    
    for cellNo in range(1,nCells+1):
        
        A = ratemaps['dfNAT0']['N'+str(cellNo)]
        B = ratemaps['dfNAT1']['N'+str(cellNo)]
        A2 = ratemaps['dfNAT2']['N'+str(cellNo)]
        
        corrAA2[cellNo-1] = pearsonr(A.flatten(),A2.flatten())[0]  
        corrAB[cellNo-1] = pearsonr(A.flatten(),B.flatten())[0]  
   
        com['NAT0'][cellNo-1,:], com['NAT1'][cellNo-1,:], com['NAT2'][cellNo-1,:] = center_of_mass(A), center_of_mass(B),  center_of_mass(A2)
    
    # Calculate the distances in centre of mass and prepare the data
    com_distAA2 = np.linalg.norm(com['NAT0'] - com['NAT2'], axis = -1) * binning
    com_distAB = np.linalg.norm(com['NAT0'] - com['NAT1'], axis = -1) * binning
    
    df1 = pd.DataFrame({'Distance': com_distAA2[PC_A], 'Box': ['AA']*np.sum(PC_A)})
    df2 = pd.DataFrame({'Distance': com_distAB[PC_A], 'Box': ['AB']*np.sum(PC_A)})
    df3 = pd.DataFrame({'Distance': com_distAB[PC_B], 'Box': ['BA']*np.sum(PC_B)})
    dfCOMDist = pd.concat([df1, df2, df3])
    dfCOMDist.reset_index(drop = True, inplace = True)
    
    # Get place cells in A compared to A'
    df1 = pd.DataFrame({'Correlation': corrAA2[PC_A],'Type': ['Place cell']*np.sum(PC_A), 'Box': ['AA\'']*np.sum(PC_A)})    
    df2 = pd.DataFrame({'Correlation': corrAA2[~PC_A],'Type': ['Non-place cell']*(nCells - np.sum(PC_A)), 'Box': ['AA\'']*(nCells - np.sum(PC_A))})    
    dfAA = pd.concat([df2, df1])    
    dfAA.reset_index(drop = True, inplace = True)
    
    # Get place cells in A compared to B
    df1 = pd.DataFrame({'Correlation': corrAB[PC_A],'Type': ['Place cell']*np.sum(PC_A), 'Box': ['AB']*np.sum(PC_A)})    
    df2 = pd.DataFrame({'Correlation': corrAB[~PC_A],'Type': ['Non-place cell']*(nCells - np.sum(PC_A)), 'Box': ['AB']*(nCells - np.sum(PC_A))})    
    dfAB = pd.concat([df2, df1])    
    dfAB.reset_index(drop = True, inplace = True)
        
    # Get place cells in B compared to A
    df1 = pd.DataFrame({'Correlation': corrAB[PC_B],'Type': ['Place cell']*np.sum(PC_B), 'Box': ['BA']*np.sum(PC_B)})    
    df2 = pd.DataFrame({'Correlation': corrAB[~PC_B],'Type': ['Non-place cell']*(nCells - np.sum(PC_B)), 'Box': ['BA']*(nCells - np.sum(PC_B))})    
    dfBA = pd.concat([df2, df1])    
    dfBA.reset_index(drop = True, inplace = True)
    
    dfData = pd.concat([dfAA, dfAB, dfBA])
    dfData.reset_index(drop = True, inplace = True)
    
    # Plot the results
    if plotter == True:
        
        fig, ax = plt.subplots(1,2,sharey=True)
        plt.suptitle('Spatial correlation for '+str(nCells)+' cells during remapping')
        # ax[0] = sns.barplot(x='Box',y='Correlation',hue='PC',data=dfCorr,ax=ax[0],estimator=np.nanmean, ci=95, errwidth = 1.5,capsize=0.05,units = 'Correlation',palette='viridis')
        ax[0] = sns.boxplot(x='Box',y='Correlation',hue='Type',data=dfData,ax=ax[0],palette='viridis', dodge = True)
        ax[0].legend([],[],frameon=False)  
        ax[0].spines[['top', 'right']].set_visible(False)
        # ax[0].set_ylim([-0.5,1.1])
        
        ax[1] = sns.violinplot(x='Box',y='Correlation',hue='Type',data=dfData,ax=ax[1],split=False,palette='viridis', inner = 'box')
        ax[1].spines[['top', 'right']].set_visible(False)
        ax[1].legend(ncol = 2, loc ='upper center', bbox_to_anchor = (0., 1.02, 1., .102), frameon=False)  
        
        ax[0].set(xlabel=None,ylabel='Spatial correlation')
        ax[1].set(xlabel=None, ylabel=None)
        
        plt.tight_layout()
        
        # Just the box plot
        fig, ax = plt.subplots(figsize = (5,5))
        ax.set_title('Spatial correlation for '+str(nCells)+' cells during remapping')
        sns.boxplot(x='Box', y='Correlation', hue='Type', data=dfData, ax=ax, palette='viridis', dodge = True, saturation = 0.9)
        ax.legend(ncol = 1, loc ='upper right', bbox_to_anchor = (0., 0.92, 1.2, .102), frameon=False)  
        ax.spines[['top', 'right']].set_visible(False)
        ax.set(xlabel = None, ylabel = 'Spatial correlation')
        
        plt.tight_layout()
    
        # # Check non-place cells with high correaltion between A-A'
        # rBool = corr_AA2 > 0.8
        # testers = np.where((rBool == ~PC_A) & (~PC_A == True))[0] # Python index
        
        # for x in testers:
        #     r1 = session_dict['Ratemaps']['dfNAT0']['N'+str(x+1)] # Convert to NATEX idx
        #     r2 = session_dict['Ratemaps']['dfNAT1']['N'+str(x+1)] # Convert to NATEX idx
        #     r3 = session_dict['Ratemaps']['dfNAT2']['N'+str(x+1)] # Convert to NATEX idx
        #     rs = r1, r2, r3
            
        #     vMin, vMax = np.min(rs), np.max(rs)
            
        #     fig, ax = plt.subplots(1,3)
        #     plt.suptitle('Cell ' + str(x+1) + ': Pearson R: ' + str(round(pearsonr(r1.flatten(), r3.flatten())[0],2)))
        
        #     for r, s, ax in zip(rs, sessionString, ax.flat): 
        #         im = ax.imshow(r, vmin = vMin, vmax = vMax) 
        #         ax.set_title(s)
        #         ax.set(xlabel=None, ylabel=None, aspect = 'equal')
        #         ax.set_axis_off()
            
        #     fig.subplots_adjust(right=0.8)
        #     cbar_ax = fig.add_axes([0.85, 0.28, 0.05, 0.45])
        #     fig.colorbar(im, cax=cbar_ax, label = 'Deconvolved events (A.U.)')
        
        fig, ax = plt.subplots()
        sns.violinplot(data = dfCOMDist, x = 'Box', y = 'Distance', palette = 'viridis', ax = ax)
        ax.set(xlabel = None, ylabel = 'Distance (cm)', ylim = (0), title = 'Distance in centre of mass for place cells')
        ax.spines[['top', 'right']].set_visible(False)
        
    # Compare the correlation and centre of mass between place cells to that of a shuffled distribution
    nShuffles = 25
    rA, com_distA, rB, com_distB = shuffle_ratemap_by_idx(ratemaps, placecell_dict, nShuffles, binning)
    
    """
    Not in use for now..
    
    # Start parallelisation of analyses in a cell wise matter
    numWorkers = 16
    futures = []
    
    # Parallell for A-A
    nShuffles = 25 
    dfNAT = session_dict['dfNAT']['dfNAT2']
    shuffleCorrsA = np.full([nShuffles, nCells], np.nan)
    shuffleCorrsB = np.full([nShuffles, nCells], np.nan)
    shuffleCOM = np.full([nCells, nShuffles, 2], np.nan)
    
    with cf.ProcessPoolExecutor(max_workers=numWorkers) as pool:
        for cellNo in range(1,nCells+1):
            futures.append(pool.submit(
                ratemap_shuffler,   # The function
                ratemaps,           # Ratemaps
                dfNAT,              # Timestamps, headpos, speed, cell activity
                nShuffles,          # The number of shuffles to perform
                cellNo              # Cell number, the iterable (NATEX idx)
            ))     
    
        for future in tqdm(cf.as_completed(futures), total=nCells):
            jobNo = future.result()[0]
            shuffleCorrsA[:,jobNo] = future.result()[1]
            shuffleCorrsB[:,jobNo] = future.result()[2]
            shuffleCOM[jobNo, :, :] = future.result()[3]
            
    # Prepare the output
    df1 = pd.DataFrame({'Correlation': shuffleCorrsA.flatten(),'Type': ['Shuffle']*(shuffleCorrsA.size), 'Box': ['A-Shuffle']*(shuffleCorrsA.size)})    
    df2 = pd.DataFrame({'Correlation': shuffleCorrsB.flatten(),'Type': ['Shuffle']*(shuffleCorrsB.size), 'Box': ['B-Shuffle']*(shuffleCorrsB.size)})    
    dfShuffle = pd.concat([dfData.copy(), df1, df2])
    
    # Separate the place cells from the non-place cells
    dfS1 = pd.DataFrame({'Correlation': shuffleCorrsA[:,PC_A].flatten(),'Type': ['Shuffle PC']*(shuffleCorrsA[:,PC_A].size), 'Box': ['A-Shuffle']*(shuffleCorrsA[:,PC_A].size)})    
    dfS2 = pd.DataFrame({'Correlation': shuffleCorrsA[:,~PC_A].flatten(),'Type': ['Shuffle nPC']*(shuffleCorrsA[:,~PC_A].size), 'Box': ['A-Shuffle']*(shuffleCorrsA[:,~PC_A].size)})    
    
    dfS3 = pd.DataFrame({'Correlation': shuffleCorrsA[:,PC_B].flatten(),'Type': ['Shuffle PC']*(shuffleCorrsB[:,PC_B].size), 'Box': ['B-Shuffle']*(shuffleCorrsB[:,PC_B].size)})    
    dfS4 = pd.DataFrame({'Correlation': shuffleCorrsB[:,~PC_B].flatten(),'Type': ['Shuffle nPC']*(shuffleCorrsB[:,~PC_B].size), 'Box': ['B-Shuffle']*(shuffleCorrsB[:,~PC_B].size)})    
   
    dfShuffle_split = pd.concat([dfData.copy(), dfS2, dfS1, dfS3, dfS4])
    
    # Add the shuffle data into the centre of mass distances
    expCoorA = np.repeat(com['NAT0'][PC_A],nShuffles, axis = 0)
    expCoorB = np.repeat(com['NAT1'][PC_B],nShuffles, axis = 0)
    shuffleCoorA = shuffleCOM[PC_A,:,:].reshape(np.sum(PC_A)*nShuffles, 2)
    shuffleCoorB = shuffleCOM[PC_B,:,:].reshape(np.sum(PC_B)*nShuffles, 2)
    
    shuffleDistA = np.linalg.norm(expCoorA - shuffleCoorA, axis = -1)
    shuffleDistB = np.linalg.norm(expCoorB - shuffleCoorB, axis = -1)

    dfCOM1 = pd.DataFrame({'Distance': shuffleDistA, 'Box': ['A-Shuffle']*len(shuffleDistA)})
    dfCOM2 = pd.DataFrame({'Distance': shuffleDistB, 'Box': ['B-Shuffle']*len(shuffleDistB)})
    dfCOMDist = pd.concat([dfCOMDist, dfCOM1, dfCOM2])
    
    
    # Plot the results from before, but with the shuffles distribution for comparison
    if plotter == True:
        
        # Just the box plot
        fig, ax = plt.subplots(figsize = (5,5))
        ax.set_title('Spatial correlation for '+str(nCells)+' cells during remapping')
        sns.boxplot(x='Box', y='Correlation', hue='Type', data=dfShuffle, ax=ax, palette='viridis', dodge = True, saturation = 0.9)
        ax.legend(ncol = 1, loc ='upper right', bbox_to_anchor = (0., 0.92, 1.2, .102), frameon=False)  
        ax.spines[['top', 'right']].set_visible(False)
        ax.set(xlabel = None, ylabel = 'Spatial correlation')
        
        plt.tight_layout()
        
        # Just the box plot - split shuffle
        fig, ax = plt.subplots(figsize = (5,5))
        ax.set_title('Spatial correlation for '+str(nCells)+' cells during remapping')
        sns.boxplot(x='Box', y='Correlation', hue='Type', data=dfShuffle_split, ax=ax, palette='viridis', dodge = True, saturation = 0.9)
        ax.legend(ncol = 2, loc ='best', frameon=False)  
        ax.spines[['top', 'right']].set_visible(False)
        ax.set(xlabel = None, ylabel = 'Spatial correlation')
        
        plt.tight_layout() 
        
        fig, ax = plt.subplots()
        sns.violinplot(data = dfCOMDist, x = 'Box', y = 'Distance', palette = 'viridis', ax = ax)
        ax.set(xlabel = None, ylabel = 'Distance (cm)', ylim = (0), title = 'Distance in centre of mass for place cells')
        ax.spines[['top', 'right']].set_visible(False)
        
        plt.tight_layout() 
        
    result_dict = {'dfData': dfData, 'dfShuffle': dfShuffle, 'dfShuffle_split': dfShuffle_split, 'dfCOMDist': dfCOMDist} 
    
    """
    
    dfShuffleA = pd.DataFrame({'Correlation': rA.flatten()})
    dfShuffleA['Type'] = 'Place cell'
    dfShuffleA['Box'] = 'A - Shuffle A'
    
    dfShuffleB = pd.DataFrame({'Correlation': rB.flatten()})
    dfShuffleB['Type'] = 'Place cell'
    dfShuffleB['Box'] = 'A - Shuffle B'
    
    dfShuffle = pd.concat([dfShuffleA, dfShuffleB])
    
    dfShuffleCOMDistA = pd.DataFrame({'Distance': com_distA.flatten()}) 
    dfShuffleCOMDistA['Box'] = 'A - Shuffle A'
    
    dfShuffleCOMDistB = pd.DataFrame({'Distance': com_distB.flatten()}) 
    dfShuffleCOMDistB['Box'] = 'A - Shuffle B'
    
    dfShuffleCOMDist = pd.concat([dfShuffleCOMDistA, dfShuffleCOMDistB])
    
    result_dict = {'dfData': dfData, 'dfCOMDist': dfCOMDist, 'dfShuffle': dfShuffle, 'dfShuffleCOMDist': dfShuffleCOMDist}
    
    return result_dict

#%% Loop over the sessions and store the output

results_folder = r'C:\Users\torstsl\Projects\axon2pmini\results'

with open(results_folder+'/sessions_overview.txt') as f:
    sessions = f.read().splitlines() 
f.close()

if __name__ == "__main__":
    
    correlate_ratemaps_dict = {}  
  
    for session in sessions: 
        
        # Load session_dict
        session_dict = pickle.load(open(session+'\session_dict.pickle','rb'))
        print('Successfully loaded session_dict from: '+str(session))
        
        # Perform the analysis and return the output
        key = session_dict['Animal_ID'] + '-' + session_dict['Date']
        correlate_ratemaps_dict[key] = func_correlate_ratemaps(session_dict, plotter = False)
        
        # Store the output
        with open(results_folder+'/correlate_ratemaps_dict_2.pickle','wb') as handle:
            pickle.dump(correlate_ratemaps_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Successfully saved results_dict in '+results_folder)

    # Plot the results from one session
    # dfSplit = correlate_ratemaps_dict['100867-090922']['dfShuffle_split']
    # dfPlot = pd.concat([dfSplit[dfSplit.Type == 'Place cell'], dfSplit[dfSplit.Type == 'Shuffle PC']]) 
    
    # # Just the box plot - split shuffle
    # fig, ax = plt.subplots(figsize = (4,5))
    # ax.set_title('Spatial correlation for place cells in remapping')
    # sns.boxplot(x='Type', y='Correlation', hue='Box', data=dfPlot, ax=ax, palette='viridis', saturation = 0.75,
    #             dodge = True, fliersize = 2)
    # ax.legend(ncol = 2, loc ='upper right', frameon=False)  
    # ax.spines[['top', 'right']].set_visible(False)
    # ax.set(xlabel = None, ylabel = 'Spatial correlation')
    
    # plt.tight_layout() 
    
    # # Just the box plot - split shuffle
    # fig, ax = plt.subplots(figsize = (4,5))
    # ax.set_title('Spatial correlation for place cells in remapping')
    # sns.violinplot(x='Type', y='Correlation', hue='Box', data=dfPlot, ax=ax, palette='viridis', saturation = 0.75, 
    #                 dodge = True, linewidth = 1.5)
    # ax.legend(ncol = 2, loc ='upper right', frameon=False)  
    # ax.spines[['top', 'right']].set_visible(False)
    # ax.set(xlabel = None, ylabel = 'Spatial correlation')
    
    # plt.tight_layout() 
    
#%% OLD SCRIPT
"""

Here is the old script before I made it into a function

"""

#%% Load data, generate ratemaps (if needed), initiate variables
if __name__ == "__main__":
    
    # session_dict = load_session_dict()
    session_dict = session_dict = pickle.load(open(r'N:\axon2pmini\Recordings\102124\280922/session_dict.pickle','rb'))
    
    if 'Ratemaps' in session_dict.keys():
        ratemaps = session_dict['Ratemaps']
    elif session_dict.get('Ratemaps') == None:
        session_ratemaps = calc_session_ratemaps(session_dict)
    
    nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)
    sessionString = ['A', 'B', 'A\'']
    
    placecell_dict = session_dict['Placecell']
    PCs = np.unique(np.concatenate([placecell_dict['NAT0'][0],placecell_dict['NAT1'][0],placecell_dict['NAT2'][0]]))
    
    palette = sns.color_palette("viridis", 256).as_hex()
    contrast = sns.color_palette('OrRd', 256).as_hex()
    
    #%% Correlate all cells to one another
    corr_AA2 = np.full([nCells], np.nan)
    corr_AB = np.full([nCells], np.nan)
    
    for cellNo in range(1,nCells+1):
        
        A = ratemaps['dfNAT0']['N'+str(cellNo)]
        B = ratemaps['dfNAT1']['N'+str(cellNo)]
        A2 = ratemaps['dfNAT2']['N'+str(cellNo)]
        
        corr_AA2[cellNo-1] = pearsonr(A.flatten(),A2.flatten())[0]  
        corr_AB[cellNo-1] = pearsonr(A.flatten(),B.flatten())[0]  
        
    PC_A = np.full([nCells], False)
    PC_A[placecell_dict['NAT0'][0]-1] = True
    
    df1 = pd.DataFrame({'Correlation': corr_AA2[PC_A],'Type': ['Place cell']*np.sum(PC_A), 'Box': ['AA\'']*np.sum(PC_A)})    
    df2 = pd.DataFrame({'Correlation': corr_AA2[~PC_A],'Type': ['Non-place cell']*(nCells - np.sum(PC_A)), 'Box': ['AA\'']*(nCells - np.sum(PC_A))})    
    dfA = pd.concat([df2, df1])    
    dfA.reset_index(drop = True, inplace = True)
    
    df1 = pd.DataFrame({'Correlation': corr_AB[PC_A],'Type': ['Place cell']*np.sum(PC_A), 'Box': ['AB']*np.sum(PC_A)})    
    df2 = pd.DataFrame({'Correlation': corr_AB[~PC_A],'Type': ['Non-place cell']*(nCells - np.sum(PC_A)), 'Box': ['AB']*(nCells - np.sum(PC_A))})    
    dfB = pd.concat([df2, df1])    
    dfB.reset_index(drop = True, inplace = True)
        
    dfData = pd.concat([dfA, dfB])
    dfData.reset_index(drop = True, inplace = True)
    
    #%% Plot the results
    
    fig, ax = plt.subplots(1,2,sharey=True)
    plt.suptitle('Spatial correlation for '+str(nCells)+' cells during remapping')
    # ax[0] = sns.barplot(x='Box',y='Correlation',hue='PC',data=dfCorr,ax=ax[0],estimator=np.nanmean, ci=95, errwidth = 1.5,capsize=0.05,units = 'Correlation',palette='viridis')
    ax[0] = sns.boxplot(x='Box',y='Correlation',hue='Type',data=dfData,ax=ax[0],palette='viridis', dodge = True)
    ax[0].legend([],[],frameon=False)  
    ax[0].spines[['top', 'right']].set_visible(False)
    # ax[0].set_ylim([-0.5,1.1])
    
    ax[1] = sns.violinplot(x='Box',y='Correlation',hue='Type',data=dfData,ax=ax[1],split=False,palette='viridis', inner = 'box')
    ax[1].spines[['top', 'right']].set_visible(False)
    ax[1].legend(ncol = 2, loc ='upper center', bbox_to_anchor = (0., 1.02, 1., .102), frameon=False)  
    
    ax[0].set(xlabel=None,ylabel='Spatial correlation')
    ax[1].set(xlabel=None, ylabel=None)
    
    plt.tight_layout()
    
    # Just the box plot
    fig, ax = plt.subplots(figsize = (5,5))
    ax.set_title('Spatial correlation for '+str(nCells)+' cells during remapping')
    sns.boxplot(x='Box', y='Correlation', hue='Type', data=dfData, ax=ax, palette='viridis', dodge = True, saturation = 0.9)
    ax.legend(ncol = 1, loc ='upper right', bbox_to_anchor = (0., 0.92, 1.2, .102), frameon=False)  
    ax.spines[['top', 'right']].set_visible(False)
    ax.set(xlabel = None, ylabel = 'Spatial correlation')
    
    plt.tight_layout()
    
    #%% Check non-place cells with high correaltion between A-A'
    
    rBool = corr_AA2 > 0.8
    testers = np.where((rBool == ~PC_A) & (~PC_A == True))[0] # Python index
    
    for x in testers:
        r1 = session_dict['Ratemaps']['dfNAT0']['N'+str(x+1)] # Convert to NATEX idx
        r2 = session_dict['Ratemaps']['dfNAT1']['N'+str(x+1)] # Convert to NATEX idx
        r3 = session_dict['Ratemaps']['dfNAT2']['N'+str(x+1)] # Convert to NATEX idx
        rs = r1, r2, r3
        
        vMin, vMax = np.min(rs), np.max(rs)
        
        fig, ax = plt.subplots(1,3)
        plt.suptitle('Cell ' + str(x+1) + ': Pearson R: ' + str(round(pearsonr(r1.flatten(), r3.flatten())[0],2)))
    
        for r, s, ax in zip(rs, sessionString, ax.flat): 
            im = ax.imshow(r, vmin = vMin, vmax = vMax) 
            ax.set_title(s)
            ax.set(xlabel=None, ylabel=None, aspect = 'equal')
            ax.set_axis_off()
        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.28, 0.05, 0.45])
        fig.colorbar(im, cax=cbar_ax, label = 'Deconvolved events (A.U.)')
    
    #%% Compare the correlation between place cells to that of a shuffled distribution

    # Start parallelisation of analyses in a cell wise matter
    numWorkers = 16
    futures = []
    
    nShuffles = 25 
    dfNAT = session_dict['dfNAT']['dfNAT2']
    shuffleCorrs = np.full([nShuffles, nCells], np.nan)
    
    with cf.ProcessPoolExecutor(max_workers=numWorkers) as pool:
        for cellNo in range(1,nCells+1):
            futures.append(pool.submit(
                ratemap_shuffler,   # The function
                ratemaps,           # Ratemaps
                dfNAT,              # Timestamps, headpos, speed, cell activity
                nShuffles,          # The number of shuffles to perform
                cellNo              # Cell number, the iterable (NATEX idx)
            ))     
    
        for future in tqdm(cf.as_completed(futures), total=nCells):
            jobNo = future.result()[0]
            shuffleCorrs[:,jobNo] = future.result()[1]
    
    #%% Plot the results from before, but with the shuffles distribution for comparison
           
    df = pd.DataFrame({'Correlation': shuffleCorrs.flatten(),'Type': ['Shuffle']*(shuffleCorrs.size), 'Box': ['A-Shuffle']*(shuffleCorrs.size)})    
    
    dfShuffle = pd.concat([dfData.copy(), df])
    
    # Just the box plot
    fig, ax = plt.subplots(figsize = (5,5))
    ax.set_title('Spatial correlation for '+str(nCells)+' cells during remapping')
    sns.boxplot(x='Box', y='Correlation', hue='Type', data=dfShuffle, ax=ax, palette='viridis', dodge = True, saturation = 0.9)
    ax.legend(ncol = 1, loc ='upper right', bbox_to_anchor = (0., 0.92, 1.2, .102), frameon=False)  
    ax.spines[['top', 'right']].set_visible(False)
    ax.set(xlabel = None, ylabel = 'Spatial correlation')
    
    plt.tight_layout()
    
    # Separate the place cells from the non-place cells
    dfS1 = pd.DataFrame({'Correlation': shuffleCorrs[:,PC_A].flatten(),'Type': ['Shuffle PC']*(shuffleCorrs[:,PC_A].size), 'Box': ['A-Shuffle']*(shuffleCorrs[:,PC_A].size)})    
    dfS2 = pd.DataFrame({'Correlation': shuffleCorrs[:,~PC_A].flatten(),'Type': ['Shuffle nPC']*(shuffleCorrs[:,~PC_A].size), 'Box': ['A-Shuffle']*(shuffleCorrs[:,~PC_A].size)})    

    dfShuffle_split = pd.concat([dfData.copy(), dfS2, dfS1])
    
    # Just the box plot - split shuffle
    fig, ax = plt.subplots(figsize = (5,5))
    ax.set_title('Spatial correlation for '+str(nCells)+' cells during remapping')
    sns.boxplot(x='Box', y='Correlation', hue='Type', data=dfShuffle_split, ax=ax, palette='viridis', dodge = True, saturation = 0.9)
    ax.legend(ncol = 2, loc ='best', frameon=False)  
    ax.spines[['top', 'right']].set_visible(False)
    ax.set(xlabel = None, ylabel = 'Spatial correlation')
    
    plt.tight_layout()
    
    #%% Old code - gives a nice little plot, hence kept
    rAB_list = []
    pvalAB_list = []
    rAA2_list = []
    pvalAA2_list = []
    
    # Iterate over cellNo
    for cellNo in range(1,nCells+1):
        A  = ratemaps['dfNAT0']['N'+str(cellNo)]
        B  = ratemaps['dfNAT1']['N'+str(cellNo)]
        A2 = ratemaps['dfNAT2']['N'+str(cellNo)]
        
        # Calculate the correlation matrix and pearson R between A vs. B and A vs. A'
        corr2d_AB = correlate2d(A, B, mode='full', boundary='fill',fillvalue=0)
        corr2d_AA2= correlate2d(A, A2, mode='full', boundary='fill',fillvalue=0)
    
        rAB,pvalAB = pearsonr(A.flatten(),B.flatten())  
        rAA2,pvalAA2 = pearsonr(A.flatten(),A2.flatten())
        
        rAB_list.append(rAB)
        pvalAB_list.append(pvalAB)
        rAA2_list.append(rAA2)
        pvalAA2_list.append(pvalAA2)
    
    # Plot the rate maps and correlation matrices for one cell
    
    # Plots all cells for now (might be tedious, only plot one cell instead)
        if cellNo == 14:    
            fig = plt.figure(figsize=(8, 6))
            fig.suptitle('Remapping for N'+str(cellNo),fontsize=20)
    
            gs = gridspec.GridSpec(6, 6)
    
            ax1 = plt.subplot(gs[0:2,0:2])
            sns.heatmap(A,ax=ax1,square=True,cmap='viridis',cbar=False,xticklabels=False,yticklabels=False)
            ax1.set_title('A')
    
            ax2 = plt.subplot(gs[0:2,2:4])
            sns.heatmap(B,ax=ax2,square=True,cmap='viridis',cbar=False,xticklabels=False,yticklabels=False)
            ax2.set_title('B')
    
            ax3 = plt.subplot(gs[0:2,4:6])
            sns.heatmap(A2,ax=ax3,square=True,cmap='viridis',cbar=False,xticklabels=False,yticklabels=False)
            ax3.set_title('A\'')
    
            ax4 = plt.subplot(gs[2:6,0:3])
            sns.heatmap(corr2d_AB,ax=ax4,square=True,cmap='viridis',cbar=False,xticklabels=False,yticklabels=False)
            ax4.set_title('A vs. B, r = '+str(round(rAB,2)))
    
            ax5 = plt.subplot(gs[2:6,3:6])
            sns.heatmap(corr2d_AA2,ax=ax5,square=True,cmap='viridis',cbar=False,xticklabels=False,yticklabels=False)
            ax5.set_title('A vs. A\', r = '+str(round(rAA2,2)))
    
            plt.tight_layout()
    
    #%%
    """
    #%% Population vector correlation
    from opexebo.analysis import population_vector_correlation
    
    ratemap_stack = np.zeros([nCells,ratemaps['dfNAT0']['N1'].shape[0],ratemaps['dfNAT0']['N1'].shape[1]])
    ratemap_stack2 = np.zeros([nCells,ratemaps['dfNAT0']['N1'].shape[0],ratemaps['dfNAT0']['N1'].shape[1]])
    
    for ii in range(0,nCells):
        ratemap_stack[ii] = ratemaps['dfNAT0']['N'+str(ii+1)]
        ratemap_stack2[ii] = ratemaps['dfNAT2']['N'+str(ii+1)]
    
    [p1,p2,p3] = population_vector_correlation(ratemap_stack,ratemap_stack2)   
    
    #%% Shuffling
    from opexebo.general import shuffle
    
    timestamps = session_dict['dfNAT']['dfNAT0']['Timestamps'].to_numpy()
    pos = np.array([session_dict['dfNAT']['dfNAT0']['Head_X'].to_numpy(),session_dict['dfNAT']['dfNAT0']['Head_Y'].to_numpy()]).transpose()
    
    [T,Tinc] = shuffle(timestamps,0.1,100,0,max(timestamps))
    pos_shuffle = np.roll(pos,np.random.randint(1,len(pos)),axis=0)
    """