# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 14:31:29 2023

@author: torstsl

Bayesian decoder using a Poission noice model and a flat prior:

    Probability = n * np.log(rate) - rate   
        where:    
            rate =  the acitivty from all rate maps for all cells in given                          
                    (i.j) from the training data                  
                n = the activity vector (from the matrix) for all cells in                  
                    that timebin from the test data       

This script decodes position based on the noice model above. More details are
at the helper function (see: calc_activityMatrix, poisson_model_lg_vector, 
calc_decodingPos). The results are plotted using in scipt helper functions.

In short, data are binned by 3 frames per bin resulting bin 0.4 seconds time 
bins. The training data is based on rate maps from different populations of 
cells. An activity matrix is calculated from the testing data x time to ensure
that the rate is compareable between training and testing. The decoder gives an
likelihood matrix of positions in the testing data given the probability of 
those cells (population) being active at that location from the training data. 
The resulting decoded position is compared to the true position, and is 
reported as the Eucledian distance from the decoded to the true position.

Decoding is performed across several paradigms:
    - A to A (odds/evens training/testing) using all cells
    - A to A (odds/evens training/testing) using only place cells
    - # Longer commented out testing code
    - Decoding only by place cells (25 random place cells)
    - Decoding only from place cells with different number of cells 
    - Decoding from only nearby place cells (anatomically)
    - Decoding from n number nearby place cells to n number random place cells
    - Decoding from only place cells with similar rate maps (Pearson R > 0.5)
    - Train in A, decode A' (with all cells and only place cells)
    - Train in A, decode B (with all cells and only place cells)
    - Decoding in a novel environments vs. a familiar environment (first exposure in B)
    
"""

import pickle
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import opexebo as op
import pandas as pd
from src.cell_activity_analysis import get_signaltracking, calc_tuningmap, calc_activityMatrix, poisson_model_lg_vector, calc_decodingPos, ttest2, plot_decodingAccuracy, plot_decodingError
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr#, ttest_ind , zscore

#%% Define helper functions for running the decoder

def run_decoder(testMatrix, binFrame, trainData, dfTest):
        
    activityMatrix = calc_activityMatrix(testMatrix, binFrame)
          
    # Perform the decoding
    likelihoodMatrix = poisson_model_lg_vector(trainData, activityMatrix)

    # Calculate the decoding error
    truePos, decodingPos, decodingError = calc_decodingPos(dfTest[['Head_X', 'Head_Y']], activityMatrix, likelihoodMatrix, binFrame)
    
    # Speed filter the results
    speedThreshold = 2.5 #cm/s
    speedIdx = dfTest['Body_speed'][::binFrame].to_numpy() > speedThreshold
    
    return activityMatrix, likelihoodMatrix, truePos, decodingPos, decodingError, speedIdx          
   
#%% Run the model on session A

def decoding_tester_A(session_dict, decoding_params, **kwargs):
    
    plotter = kwargs.get('plotter', False)
    
    nSessions = nSessions = len(session_dict['dfNAT'])
    nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)
    
    for sessionNo in range(nSessions-2): # For now, only look at the first session
        
        dfNAT = session_dict['dfNAT']['dfNAT'+str(sessionNo)]
        # nFramesSession = int(session_dict['ExperimentInformation']['FrameinEachSession'][sessionNo])   
        
        # Calculate the headpos for the entire session, set bin_edges for ratemaps from these
        headpos = (dfNAT[['Head_X','Head_Y']]).to_numpy()
        
        x_edges = np.linspace(headpos[:,0].min(), headpos[:,0].max(), nBins+1)
        y_edges = np.linspace(headpos[:,1].min(), headpos[:,1].max(), nBins+1)
        bin_edges = (x_edges, y_edges)
         
        # Split the session in two, bin by bin given by number of frames 
        binFrame, timeBin = decoding_params['binFrame'], decoding_params['timeBin']
        
        # Every other binFrame datapoints go to train and tests data
        trainIdx = np.arange(len(dfNAT) - len(dfNAT)%binFrame).reshape((len(dfNAT)//binFrame, binFrame))[::2].flatten() # Index of every two 3-by-3 datapoint
        testIdx = np.arange(len(dfNAT) - len(dfNAT)%binFrame).reshape((len(dfNAT)//binFrame, binFrame))[1::2].flatten() # As above, but starting from the second 3-by-3 bin
        
        dfTrain, dfTest = dfNAT.iloc[trainIdx].copy(), dfNAT.iloc[testIdx].copy()
        
        # Z-score the deconvolved signal for each cell to its own signal (dealing with outliers with extremely high signal values)
        #dfTrain.iloc[:, 15:dfTrain.shape[1]:4] = zscore(dfTrain.iloc[:, 15:dfTrain.shape[1]:4], axis = 0, nan_policy = 'omit')
        #dfTest.iloc[:, 15:dfTest.shape[1]:4] = zscore(dfTest.iloc[:, 15:dfTest.shape[1]:4], axis = 0, nan_policy = 'omit')
        
        # Normalize from 0-1 by dividing by max value for each cell
        #dfTrain.iloc[:, 15:dfTrain.shape[1]:4] = (dfTrain.iloc[:, 15:dfTrain.shape[1]:4]) / np.nanmax(dfTrain.iloc[:, 15:dfTrain.shape[1]:4], axis = 0)
        #dfTest.iloc[:, 15:dfTest.shape[1]:4] = (dfTest.iloc[:, 15:dfTest.shape[1]:4]) / np.nanmax(dfTest.iloc[:, 15:dfTest.shape[1]:4], axis = 0)
        
        # Get occupancy for the train data, and calculate rate maps for this   
        timestampsTrain = (dfTrain.Timestamps).to_numpy()
        headposTrain    = (dfTrain[['Head_X','Head_Y']]).to_numpy()
        
        trainOccupancy = op.analysis.spatial_occupancy(timestampsTrain, np.transpose(headposTrain), boxSize, bin_edges = bin_edges)[0]
       
        # Create a 3D matrix of rate maps: One half --> Training data
        trainData = np.full([nCells, nBins, nBins], np.nan)
    
        for cellNo in range(1, nCells+1):
            signaltracking = get_signaltracking(dfTrain, cellNo, signal = 'deconvolved', speed_threshold = 2.5)
            
            trainTuningmap = calc_tuningmap(trainOccupancy, bin_edges[0], bin_edges[1], signaltracking, 2.5)['tuningmap']
            trainRatemap = gaussian_filter(trainTuningmap, 1.5)
            trainData[cellNo-1] = trainRatemap
        
        # Scale the training data to match the frequency of a time bin    
        trainData = trainData * timeBin

    # Calculate activity matrix, perform the decoding and calculate results from it
    testMatrix = dfTest.iloc[:, 15:dfTest.shape[1]:4] # Activity for all cells for test timestamps, to be binned
    
    # Perform the decoding and calculate decoding error
    activityMatrix, likelihoodMatrix, truePos, decodingPos, decodingError, speedIdx = run_decoder(testMatrix, binFrame, trainData, dfTest)
   
    stats = pearsonr(decodingError[speedIdx], dfTest['Body_speed'][::binFrame].to_numpy()[speedIdx])
    
    # Put output into a results dict
    decode_allCell_A = {'activityMatrix': activityMatrix,
                            'likelihoodMatrix': likelihoodMatrix,
                            'truePos': truePos,
                            'decodingPos': decodingPos,
                            'decodingError': decodingError,
                            'speedIdx': speedIdx}
           
    if plotter == True:        
    # Plot the max activity per cell over all bins from the activity matrix
        fig, ax = plt.subplots(2,1)
        ax[0].set_title('Max activity per cell over all time bins')
        ax[0].plot(np.nanmax(activityMatrix, axis = 0), color = 'grey')
        ax[0].set_xlim([0, nCells])
        ax[0].set_xlabel('Cell number')
        ax[0].set_ylabel('Max activity, sum decon.')
        ax[1].hist(np.nanmax(activityMatrix, axis = 0), 100, color = 'grey')
        ax[1].set_xlabel('Max activity, sum decon.')
        ax[1].set_ylabel('Num cells')
        ax[1].vlines(np.nanmax(activityMatrix, axis = 0).max(), 0, np.histogram(np.nanmax(activityMatrix, axis = 0), 100)[0].max(), 
                     color = 'r', linestyles = '--')
        plt.tight_layout()
        
        # Plot a 2D histogram to see if some positions are biased or not (not speed filtered)
        fig, ax = plt.subplots(1,1, sharex = True, sharey = True)
        im = ax.hist2d(decodingPos[:,0], decodingPos[:,1], bins = nBins)
        ax.set_aspect('equal')
        ax.set_title('2D histogram of decoded position')
        ax.set_xlabel('Decoded X')
        ax.set_ylabel('Decoded Y')
        fig.colorbar(im[3], label = 'Time bins')
        plt.tight_layout()
    
        # Plot the decoding results
        timeBinNo = 0
        
        # Plot the log likelihood matrix for this time bin
        y_ind, x_ind = np.unravel_index(np.argmax(likelihoodMatrix[timeBinNo]), (32,32)) # Row (y), column (x)
        
        fig, ax = plt.subplots()
        ax.imshow(likelihoodMatrix[timeBinNo])
        ax.scatter(x_ind, y_ind, marker = 'x', color = 'r')
        ax.set_title('Likelihood matrix')
        
        # Plot the trajectory for the test data with true and decoded position
        fig, ax = plt.subplots()
        ax.set_title('Decoded vs. true position')
        ax.set_aspect('equal')
        ax.plot(truePos[:,0], truePos[:,1], color = 'grey', zorder = 0)
        ax.scatter(decodingPos[timeBinNo][0], decodingPos[timeBinNo][1], marker = 'x', color = 'r', zorder = 1, label = 'Decoded')
        ax.scatter(truePos[timeBinNo][0], truePos[timeBinNo][1], marker='o', color = 'b', zorder = 1, label = 'True')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon = False)
        plt.tight_layout()
        
        # Plot the distribution of the decoding error
        plot_decodingError(decodingError, 'Decoding error')
        
        # Plot the decoding error and speed together
        stats = pearsonr(decodingError, dfTest['Body_speed'][::binFrame].to_numpy())
        
        fig, ax = plt.subplots(figsize = (14,2))
        ax.plot(decodingError[0:500], color = palette[100], label = 'Decoding error')
        ax.plot(dfTest['Body_speed'][::binFrame].to_numpy()[0:500], color = contrast[190], label = 'Speed')
        ax.legend(ncol = 2, frameon = False)
        ax.set_title('R = '+str(round(stats[0],2))+', p = '+str(round(stats[1],20)))
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim([0,500])
        
        start, stop = 0, 300
        
        # Plot the decoding accuracy without speed filer
        plot_decodingAccuracy(truePos, decodingPos, np.full([len(truePos)], True), start, stop, timeBin, 'Decoding accuracy (no filter)', save = False)
        
        # Plot the decoding accuracy of x and y separately with speed filter
        plot_decodingAccuracy(truePos, decodingPos, speedIdx, start, stop, timeBin, 'Decoding accuracy: A vs. A', save = True)
        
        # Plot the distribution of the decoding error
        plot_decodingError(decodingError[speedIdx], 'Decoding error', save = True)
        
        # Plot the decoding error and speed together
        fig, ax = plt.subplots(figsize = (14,2))
        ax.plot(decodingError[speedIdx][0:500], color = palette[100], label = 'Decoding error')
        ax.plot(dfTest['Body_speed'][::binFrame].to_numpy()[speedIdx][0:500], color = contrast[190], label = 'Speed')
        ax.set_xlim([0,500])
        ax.legend(ncol = 1, loc = 'upper right', frameon = False)
        ax.set_title('R = '+str(round(stats[0],2))+', p = '+str(round(stats[1],20)))
        ax.spines[['top', 'right']].set_visible(False)
        
        # Plot a 2D histogram to see if some positions are biased or not (speed filtered)
        fig, ax = plt.subplots(1,1, sharex = True, sharey = True)
        im = ax.hist2d(decodingPos[speedIdx, 0], decodingPos[speedIdx, 1], bins = (nBins))
        ax.set_aspect('equal')
        ax.set_title('2D histogram of decoded position (speed filtered)')
        ax.set_xlabel('Decoded X')
        ax.set_ylabel('Decoded Y')
        fig.colorbar(im[3], label = 'Time bins')
        plt.tight_layout()

    # From A, train only from place cells and decode position in A
        
    # Place cells
    placecells = session_dict['Placecell']['NAT0'][0]
    
    # From the previous trainData, grab only that for the place cells
    trainData = trainData[placecells-1] # From NATEX to Python index
       
    # Calculate the activity matrix
    keys = []
    for x in placecells: keys.append('Deconvolved, N'+str(x))
    testMatrix = dfTest[keys] # Activity for all cells for test timestamps, to be binned
    
    # Perform the decoding and calculate decoding error
    activityMatrix, likelihoodMatrix, truePos, decodingPos, decodingError, speedIdx = run_decoder(testMatrix, binFrame, trainData, dfTest)
    
    stats = pearsonr(decodingError[speedIdx], dfTest['Body_speed'][::binFrame].to_numpy()[speedIdx])
    
    decode_PC_A = {'activityMatrix': activityMatrix,
                       'likelihoodMatrix': likelihoodMatrix,
                       'truePos': truePos,
                       'decodingPos': decodingPos,
                       'decodingError': decodingError,
                       'speedIdx': speedIdx}
    
    if plotter == True:
    
        # Plot the log likelihood matrix for this time bin
        timeBinNo = 0
        
        y_ind, x_ind = np.unravel_index(np.argmax(likelihoodMatrix[timeBinNo]), (32,32)) # Row, column
        
        plt.figure()
        plt.imshow(likelihoodMatrix[timeBinNo])
        plt.scatter(x_ind, y_ind, marker = 'x', color = 'r')
        plt.title('Likelihood matrix')
        
        # Plot the trajectory for the test data with true and decoded position
        fig, ax = plt.subplots()
        ax.set_title('Decoded vs. true position (A vs. A)')
        ax.set_aspect('equal')
        ax.plot(truePos[:,0], truePos[:,1], color = 'grey', zorder = 0)
        ax.scatter(decodingPos[timeBinNo][0], decodingPos[timeBinNo][1], marker = 'x', color = 'r', zorder = 1, label = 'Decoded')
        ax.scatter(truePos[timeBinNo][0], truePos[timeBinNo][1], marker='o', color = 'b', zorder = 1, label = 'True')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon = False)
        plt.tight_layout()
        
        # Plot the distribution of the decoding error (with and without speed filtering)
        plot_decodingError(decodingError, 'Decoding error (A vs. A)', save = False)
        plot_decodingError(decodingError[speedIdx], 'Decoding error (A vs. A)', save = True)
        
        # Plot the decoding error and speed together
        if decodingError.shape[0] == dfTest['Body_speed'][::binFrame].to_numpy().shape[0]:
            stats = pearsonr(decodingError, dfTest['Body_speed'][::binFrame].to_numpy())
        else: 
            nDel = dfTest.shape[0]%binFrame
            dfTest = dfTest.iloc[0:len(dfTest)-nDel]
            stats = pearsonr(decodingError, dfTest['Body_speed'][::binFrame][0:decodingError.shape[0]].to_numpy())
        
        fig, ax = plt.subplots(figsize = (14,2))
        ax.plot(decodingError[0:500], color = palette[100], label = 'Decoding error')
        ax.plot(dfTest['Body_speed'][::binFrame].to_numpy()[0:500], color = contrast[190], label = 'Speed')
        ax.legend(ncol = 1, frameon = False)
        ax.set_title('R = '+str(round(stats[0],2))+', p = '+str(round(stats[1],20)))
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim([0,500])
        
        # Plot the decoding accuracy of x and y separately
        plot_decodingAccuracy(truePos, decodingPos, speedIdx, start, stop, timeBin, 'Decoding accuracy: A vs. A', save = True)
        
        # Plot the decoding error and speed together        
        fig, ax = plt.subplots(figsize = (14,2))
        ax.plot(decodingError[speedIdx][0:500], color = palette[100], label = 'Decoding error')
        ax.plot(dfTest['Body_speed'][::binFrame].to_numpy()[speedIdx][0:500], color = contrast[190], label = 'Speed')
        ax.set_xlim([0,500])
        ax.legend(ncol = 2, bbox_to_anchor=(0.5, 0.7, 0.5, 0.5), frameon = False)
        ax.set_title('R = '+str(round(stats[0],2))+', p = '+str(round(stats[1],20)))
        ax.spines[['top', 'right']].set_visible(False)
    
    # Outputs are decoding results from ALL cells in session A and just the PCs in session A
    
    return decode_allCell_A, decode_PC_A

#%% TESTING: Look at weird signal
"""
import time
from src.cell_activity_analysis import poisson_model_vector, poisson_model_lg

truePos = truePos[speedIdx]
testPos = decodingPos[speedIdx]
testErr = decodingError[speedIdx]

fig, ax = plt.subplots(1,2)
ax[0].plot(testPos[:,0], testPos[:,1], color = palette[80])
ax[0].set_aspect('equal')
ax[1].scatter(testPos[:,0], testPos[:,1],  cmap = 'viridis')
ax[1].set_aspect('equal')

fig, ax = plt.subplots(2,2)
ax[0,0].plot(truePos[:,0], truePos[:,1], linewidth = 0.5, color = palette[80], alpha = 0.75)
ax[0,1].plot(testPos[:,0], testPos[:,1], linewidth = 0.5, color = contrast[180], alpha = 0.75)
ax[0,0].set_aspect('equal')
ax[0,1].set_aspect('equal')
ax[1,0].scatter(truePos[:,0], truePos[:,1], s = 5, color = palette[80], alpha = 0.75)
ax[1,1].scatter(testPos[:,0], testPos[:,1], s = 5, color = contrast[180], alpha = 0.75)
ax[1,0].set_aspect('equal')
ax[1,1].set_aspect('equal')

fig, ax = plt.subplots()
ax.hist2d(testPos[:,0], testPos[:,1], (32,32))
ax.set_aspect('equal')

fig, ax = plt.subplots(2,1, figsize = (10,4), sharex = True)
ax[0].plot(truePos[0:100,0], color = palette[100], label = 'True') 
ax[0].plot(testPos[0:100,0], color = contrast[190], label = 'Decoded')
ax[0].set_ylabel('X')
ax[0].set_xlim([0,100])
ax[0].legend(ncol = 2, loc='best', bbox_to_anchor=(0.5, 1.05, 0.5, 0.5))
ax[1].plot(truePos[0:100,1], color = palette[100], label = 'True') 
ax[1].plot(testPos[0:100,1], color = contrast[190], label = 'Decoded')
ax[1].set_ylabel('Y')
ax[1].set_xlim([0,100])
fig.supxlabel('Time bin')
fig.suptitle('Speed filtered')

fig, ax = plt.subplots(1,5, figsize = (10,4))
for ii in range(5):
    ax[ii].imshow(likelihoodMatrix[speedIdx][20+ii])
    y_ind, x_ind = np.unravel_index(np.argmax(likelihoodMatrix[speedIdx][20+ii]), (32,32)) # Row, column
    ax[ii].scatter(x_ind, y_ind, marker = 'x', color = 'r')
    ax[ii].axis('off')


tData = trainData[0:100]
aMatrix = activityMatrix[:,0:100]

# original code from Torstein
t0 = time.perf_counter()
ground_truth = poisson_model_vector(tData, aMatrix)
t1 = time.perf_counter()
print(t1-t0)

# log code
t0 = time.perf_counter()
test = poisson_model_lg(tData, aMatrix)
t1 = time.perf_counter()
print(t1-t0)

# new timebin
i = 0
test_fig = test[i]
gt_fig = np.log(ground_truth[i])
df_fig = np.abs(test_fig - gt_fig)
vmin = min(np.min(gt_fig), np.min(test_fig), np.min(df_fig))
vmax = max(np.max(gt_fig), np.max(test_fig), np.max(df_fig))

fig, ax = plt.subplots(1,3, figsize=(10,3))
ax[0].set_title("Ground truth")
ax[0].imshow(gt_fig, vmin=vmin, vmax=vmax)
ax[1].set_title("Test code")
ax[1].imshow(test_fig, vmin=vmin, vmax=vmax)
ax[2].set_title("Diff")
ax[2].imshow(df_fig, vmin=vmin, vmax=vmax)
plt.show()
print(f"Relative error: { np.nanmax(df_fig / test_fig)}")


# Log code
t0 = time.perf_counter()
ground_truth = poisson_model_lg(tData, aMatrix)
t1 = time.perf_counter()
print(t1-t0)

# Vectorized log code
t0 = time.perf_counter()
test = poisson_model_lg_vector(tData, aMatrix)
t1 = time.perf_counter()
print(t1-t0)

# new timebin
i = 5
test_fig = test[i]
gt_fig = ground_truth[i]
df_fig = np.abs(test_fig - gt_fig)
vmin = min(np.min(gt_fig), np.min(test_fig), np.min(df_fig))
vmax = max(np.max(gt_fig), np.max(test_fig), np.max(df_fig))

fig, ax = plt.subplots(1,3, figsize=(10,3))
ax[0].set_title("Ground truth")
ax[0].imshow(gt_fig, vmin=vmin, vmax=vmax)
ax[1].set_title("Test code")
ax[1].imshow(test_fig, vmin=vmin, vmax=vmax)
ax[2].set_title("Diff")
ax[2].imshow(df_fig, vmin=vmin, vmax=vmax)
plt.show()
print(f"Relative error: { np.nanmax(df_fig / test_fig)}")

t, d, e, b = calc_decodingPos(dfTest[['Head_X', 'Head_Y']], aMatrix, test, binFrame)

# Decoding accuracy 
histbins = 50 

fig, ax = plt.subplots()
ax.hist(e[speedIdx], histbins, color = 'grey')
ax.vlines(np.median(e[speedIdx]), 0, np.histogram(e[speedIdx],histbins)[0].max(), color = contrast[190])
ax.set_xlim(0)
ax.set_title('Decoding error (speed filtered), median = '+str(round(np.median(e[speedIdx]),2)) + ' cm')
ax.set_xlabel('Distance (cm)')
ax.set_ylabel('Counts')
plt.tight_layout() 

fig, ax = plt.subplots(2,1, figsize = (10,4), sharex = True)
ax[0].plot(t[speedIdx][0:50,0], color = palette[100], label = 'True') 
ax[0].plot(d[speedIdx][0:50,0], color = contrast[190], label = 'Decoded')
ax[0].set_ylabel('X')
ax[0].set_xlim([0,50])
ax[0].legend(ncol = 2, loc='best', bbox_to_anchor=(0.5, 1.05, 0.5, 0.5))
ax[1].plot(t[speedIdx][0:50,1], color = palette[100], label = 'True') 
ax[1].plot(d[speedIdx][0:50,1], color = contrast[190], label = 'Decoded')
ax[1].set_ylabel('Y')
ax[1].set_xlim([0,50])
fig.supxlabel('Time bin')
fig.suptitle('Decoding accuracy (log, speed filtered)')


t_bin = np.full(truePos.shape, np.nan)
for i, j, n in zip(truePos[:,0], truePos[:,1], range(2500)):
    x_ind = np.where(i <= bin_edges[0])[0][0] - 1
    y_ind = np.where(j <= bin_edges[1])[0][0] - 1
    if x_ind == -1: x_ind = 0
    if y_ind == -1: y_ind = 0
    t_bin[n,0] = x_ind
    t_bin[n,1] = y_ind

fig, axes = plt.subplots(3,3, figsize = (10,10))
for ii, ax in enumerate(axes.flat):
    ax.imshow(np.flipud(likelihoodMatrix[ii]))
    y, x = np.unravel_index(np.argmax(np.flipud(likelihoodMatrix[ii])), (32,32)) # Row, column, needs to flip the matrix
    ax.scatter(x, y, s = 100, marker = 'x', color = contrast[180])
    ax.scatter(t_bin[ii,0], t_bin[ii,1], s = 100, color = palette[100]) 
    ax.set_xlim([0,32])
    ax.set_ylim([0,32])
    ax.axis('off')
    
fig, axes = plt.subplots(3,3, figsize = (10,10))
for ii, ax in enumerate(axes.flat):
    ax.scatter(truePos[ii,0], truePos[ii,1], s = 100, color = palette[100])
    ax.scatter(decodingPos[ii,0], decodingPos[ii,1], marker = 'x',s = 100, color = contrast[180])
    ax.set_xlim(-45,45)
    ax.set_ylim(-45,45)
"""
#%% Decode only by 25 random place cells

def decoding_from_25_placecells(session_dict, decoding_params, **kwargs): 
    
    plotter = kwargs.get('plotter', False)
    
    placecell_dict = session_dict['Placecell']
    nSessions = nSessions = len(session_dict['dfNAT'])
    
    maxPC = 25
    random = True
    
    decode_25PC = {}
    
    for sessionNo in range(nSessions):
        
        dfNAT = session_dict['dfNAT']['dfNAT'+str(sessionNo)]
        nPC = placecell_dict['NAT'+str(sessionNo)][0].shape[0]
        
        # If random == True, only use a defined number of place cells picked randomly
        if random == True:
            randPC = np.random.choice(placecell_dict['NAT'+str(sessionNo)][0], maxPC, replace = False)
            PCs = ['Deconvolved, N'+str(x) for x in randPC]        
            dfNAT = dfNAT[list(dfNAT.keys()[0:12])+PCs]
            
        else:    
            PCs = ['Deconvolved, N'+str(x) for x in placecell_dict['NAT'+str(sessionNo)][0]]
            dfNAT = dfNAT[list(dfNAT.keys()[0:12])+PCs]
        
        # Calculate the headpos for the entire session, set bin_edges for ratemaps from these
        headpos = (dfNAT[['Head_X','Head_Y']]).to_numpy()
        
        x_edges = np.linspace(headpos[:,0].min(), headpos[:,0].max(), nBins+1)
        y_edges = np.linspace(headpos[:,1].min(), headpos[:,1].max(), nBins+1)
        bin_edges = (x_edges, y_edges)
         
        # Split the session in two, bin by bin given by number of frames 
        binFrame, timeBin = decoding_params['binFrame'], decoding_params['timeBin']
        
        # Every other binFrame datapoints go to train and tests data
        trainIdx = np.arange(len(dfNAT) - len(dfNAT)%binFrame).reshape((len(dfNAT)//binFrame, binFrame))[::2].flatten() # Index of every two 3-by-3 datapoint
        testIdx = np.arange(len(dfNAT) - len(dfNAT)%binFrame).reshape((len(dfNAT)//binFrame, binFrame))[1::2].flatten() # As above, but starting from the second 3-by-3 bin
        
        dfTrain, dfTest = dfNAT.iloc[trainIdx].copy(), dfNAT.iloc[testIdx].copy()
        
        # Get occupancy for the train data, and calculate rate maps for this   
        timestampsTrain = (dfTrain.Timestamps).to_numpy()
        headposTrain    = (dfTrain[['Head_X','Head_Y']]).to_numpy()
        
        trainOccupancy = op.analysis.spatial_occupancy(timestampsTrain, np.transpose(headposTrain), boxSize, bin_edges = bin_edges)[0]
       
        # Create a 3D matrix of rate maps: One half --> Training data
        if random == True:
            trainData = np.full([maxPC, nBins, nBins], np.nan)
            
            for num in range(0, maxPC):
                cellNo = randPC[num]
                signaltracking = get_signaltracking(dfTrain, cellNo, signal = 'deconvolved', speed_threshold = 2.5)
                
                trainTuningmap = calc_tuningmap(trainOccupancy, bin_edges[0], bin_edges[1], signaltracking, 2.5)['tuningmap']
                trainRatemap = gaussian_filter(trainTuningmap, 1.5)
                trainData[num] = trainRatemap
                
        else:    
            trainData = np.full([nPC, nBins, nBins], np.nan)
            
            for num in range(0, nPC):
                cellNo = placecell_dict['NAT'+str(sessionNo)][0][num]
                signaltracking = get_signaltracking(dfTrain, cellNo, signal = 'deconvolved', speed_threshold = 2.5)
                
                trainTuningmap = calc_tuningmap(trainOccupancy, bin_edges[0], bin_edges[1], signaltracking, 2.5)['tuningmap']
                trainRatemap = gaussian_filter(trainTuningmap, 1.5)
                trainData[num] = trainRatemap
        
        # Scale the training data to match the frequency of a time bin    
        trainData = trainData * timeBin
    
        # Calculate activity matrix, perform the decoding and calculate results from it
        testMatrix = dfTest.iloc[:, 12:] # Activity for all cells for test timestamps, to be binned
        
        # Perform the decoding and calculate decoding error
        activityMatrix, likelihoodMatrix, truePos, decodingPos, decodingError, speedIdx = run_decoder(testMatrix, binFrame, trainData, dfTest)
        
        decode_25PC['NAT'+str(sessionNo)] = {'activityMatrix': activityMatrix,
                                                 'likelihoodMatrix': likelihoodMatrix,
                                                 'truePos': truePos,
                                                 'decodingPos': decodingPos,
                                                 'decodingError': decodingError,
                                                 'speedIdx': speedIdx}
        
        if plotter == True:
            
            sessionName = ['A','B','A\'']
            
            fig, ax = plt.subplots()
            ax.set_title('Session '+sessionName[sessionNo]+' , PCs: '+str(maxPC)+'/'+str(nPC))
            ax.imshow(np.arange(100).reshape(1,100), cmap = 'OrRd')
            ax.set(frame_on=False, xticks=[], yticks=[])
            
            # Plot the trajectory for the test data with true and decoded position
            timeBinNo = 0
            
            fig, ax = plt.subplots()
            ax.set_title('Decoded vs. true position')
            ax.set_aspect('equal')
            ax.plot(truePos[:,0], truePos[:,1], color = 'grey', zorder = 0)
            ax.scatter(decodingPos[timeBinNo][0], decodingPos[timeBinNo][1], marker = 'x', color = 'r', zorder = 1, label = 'Decoded')
            ax.scatter(truePos[timeBinNo][0], truePos[timeBinNo][1], marker='o', color = 'b', zorder = 1, label = 'True')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon = False)
            plt.tight_layout()
            
            # Plot the distribution of the decoding error
            start, stop = 0, 300
            
            plot_decodingError(decodingError, 'Decoding error (no filter) ' +str(maxPC)+' PCs', save = False)
            
            # Decoding accuracy with speed filter
            plot_decodingAccuracy(truePos, decodingPos, speedIdx, start, stop, timeBin, 
                                      'Decoding accuracy (speed filtered), '+str(maxPC)+' PCs', save = False)
                
            # Plot the distribution of the decoding error
            plot_decodingError(decodingError[speedIdx], 'Decoding error (speed filtered) ' +str(maxPC)+' PCs', save = False)
    
    # Dict over all sessions with results from decoding by 25 place cells
    
    return decode_25PC


def decoding_from_nPCs(session_dict, decoding_params, **kwargs):

     plotter = kwargs.get('plotter', False)

     nSessions = nSessions = len(session_dict['dfNAT'])
     placecell_dict = session_dict['Placecell']
    
     maxPC_arr = np.linspace(25, 200, 8, dtype = int)
     nIter = 50
     medians, means = {}, {}
    
     decode_nPC = {}
    
     for maxPC in maxPC_arr:    
         medians[maxPC] = np.full([nIter, nSessions], np.nan)
         means[maxPC] = np.full([nIter, nSessions], np.nan)
    
         for iteration in range(nIter):
             for sessionNo in range(nSessions):
    
                 dfNAT = session_dict['dfNAT']['dfNAT'+str(sessionNo)]
                 nPC = placecell_dict['NAT'+str(sessionNo)][0].shape[0]
    
                 if nPC >= maxPC:
                     # Only use a defined number of place cells picked randomly
                     randPC = np.random.choice(placecell_dict['NAT'+str(sessionNo)][0], maxPC, replace = False)
                     PCs = ['Deconvolved, N'+str(x) for x in randPC]        
                     dfNAT = dfNAT[list(dfNAT.keys()[0:12])+PCs]
    
                     # Calculate the headpos for the entire session, set bin_edges for ratemaps from these
                     headpos = (dfNAT[['Head_X','Head_Y']]).to_numpy()
    
                     x_edges = np.linspace(headpos[:,0].min(), headpos[:,0].max(), nBins+1)
                     y_edges = np.linspace(headpos[:,1].min(), headpos[:,1].max(), nBins+1)
                     bin_edges = (x_edges, y_edges)
    
                     # Split the session in two, bin by bin given by number of frames 
                     binFrame, timeBin = decoding_params['binFrame'], decoding_params['timeBin']
    
                     # Every other binFrame datapoints go to train and tests data
                     trainIdx = np.arange(len(dfNAT) - len(dfNAT)%binFrame).reshape((len(dfNAT)//binFrame, binFrame))[::2].flatten() # Index of every two 3-by-3 datapoint
                     testIdx = np.arange(len(dfNAT) - len(dfNAT)%binFrame).reshape((len(dfNAT)//binFrame, binFrame))[1::2].flatten() # As above, but starting from the second 3-by-3 bin
    
                     dfTrain, dfTest = dfNAT.iloc[trainIdx].copy(), dfNAT.iloc[testIdx].copy()
    
                     # Get occupancy for the train data, and calculate rate maps for this   
                     timestampsTrain = (dfTrain.Timestamps).to_numpy()
                     headposTrain    = (dfTrain[['Head_X','Head_Y']]).to_numpy()
    
                     trainOccupancy = op.analysis.spatial_occupancy(timestampsTrain, np.transpose(headposTrain), boxSize, bin_edges = bin_edges)[0]
    
                     # Create a 3D matrix of rate maps: One half --> Training data
                     trainData = np.full([maxPC, nBins, nBins], np.nan)
    
                     for num in range(0, maxPC):
                         cellNo = randPC[num]
                         signaltracking = get_signaltracking(dfTrain, cellNo, signal = 'deconvolved', speed_threshold = 2.5)
    
                         trainTuningmap = calc_tuningmap(trainOccupancy, bin_edges[0], bin_edges[1], signaltracking, 2.5)['tuningmap']
                         trainRatemap = gaussian_filter(trainTuningmap, 1.5)
                         trainData[num] = trainRatemap
    
                     # Scale the training data to match the frequency of a time bin    
                     trainData = trainData * timeBin
    
                     # Calculate activity matrix, perform the decoding and calculate results from it
                     testMatrix = dfTest.iloc[:, 12:] # Activity for all cells for test timestamps, to be binned
    
                     # Perform the decoding and calculate decoding error
                     activityMatrix, likelihoodMatrix, truePos, decodingPos, decodingError, speedIdx = run_decoder(testMatrix, binFrame, trainData, dfTest)
    
                     # Decoding accuracy with speed filter            
                     medians[maxPC][iteration, sessionNo] = np.median(decodingError[speedIdx])
                     means[maxPC][iteration, sessionNo] = np.nanmean(decodingError[speedIdx])
    
                 decode_nPC['NAT'+str(sessionNo)] = {'activityMatrix': activityMatrix,
                                                     'likelihoodMatrix': likelihoodMatrix,
                                                     'truePos': truePos,
                                                     'decodingPos': decodingPos,
                                                     'decodingError': decodingError,
                                                     'speedIdx': speedIdx}
    
     # Plot decoding error as violins and boxes for each session
     if plotter == True: 
         sessionName = ['A','B','A\'']
    
         dfA, dfB, dfA2 = {}, {}, {}
    
         for key in means.keys():
             dfA[key] = means[key][:,0]
             dfB[key] = means[key][:,1]
             dfA2[key] = means[key][:,2]
    
         dfPlot = pd.DataFrame(dfA), pd.DataFrame(dfB), pd.DataFrame(dfA2)
    
         for sessionNo in range(nSessions):
             df = dfPlot[sessionNo]
             fig, ax = plt.subplots(1, 2, figsize = (10,5), sharey = True)
             sns.violinplot(data = df, ax = ax[0], palette = 'viridis')
             ax[0].spines[['top', 'right']].set_visible(False)
             sns.boxplot(data = df, ax = ax[1], palette = 'viridis')
             ax[1].spines[['top', 'right']].set_visible(False)
             #sns.lineplot(data = dfA.melt(), x = 'variable', y = 'value', ax = ax[2], estimator = 'mean', ci = 95)
             fig.supxlabel('Number of cells')
             fig.supylabel('Decoding error (cm)')
             fig.suptitle(sessionName[sessionNo]+': Decoding error by number of random place cells (50 it.)')
             plt.tight_layout()
    
         # The the decoding error as violins and boxes for all sessions concatenated
         dfPlotAll = {}
         for key in means.keys(): dfPlotAll[key] = means[key].flatten()
         dfPlotAll = pd.DataFrame.from_dict(dfPlotAll)
    
         fig, ax = plt.subplots(1, 2, figsize = (9,5), sharey = True)
         sns.violinplot(data = dfPlotAll, ax = ax[0], palette = 'viridis')
         ax[0].spines[['top', 'right']].set_visible(False)
         sns.boxplot(data = dfPlotAll, ax = ax[1], palette = 'viridis')
         ax[1].spines[['top', 'right']].set_visible(False)
         #sns.stripplot(data = dfPlotAll, ax = ax[1], palette = 'viridis', jitter = 0.01, alpha = 0.5)
         fig.supxlabel('Number of cells')
         fig.supylabel('Decoding error (cm)')
         fig.suptitle('Decoding error by number of random place cells (150 it.), all sessions')
         plt.tight_layout()
    
         # Plot the decoding error as swarm for all sessions
         fig, ax = plt.subplots(figsize = (5,4))
         sns.swarmplot(data = dfPlotAll, ax = ax, palette = 'viridis', size = 2)
         ax.set_xlabel('Number of cells')
         ax.set_ylabel('Decoding error (cm)')
         ax.set_title('Decoding error by number of random place cells (150 it.), all sessions')
         ax.spines[['top', 'right']].set_visible(False)
         plt.tight_layout()

     return decode_nPC, medians, means
   
#%% Decode only by place cells: Iterate over number of place cells and perform the random sampling and decoding several times

def decode_from_nPCs(session_dict, decoding_params, **kwargs):

    plotter = kwargs.get('plotter', False)
    
    nSessions = nSessions = len(session_dict['dfNAT'])
    placecell_dict = session_dict['Placecell']
    
    maxPC_arr = np.linspace(25, 200, 8, dtype = int)
    nIter = 50
    medians, means = {}, {}
    
    decode_nPC = {}
    
    for maxPC in maxPC_arr:    
        medians[maxPC] = np.full([nIter, nSessions], np.nan)
        means[maxPC] = np.full([nIter, nSessions], np.nan)
    
        for iteration in range(nIter):
            for sessionNo in range(nSessions):
                
                dfNAT = session_dict['dfNAT']['dfNAT'+str(sessionNo)]
                # nFramesSession = int(session_dict['ExperimentInformation']['FrameinEachSession'][sessionNo])
                nPC = placecell_dict['NAT'+str(sessionNo)][0].shape[0]
                
                if nPC >= maxPC:
                    # Only use a defined number of place cells picked randomly
                    randPC = np.random.choice(placecell_dict['NAT'+str(sessionNo)][0], maxPC, replace = False)
                    PCs = ['Deconvolved, N'+str(x) for x in randPC]        
                    dfNAT = dfNAT[list(dfNAT.keys()[0:12])+PCs]
                    
                    # Calculate the headpos for the entire session, set bin_edges for ratemaps from these
                    headpos = (dfNAT[['Head_X','Head_Y']]).to_numpy()
                    
                    x_edges = np.linspace(headpos[:,0].min(), headpos[:,0].max(), nBins+1)
                    y_edges = np.linspace(headpos[:,1].min(), headpos[:,1].max(), nBins+1)
                    bin_edges = (x_edges, y_edges)
                     
                    # Split the session in two, bin by bin given by number of frames 
                    binFrame, timeBin = decoding_params['binFrame'], decoding_params['timeBin']
                    
                    # Every other binFrame datapoints go to train and tests data
                    trainIdx = np.arange(len(dfNAT) - len(dfNAT)%binFrame).reshape((len(dfNAT)//binFrame, binFrame))[::2].flatten() # Index of every two 3-by-3 datapoint
                    testIdx = np.arange(len(dfNAT) - len(dfNAT)%binFrame).reshape((len(dfNAT)//binFrame, binFrame))[1::2].flatten() # As above, but starting from the second 3-by-3 bin
                    
                    dfTrain, dfTest = dfNAT.iloc[trainIdx].copy(), dfNAT.iloc[testIdx].copy()
                    
                    # Get occupancy for the train data, and calculate rate maps for this   
                    timestampsTrain = (dfTrain.Timestamps).to_numpy()
                    headposTrain    = (dfTrain[['Head_X','Head_Y']]).to_numpy()
                    
                    trainOccupancy = op.analysis.spatial_occupancy(timestampsTrain, np.transpose(headposTrain), boxSize, bin_edges = bin_edges)[0]
                   
                    # Create a 3D matrix of rate maps: One half --> Training data
                    trainData = np.full([maxPC, nBins, nBins], np.nan)
        
                    for num in range(0, maxPC):
                        cellNo = randPC[num]
                        signaltracking = get_signaltracking(dfTrain, cellNo, signal = 'deconvolved', speed_threshold = 2.5)
                        
                        trainTuningmap = calc_tuningmap(trainOccupancy, bin_edges[0], bin_edges[1], signaltracking, 2.5)['tuningmap']
                        trainRatemap = gaussian_filter(trainTuningmap, 1.5)
                        trainData[num] = trainRatemap
                    
                    # Scale the training data to match the frequency of a time bin    
                    trainData = trainData * timeBin
                
                    # Calculate activity matrix, perform the decoding and calculate results from it
                    testMatrix = dfTest.iloc[:, 12:] # Activity for all cells for test timestamps, to be binned
                    
                    # Perform the decoding and calculate decoding error
                    activityMatrix, likelihoodMatrix, truePos, decodingPos, decodingError, speedIdx = run_decoder(testMatrix, binFrame, trainData, dfTest)
                    
                    # Decoding accuracy with speed filter            
                    medians[maxPC][iteration, sessionNo] = np.median(decodingError[speedIdx])
                    means[maxPC][iteration, sessionNo] = np.nanmean(decodingError[speedIdx])
    
                decode_nPC['NAT'+str(sessionNo)] = {'activityMatrix': activityMatrix,
                                                        'likelihoodMatrix': likelihoodMatrix,
                                                        'truePos': truePos,
                                                        'decodingPos': decodingPos,
                                                        'decodingError': decodingError,
                                                        'speedIdx': speedIdx}
    
    # Plot decoding error as violins and boxes for each session
    if plotter == True: 
        sessionName = ['A','B','A\'']
        
        dfA, dfB, dfA2 = {}, {}, {}
        
        for key in means.keys():
            dfA[key] = means[key][:,0]
            dfB[key] = means[key][:,1]
            dfA2[key] = means[key][:,2]
        
        dfPlot = pd.DataFrame(dfA), pd.DataFrame(dfB), pd.DataFrame(dfA2)
        
        for sessionNo in range(nSessions):
            df = dfPlot[sessionNo]
            fig, ax = plt.subplots(1, 2, figsize = (10,5), sharey = True)
            sns.violinplot(data = df, ax = ax[0], palette = 'viridis')
            ax[0].spines[['top', 'right']].set_visible(False)
            sns.boxplot(data = df, ax = ax[1], palette = 'viridis')
            ax[1].spines[['top', 'right']].set_visible(False)
            #sns.lineplot(data = dfA.melt(), x = 'variable', y = 'value', ax = ax[2], estimator = 'mean', ci = 95)
            fig.supxlabel('Number of cells')
            fig.supylabel('Decoding error (cm)')
            fig.suptitle(sessionName[sessionNo]+': Decoding error by number of random place cells (50 it.)')
            plt.tight_layout()
        
        # The the decoding error as violins and boxes for all sessions concatenated
        dfPlotAll = {}
        for key in means.keys(): dfPlotAll[key] = means[key].flatten()
        dfPlotAll = pd.DataFrame.from_dict(dfPlotAll)
        
        fig, ax = plt.subplots(1, 2, figsize = (9,5), sharey = True)
        sns.violinplot(data = dfPlotAll, ax = ax[0], palette = 'viridis')
        ax[0].spines[['top', 'right']].set_visible(False)
        sns.boxplot(data = dfPlotAll, ax = ax[1], palette = 'viridis')
        ax[1].spines[['top', 'right']].set_visible(False)
        #sns.stripplot(data = dfPlotAll, ax = ax[1], palette = 'viridis', jitter = 0.01, alpha = 0.5)
        fig.supxlabel('Number of cells')
        fig.supylabel('Decoding error (cm)')
        fig.suptitle('Decoding error by number of random place cells (150 it.), all sessions')
        plt.tight_layout()
        
        # Plot the decoding error as swarm for all sessions
        fig, ax = plt.subplots(figsize = (5,4))
        sns.swarmplot(data = dfPlotAll, ax = ax, palette = 'viridis', size = 2)
        ax.set_xlabel('Number of cells')
        ax.set_ylabel('Decoding error (cm)')
        ax.set_title('Decoding error by number of random place cells (150 it.), all sessions')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()

    return decode_nPC, medians, means

#%% Look at the decoding accuracy when only taking nearby place cells

def decoding_nearby_PC(session_dict, decoding_params, scale, **kwargs):

    plotter = kwargs.get('plotter', False)
    
    nSessions = len(session_dict['dfNAT'])
    nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)
    
    placecell_dict = session_dict['Placecell']
    
    # Get the anatomical distance
    cell_centre = np.full([nCells, 2], np.nan)
    for cellNo in range(nCells): cell_centre[cellNo] = session_dict['ExperimentInformation']['CellStat'][cellNo]['med'][::-1] # x, y
    anatomicalDist = np.linalg.norm(cell_centre - cell_centre[:,None], axis=-1)/scale # Scaled by µm/pixel
    
    # Decoding by anatomical distances for a number of iterations by a random PC and it's neighbours (chosen within given dsitance)
    maxPC_arr = np.linspace(25, 200, 8, dtype = int)
    nIter = 50
    mediansD, meansD, distIter = {}, {}, {}

    for maxPC in maxPC_arr:    
        mediansD[maxPC] = np.full([nIter, nSessions], np.nan)
        meansD[maxPC] = np.full([nIter, nSessions], np.nan)
        distIter[maxPC] = np.full([nIter, nSessions], np.nan)
    
        for iteration in range(nIter):
            for sessionNo in range(nSessions):
                
                dfNAT = session_dict['dfNAT']['dfNAT'+str(sessionNo)]
                # nFramesSession = int(session_dict['ExperimentInformation']['FrameinEachSession'][sessionNo])
                nPC = placecell_dict['NAT'+str(sessionNo)][0].size
                
                if nPC >= maxPC:  
                    # Get the distance matrix for place cells in this session
                    distMatrix = anatomicalDist[:, placecell_dict['NAT'+str(sessionNo)][0]-1][placecell_dict['NAT'+str(sessionNo)][0]-1, :]
                    
                    # Get a random reference cell, get the index to the k closest neighbours and the mean distance to those
                    refCell = np.random.randint(0,nPC)
                    nearbyCells = np.argsort(distMatrix[refCell,:])[1:maxPC+1] # Pythonic index, not NATEX index
                    nearbyNX = placecell_dict['NAT'+str(sessionNo)][0][nearbyCells] # NATEX index
                    argCells = np.concatenate((nearbyNX, np.array([placecell_dict['NAT'+str(sessionNo)][0][refCell]]))) # NATEX index (nearby + ref.)
                    distIter[maxPC][iteration, sessionNo] = np.nanmean(distMatrix[refCell, np.argsort(distMatrix[refCell,:])[1:maxPC+1]])
                    
                    # Filter out the dfNAT in this session for selected place cells (neighbours + reference cell)
                    PCs = ['Deconvolved, N'+str(x) for x in argCells]
                    dfNAT = dfNAT[list(dfNAT.keys()[0:12])+PCs]
                    
                    # Calculate the headpos for the entire session, set bin_edges for ratemaps from these
                    headpos = (dfNAT[['Head_X','Head_Y']]).to_numpy()
                    
                    x_edges = np.linspace(headpos[:,0].min(), headpos[:,0].max(), nBins+1)
                    y_edges = np.linspace(headpos[:,1].min(), headpos[:,1].max(), nBins+1)
                    bin_edges = (x_edges, y_edges)
                     
                    # Split the session in two, bin by bin given by number of frames 
                    binFrame, timeBin = decoding_params['binFrame'], decoding_params['timeBin']
                    
                    # Every other binFrame datapoints go to train and tests data
                    trainIdx = np.arange(len(dfNAT) - len(dfNAT)%binFrame).reshape((len(dfNAT)//binFrame, binFrame))[::2].flatten() # Index of every two 3-by-3 datapoint
                    testIdx = np.arange(len(dfNAT) - len(dfNAT)%binFrame).reshape((len(dfNAT)//binFrame, binFrame))[1::2].flatten() # As above, but starting from the second 3-by-3 bin
                    
                    dfTrain, dfTest = dfNAT.iloc[trainIdx].copy(), dfNAT.iloc[testIdx].copy()
                    
                    # Get occupancy for the train data, and calculate rate maps for this   
                    timestampsTrain = (dfTrain.Timestamps).to_numpy()
                    headposTrain    = (dfTrain[['Head_X','Head_Y']]).to_numpy()
                    
                    trainOccupancy = op.analysis.spatial_occupancy(timestampsTrain, np.transpose(headposTrain), boxSize, bin_edges = bin_edges)[0]
                   
                    # Create a 3D matrix of rate maps: One half --> Training data
                    trainData = np.full([maxPC+1, nBins, nBins], np.nan)
                    
                    for num in range(0, maxPC+1):
                        cellNo = argCells[num]
                        signaltracking = get_signaltracking(dfTrain, cellNo, signal = 'deconvolved', speed_threshold = 2.5)
                        
                        trainTuningmap = calc_tuningmap(trainOccupancy, bin_edges[0], bin_edges[1], signaltracking, 2.5)['tuningmap']
                        trainRatemap = gaussian_filter(trainTuningmap, 1.5)
                        trainData[num] = trainRatemap
                    
                    # Scale the training data to match the frequency of a time bin    
                    trainData = trainData * timeBin
                    
                    # Calculate activity matrix, perform the decoding and calculate results from it
                    testMatrix = dfTest.iloc[:, 12:] # Activity for all cells for test timestamps, to be binned
                    
                    # Perform the decoding and calculate decoding error
                    activityMatrix, likelihoodMatrix, truePos, decodingPos, decodingError, speedIdx = run_decoder(testMatrix, binFrame, trainData, dfTest)
                    
                    # Decoding accuracy with zero likelihoodMatrix removed and speed filter           
                    mediansD[maxPC][iteration, sessionNo] = np.median(decodingError[speedIdx])
                    meansD[maxPC][iteration, sessionNo] = np.nanmean(decodingError[speedIdx])
                         
    # Plot the results
    if plotter == True:
        sessionName = ['A','B','A\'']
        
        dfAD, dfBD, dfA2D = {}, {}, {}
        
        for key in meansD.keys():
            dfAD[key] = meansD[key][:,0]
            dfBD[key] = meansD[key][:,1]
            dfA2D[key] = meansD[key][:,2]
        
        dfPlotD = pd.DataFrame(dfAD), pd.DataFrame(dfBD), pd.DataFrame(dfA2D)
        
        for sessionNo in range(nSessions):
            df = dfPlotD[sessionNo]
            fig, ax = plt.subplots(1, 2, figsize = (8,5), sharey = True)
            sns.violinplot(data = df, ax = ax[0], palette = 'viridis')
            sns.boxplot(data = df, ax = ax[1], palette = 'viridis')
            #sns.lineplot(data = dfA.melt(), x = 'variable', y = 'value', ax = ax[2], estimator = 'mean', ci = 95)
            ax[0].spines[['top', 'right']].set_visible(False)
            ax[1].spines[['top', 'right']].set_visible(False)
        
            fig.supxlabel('Number of cells')
            fig.supylabel('Decoding error (cm)')
            fig.suptitle(sessionName[sessionNo]+': Decoding error from nearby place cells (50 it.)')
            plt.tight_layout()
        
        dfPlotAllD = {}
        for key in meansD.keys(): dfPlotAllD[key] = meansD[key].flatten()
        dfPlotAllD = pd.DataFrame.from_dict(dfPlotAllD)
        
        # Plot all decoding errors for all sessions as a function of cell number
        fig, ax = plt.subplots(1, 2, figsize = (8,5), sharey = True)
        sns.violinplot(data = dfPlotAllD, ax = ax[0], palette = 'viridis')
        sns.boxplot(data = dfPlotAllD, ax = ax[1], palette = 'viridis')
        #sns.stripplot(data = dfPlotAll, ax = ax[1], palette = 'viridis', jitter = 0.01, alpha = 0.5)
        ax[0].spines[['top', 'right']].set_visible(False)
        ax[1].spines[['top', 'right']].set_visible(False)
        fig.supxlabel('Number of cells')
        fig.supylabel('Decoding error (cm)')
        fig.suptitle('Decoding error from nearby place cells (150 it.), all sessions')
        plt.tight_layout()
        
        # Plot the decoding error as a function of cell number as a swarm plot
        fig, ax = plt.subplots()
        sns.swarmplot(data = dfPlotAllD, ax = ax, palette = 'viridis', size = 2)
        ax.set_xlabel('Number of cells')
        ax.set_ylabel('Decoding error (cm)')
        ax.set_title('Decoding error from nearby place cells (150 it.), all sessions')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
            
        dfDistIter = {}
        for key in distIter.keys(): dfDistIter[key] = distIter[key].flatten()
        dfDistIter = pd.DataFrame.from_dict(dfDistIter)    
        
        # Plot the mean distance from the referance cell as a function of cell number as a violin plot
        fig, ax = plt.subplots()
        sns.violinplot(data = dfDistIter, ax = ax, palette = 'viridis')
        ax.set_xlabel('Number of cells')
        ax.set_ylabel('Distance from reference cell (cm)')
        ax.set_title('Mean distance from reference cell from decoding population')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        
    # Return the median and mean decoding error (speed filtered) per nPC with distances in distIter    
    
    return mediansD, meansD, distIter, anatomicalDist

#%% Compare n number closest place cells to n number random place cells

def decoding_nearby_PC_control(session_dict, decoding_params, anatomicalDist, scale, **kwargs):

    plotter = kwargs.get('plotter', False)
    
    nSessions = nSessions = len(session_dict['dfNAT'])
    
    placecell_dict = session_dict['Placecell']    
   
    # Get random place cells (loop over number of cells), decode based on those, note the mean anatomical distance between them
    mediansDrand, meansDrand, distIterrand = {}, {}, {}
    maxPC_arr = np.linspace(25, 200, 8, dtype = int)
    nIter = 50

    for maxPC in maxPC_arr:    
        mediansDrand[maxPC] = np.full([nIter, nSessions], np.nan)
        meansDrand[maxPC] = np.full([nIter, nSessions], np.nan)
        distIterrand[maxPC] = np.full([nIter, nSessions], np.nan)
        
        for iteration in range(nIter):
            for sessionNo in range(nSessions):
                
                dfNAT = session_dict['dfNAT']['dfNAT'+str(sessionNo)]
                nPC = placecell_dict['NAT'+str(sessionNo)][0].shape[0]
                
                if nPC >= maxPC: 
                    # Get the distance matrix for place cells in this session
                    distMatrix = anatomicalDist[:, placecell_dict['NAT'+str(sessionNo)][0]-1][placecell_dict['NAT'+str(sessionNo)][0]-1, :]
                    
                    # Get k random cells from this session to perform the decoding on
                    randCells = np.random.choice(nPC, maxPC+1, replace = False) # Python index
                    argCells = placecell_dict['NAT'+str(sessionNo)][0][randCells] # NATEX index
                    
                    # Use randCells[0] as ref. cell, compute mean distance to the rest of the cells
                    distIterrand[maxPC][iteration, sessionNo] = np.nanmean(distMatrix[randCells[0], np.argsort(distMatrix[randCells[0]])[1:maxPC+1]])
                    
                    # Filter out the dfNAT in this session for selected place cells (neighbours + reference cell)
                    PCs = ['Deconvolved, N'+str(x) for x in argCells]
                    dfNAT = dfNAT[list(dfNAT.keys()[0:12])+PCs]
                    
                    # Calculate the headpos for the entire session, set bin_edges for ratemaps from these
                    headpos = (dfNAT[['Head_X','Head_Y']]).to_numpy()
                    
                    x_edges = np.linspace(headpos[:,0].min(), headpos[:,0].max(), nBins+1)
                    y_edges = np.linspace(headpos[:,1].min(), headpos[:,1].max(), nBins+1)
                    bin_edges = (x_edges, y_edges)
                     
                    # Split the session in two, bin by bin given by number of frames 
                    binFrame, timeBin = decoding_params['binFrame'], decoding_params['timeBin']
                    
                    # Every other binFrame datapoints go to train and tests data
                    trainIdx = np.arange(len(dfNAT) - len(dfNAT)%binFrame).reshape((len(dfNAT)//binFrame, binFrame))[::2].flatten() # Index of every two 3-by-3 datapoint
                    testIdx = np.arange(len(dfNAT) - len(dfNAT)%binFrame).reshape((len(dfNAT)//binFrame, binFrame))[1::2].flatten() # As above, but starting from the second 3-by-3 bin
                    
                    dfTrain, dfTest = dfNAT.iloc[trainIdx].copy(), dfNAT.iloc[testIdx].copy()
                    
                    # Get occupancy for the train data, and calculate rate maps for this   
                    timestampsTrain = (dfTrain.Timestamps).to_numpy()
                    headposTrain    = (dfTrain[['Head_X','Head_Y']]).to_numpy()
                    
                    trainOccupancy = op.analysis.spatial_occupancy(timestampsTrain, np.transpose(headposTrain), boxSize, bin_edges = bin_edges)[0]
                   
                    # Create a 3D matrix of rate maps: One half --> Training data
                    trainData = np.full([maxPC+1, nBins, nBins], np.nan)
                    
                    for num in range(0, maxPC+1):
                        cellNo = argCells[num]
                        signaltracking = get_signaltracking(dfTrain, cellNo, signal = 'deconvolved', speed_threshold = 2.5)
                        
                        trainTuningmap = calc_tuningmap(trainOccupancy, bin_edges[0], bin_edges[1], signaltracking, 2.5)['tuningmap']
                        trainRatemap = gaussian_filter(trainTuningmap, 1.5)
                        trainData[num] = trainRatemap
                        
                    # Scale the training data to match the frequency of a time bin    
                    trainData = trainData * timeBin
                    
                    # Calculate activity matrix, perform the decoding and calculate results from it
                    testMatrix = dfTest.iloc[:, 12:] # Activity for all cells for test timestamps, to be binned
                    
                    # Perform the decoding and calculate decoding error
                    activityMatrix, likelihoodMatrix, truePos, decodingPos, decodingError, speedIdx = run_decoder(testMatrix, binFrame, trainData, dfTest)
                    
                    # Decoding accuracy with speed filter           
                    mediansDrand[maxPC][iteration, sessionNo] = np.median(decodingError[speedIdx])
                    meansDrand[maxPC][iteration, sessionNo] = np.nanmean(decodingError[speedIdx])
    
    # Plot the results      
    dfADr, dfBDr, dfA2Dr = {}, {}, {}
    
    for key in meansDrand.keys():
        dfADr[key] = meansDrand[key][:,0]
        dfBDr[key] = meansDrand[key][:,1]
        dfA2Dr[key] = meansDrand[key][:,2]
    
    dfPlotDr = pd.DataFrame(dfADr), pd.DataFrame(dfBDr), pd.DataFrame(dfA2Dr)
    
    if plotter == True: 
        sessionName = ['A','B','A\'']
        
        for sessionNo in range(nSessions):
            df = dfPlotDr[sessionNo]
            fig, ax = plt.subplots(1, 2, figsize = (10,5), sharey = True)
            sns.violinplot(data = df, ax = ax[0], palette = 'viridis')
            ax[0].spines[['top', 'right']].set_visible(False)
            sns.boxplot(data = df, ax = ax[1], palette = 'viridis')
            ax[1].spines[['top', 'right']].set_visible(False)
            fig.supxlabel('Number of cells')
            fig.supylabel('Decoding error (cm)')
            fig.suptitle(sessionName[sessionNo]+': Decoding error by number of random place cells')
            plt.tight_layout()
    
    dfPlotAllDr = {}
    for key in meansDrand.keys(): dfPlotAllDr[key] = meansDrand[key].flatten()
    dfPlotAllDr = pd.DataFrame.from_dict(dfPlotAllDr)
    
    if plotter == True: 
        
        # Plot the decoding error as a function of number of cells (all session)
        fig, ax = plt.subplots(1, 2, figsize = (10,5), sharey = True)
        sns.violinplot(data = dfPlotAllDr, ax = ax[0], palette = 'viridis')
        ax[0].spines[['top', 'right']].set_visible(False)
        sns.boxplot(data = dfPlotAllDr, ax = ax[1], palette = 'viridis')
        ax[1].spines[['top', 'right']].set_visible(False)
        # sns.stripplot(data = dfPlotAll, ax = ax[1], palette = 'viridis', jitter = 0.01, alpha = 0.5)
        fig.supxlabel('Number of cells')
        fig.supylabel('Decoding error (cm)')
        fig.suptitle('Decoding error by number of random place cells (150 it.), all sessions')
        plt.tight_layout()
    
        # Plot the decoding error as a function of number of cells (all session)
        fig, ax = plt.subplots()
        sns.swarmplot(data = dfPlotAllDr, ax = ax, palette = 'viridis', size = 2)
        ax.set_xlabel('Number of cells')
        ax.set_ylabel('Decoding error (cm)')
        ax.set_title('Decoding error random place cells (150 it.), all sessions')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        
    dfDistIterR = {}
    for key in distIterrand.keys(): dfDistIterR[key] = distIterrand[key].flatten()
    dfDistIterR = pd.DataFrame.from_dict(dfDistIterR)    
    
    if plotter == True: 
        
        # Plot the mean distance from the reference cell as a function of number of cells (all sessions)
        fig, ax = plt.subplots()
        sns.violinplot(data = dfDistIterR, ax = ax, palette = 'viridis')
        ax.set_xlabel('Number of cells')
        ax.set_ylabel('Distance from reference cell (cm)')
        ax.set_title('Mean distance from reference cell from decoding population')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
    
    # Compare the nearby cells and random cells
    dfPC = dfPlotAllDr.melt().copy()
    dfPC['Mark'] = 'Nearby'
    dfPCrand = dfPlotAllDr.melt().copy()
    dfPCrand['Mark'] = 'Shuffle'
    
    dfPCs = pd.concat([dfPC, dfPCrand])
    
    if plotter == True: 
    
        fig, ax = plt.subplots()
        sns.violinplot(data = dfPCs, x = 'variable', y = 'value', hue = 'Mark', ax = ax, palette = 'viridis', split = True)
        ax.set_xlabel('Number of cells')
        ax.set_ylabel('Decoding error (cm)')
        ax.set_title('Decoding error of nearby and random place cells (all sessions)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon = False)
        plt.tight_layout()
        
        plt.savefig('N:/axon2pmini/Illustrations/img.svg', format = 'svg') 
    
    # Do statistics on the two samples and calculate the confidence interval of the data
    tStats = pd.DataFrame()
    for tt in range(dfPlotAllDr.shape[1]): tStats = pd.concat([tStats, ttest2(dfPlotAllDr.to_numpy()[:,tt], dfPlotAllDr.to_numpy()[:,tt])])
    tStats.set_index(dfPlotAllDr.keys(), inplace = True)
    
    print(tStats)
        
    # Return the median and mean decoding error (speed filtered) per random nPC with distances in distIter     
    
    return mediansDrand, meansDrand, distIterrand, tStats   

#%% Decode just from PCs with similar rate maps (based on one reference cell) set to r > 0.5, only in session A

def decoding_by_ratemapscorr(session_dict, decoding_params, selfPC, **kwargs):
    
    plotter = kwargs.get('plotter', False) 
    
    nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)
    
    # Use a reference PC
    fig, ax = plt.subplots()
    ax.imshow(session_dict['Ratemaps']['dfNAT0']['N'+str(selfPC)])

    rVals = np.full(nCells, np.nan)
    for n in range(nCells):
        rVals[n] = pearsonr(session_dict['Ratemaps']['dfNAT0']['N'+str(selfPC)].flatten(), session_dict['Ratemaps']['dfNAT0']['N'+str(n+1)].flatten())[0]
    
    boolPC = rVals > 0.5
    print('Cells with correlation > 0,5: ' + str((np.nansum(boolPC))))
    
    # Decode
    dfNAT = session_dict['dfNAT']['dfNAT0']
    
    # Filter out the dfNAT in this session for selected place cells 
    PCs = ['Deconvolved, N'+str(x) for x in (np.where(boolPC == True)[0]+1)]
    dfNAT = dfNAT[list(dfNAT.keys()[0:12])+PCs]
    
    # rPCList = ['Deconvolved, N'+str(x) for x in rPC]
    # dfNAT = dfNAT[list(dfNAT.keys()[0:12])+rPCList]
    
    # Calculate the headpos for the entire session, set bin_edges for ratemaps from these
    headpos = (dfNAT[['Head_X','Head_Y']]).to_numpy()
    
    x_edges = np.linspace(headpos[:,0].min(), headpos[:,0].max(), nBins+1)
    y_edges = np.linspace(headpos[:,1].min(), headpos[:,1].max(), nBins+1)
    bin_edges = (x_edges, y_edges)
     
    # Split the session in two, bin by bin given by number of frames 
    binFrame, timeBin = decoding_params['binFrame'], decoding_params['timeBin']
    
    # Every other binFrame datapoints go to train and tests data
    trainIdx = np.arange(len(dfNAT) - len(dfNAT)%binFrame).reshape((len(dfNAT)//binFrame, binFrame))[::2].flatten() # Index of every two 3-by-3 datapoint
    testIdx = np.arange(len(dfNAT) - len(dfNAT)%binFrame).reshape((len(dfNAT)//binFrame, binFrame))[1::2].flatten() # As above, but starting from the second 3-by-3 bin
    
    dfTrain, dfTest = dfNAT.iloc[trainIdx].copy(), dfNAT.iloc[testIdx].copy()
    
    # Get occupancy for the train data, and calculate rate maps for this   
    timestampsTrain = (dfTrain.Timestamps).to_numpy()
    headposTrain    = (dfTrain[['Head_X','Head_Y']]).to_numpy()
    
    trainOccupancy = op.analysis.spatial_occupancy(timestampsTrain, np.transpose(headposTrain), boxSize, bin_edges = bin_edges)[0]
    
    # Create a 3D matrix of rate maps: One half --> Training data
    trainData = np.full([len(PCs), nBins, nBins], np.nan)
    # trainData = np.full([len(rPCList), nBins, nBins], np.nan)

    for num in range(0, len(PCs)):
    # for num, cellNo in enumerate(rPC):    
        cellNo = (np.where(boolPC == True)[0][num])+1
        signaltracking = get_signaltracking(dfTrain, cellNo, signal = 'deconvolved', speed_threshold = 2.5)
        
        trainTuningmap = calc_tuningmap(trainOccupancy, bin_edges[0], bin_edges[1], signaltracking, 2.5)['tuningmap']
        trainRatemap = gaussian_filter(trainTuningmap, 1.5)
        trainData[num] = trainRatemap
    
    # Scale the training data to match the frequency of a time bin    
    trainData = trainData * timeBin
    
    # Calculate activity matrix, perform the decoding and calculate results from it
    testMatrix = dfTest.iloc[:, 12:] # Activity for all cells for test timestamps, to be binned
    
    # Perform the decoding and calculate decoding error
    activityMatrix, likelihoodMatrix, truePos, decodingPos, decodingError, speedIdx = run_decoder(testMatrix, binFrame, trainData, dfTest)
    
    decode_ratemapcorr = {'activityMatrix': activityMatrix,
                          'likelihoodMatrix': likelihoodMatrix,
                          'truePos': truePos,
                          'decodingPos': decodingPos,
                          'decodingError': decodingError}
            
    # Calculate a "summed" rate map
    ratemaps = np.ma.zeros([len(PCs), 32, 32])
    # ratemaps = np.ma.zeros([len(rPC), 32, 32])
    
    for n in range(np.nansum(boolPC)):
    # for n, N in enumerate(rPC):   
        N = str(np.where(boolPC == True)[0][n]+1)
        ratemaps[n] = session_dict['Ratemaps']['dfNAT0']['N'+N]
            
    summedMap = np.nansum(ratemaps, axis = 0)

    if plotter == True:
        # Decoding accuracy speed filtered
        start, stop = 0, 300
        plot_decodingAccuracy(truePos, decodingPos, speedIdx, start, stop, timeBin, 'Decoding accuracy', save = False)
        
        # plt.savefig('N:/axon2pmini/Article/Figures/Figure 5/fig5I_correlated_decoding_example1.svg', format = 'svg') 
        
        # Decoding error speed filtered
        fig, ax = plt.subplots(1,2, figsize = (10,5))
        sns.histplot(data = decodingError[speedIdx], bins = 50, ax = ax[0], 
                     color = 'gray', fill = True, kde = False, edgecolor='gray')
        ax[0].vlines(np.median(decodingError[speedIdx]), 0, np.histogram(decodingError[speedIdx], 50)[0].max(), 
              color = contrast[210], label = 'Median: ' + str(round(np.median(decodingError[speedIdx]),1)))
        ax[0].vlines(np.nanmean(decodingError[speedIdx]), 0, np.histogram(decodingError[speedIdx], 50)[0].max(), 
              color = contrast[150], label = 'Mean: ' + str(round(np.nanmean(decodingError[speedIdx]),1)))
        ax[0].legend(frameon = False)
        ax[0].set_xlim(0)
        ax[0].set_xlabel('Decoding error (cm)')
        ax[0].set_ylabel('Counts')
        ax[0].set_title('Decoding accuracy from correlated PCs, n = '+str(len(PCs)))
        ax[0].spines[['top', 'right']].set_visible(False)
    
        sns.heatmap(data = summedMap, ax = ax[1], cmap = 'viridis', cbar = False, mask = summedMap.mask, 
                           square = True, robust = True, xticklabels = False, yticklabels = False)
        ax[1].axis('off') 
        ax[1].set_title('Summed ratemap of correlated cells') 
        plt.tight_layout()  

        # plt.savefig('N:/axon2pmini/Article/Figures/Figure 5/fig5I_correlated_decoding_example2.svg', format = 'svg') 

    # Return the decoding results and the "summed" ratemap for the cells used to decode

    return decode_ratemapcorr, summedMap

#%% From A, decode the position in A'

def decoding_A_to_A2_nCells(session_dict, decoding_params, **kwargs):
    
    plotter = kwargs.get('plotter', False) 
    
    nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)

    # Define bins
    binFrame, timeBin = decoding_params['binFrame'], decoding_params['timeBin']
    
    # Create the train data, which is the ratemaps from session A, and scale it to the time per bin (ratemaps are in Hz)
    trainData = np.full([nCells, nBins, nBins], np.nan)
    
    keys = list(session_dict['Ratemaps']['dfNAT0'].keys())
    for x in range(len(keys)): trainData[x] = session_dict['Ratemaps']['dfNAT0'][keys[x]]
    
    trainData = trainData * timeBin
        
    # Calculate the activity matrix
    dfTest = session_dict['dfNAT']['dfNAT2']
    testMatrix = dfTest.iloc[:, 15:dfTest.shape[1]:4] # Activity for all cells for test timestamps, to be binned
    
    # Perform the decoding and calculate decoding error
    activityMatrix, likelihoodMatrix, truePos, decodingPos, decodingError, speedIdx = run_decoder(testMatrix, binFrame, trainData, dfTest)
    
    decode_AA2_nCells = {'activityMatrix': activityMatrix,
                         'likelihoodMatrix': likelihoodMatrix,
                         'truePos': truePos,
                         'decodingPos': decodingPos,
                         'decodingError': decodingError,
                         'speedIdx': speedIdx}
    
    if plotter == True:
        # Plot the decoding results
        timeBinNo = 0
        
        # Plot the log likelihood matrix for this time bin
        y_ind, x_ind = np.unravel_index(np.argmax(likelihoodMatrix[timeBinNo]), (32,32)) # Row, column
        
        plt.figure()
        plt.imshow(likelihoodMatrix[timeBinNo])
        plt.scatter(x_ind, y_ind, marker = 'x', color = 'r')
        plt.title('Likelihood matrix')
        
        # Plot the trajectory for the test data with true and decoded position
        fig, ax = plt.subplots()
        ax.set_title('Decoded vs. true position (A vs. A\')')
        ax.set_aspect('equal')
        ax.plot(truePos[:,0], truePos[:,1], color = 'grey', zorder = 0)
        ax.scatter(decodingPos[timeBinNo][0], decodingPos[timeBinNo][1], marker = 'x', color = 'r', zorder = 1, label = 'Decoded')
        ax.scatter(truePos[timeBinNo][0], truePos[timeBinNo][1], marker='o', color = 'b', zorder = 1, label = 'True')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon = False)
        plt.tight_layout()
        
        # Plot the distribution of the decoding error (with and without speed filtering)
        plot_decodingError(decodingError, 'Decoding error (A vs. A\')', save = False)
        plot_decodingError(decodingError[speedIdx], 'Decoding error (A vs. A\')', save = True)
        
        # Plot the decoding error and speed together
        if decodingError.shape[0] == dfTest['Body_speed'][::binFrame].to_numpy().shape[0]:
            stats = pearsonr(decodingError, dfTest['Body_speed'][::binFrame].to_numpy())
        else: 
            nDel = dfTest.shape[0]%binFrame
            dfTest = dfTest.iloc[0:len(dfTest)-nDel]
            stats = pearsonr(decodingError, dfTest['Body_speed'][::binFrame][0:decodingError.shape[0]].to_numpy())
        
        fig, ax = plt.subplots(figsize = (14,2))
        ax.plot(decodingError[0:500], color = palette[100], label = 'Decoding error')
        ax.plot(dfTest['Body_speed'][::binFrame].to_numpy()[0:500], color = contrast[190], label = 'Speed')
        ax.legend(ncol = 1, frameon = False)
        ax.set_title('R = '+str(round(stats[0],2))+', p = '+str(round(stats[1],20)))
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim([0,500])
        
        # Plot the decoding accuracy of x and y separately
        start, stop = 0, 300
        plot_decodingAccuracy(truePos, decodingPos, speedIdx, start, stop, timeBin, 'Decoding accuracy A vs. A\'', save = True)
        
        # Plot the decoding error and speed together
        stats = pearsonr(decodingError[speedIdx], dfTest['Body_speed'][::binFrame].to_numpy()[speedIdx])
        
        fig, ax = plt.subplots(figsize = (14,2))
        ax.plot(decodingError[speedIdx][0:500], color = palette[100], label = 'Decoding error')
        ax.plot(dfTest['Body_speed'][::binFrame].to_numpy()[speedIdx][0:500], color = contrast[190], label = 'Speed')
        ax.set_xlim([0,500])
        ax.legend(ncol = 2, bbox_to_anchor=(0.5, 0.7, 0.5, 0.5), frameon = False)
        ax.set_title('R = '+str(round(stats[0],2))+', p = '+str(round(stats[1],20)))
        ax.spines[['top', 'right']].set_visible(False)

    return decode_AA2_nCells
    
#%% From A, decode the position in A' using only place cells

def decoding_A_to_A2_PC(session_dict, decoding_params, **kwargs):
    
    plotter = kwargs.get('plotter', False) 
    
    # Define bins
    binFrame, timeBin = decoding_params['binFrame'], decoding_params['timeBin']
 
    placecells = session_dict['Placecell']['NAT0'][0]

    # Create the train data, which is the ratemaps from session A, and scale it to the time per bin (ratemaps are in Hz)
    trainData = np.full([placecells.size, nBins, nBins], np.nan)
    
    for x in range(placecells.size): 
        PC = placecells[x]
        trainData[x] = session_dict['Ratemaps']['dfNAT0']['N'+str(PC)]
    
    trainData = trainData * timeBin
        
    # Calculate the activity matrix
    dfTest = session_dict['dfNAT']['dfNAT2']
    
    keys = []
    for x in placecells: keys.append('Deconvolved, N'+str(x))
    testMatrix = dfTest[keys] # Activity for all cells for test timestamps, to be binned
    
    # Perform the decoding and calculate decoding error
    activityMatrix, likelihoodMatrix, truePos, decodingPos, decodingError, speedIdx = run_decoder(testMatrix, binFrame, trainData, dfTest)
    
    decode_AA2_PC = {'activityMatrix': activityMatrix,
                     'likelihoodMatrix': likelihoodMatrix,
                     'truePos': truePos,
                     'decodingPos': decodingPos,
                     'decodingError': decodingError,
                     'speedIdx': speedIdx}
    
    if plotter == True:
        # Plot the log likelihood matrix for this time bin
        timeBinNo = 0
        
        y_ind, x_ind = np.unravel_index(np.argmax(likelihoodMatrix[timeBinNo]), (32,32)) # Row, column
        
        plt.figure()
        plt.imshow(likelihoodMatrix[timeBinNo])
        plt.scatter(x_ind, y_ind, marker = 'x', color = 'r')
        plt.title('Likelihood matrix')
        
        # Plot the trajectory for the test data with true and decoded position
        fig, ax = plt.subplots()
        ax.set_title('Decoded vs. true position (A vs. A\')')
        ax.set_aspect('equal')
        ax.plot(truePos[:,0], truePos[:,1], color = 'grey', zorder = 0)
        ax.scatter(decodingPos[timeBinNo][0], decodingPos[timeBinNo][1], marker = 'x', color = 'r', zorder = 1, label = 'Decoded')
        ax.scatter(truePos[timeBinNo][0], truePos[timeBinNo][1], marker='o', color = 'b', zorder = 1, label = 'True')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon = False)
        plt.tight_layout()
        
        # Plot the distribution of the decoding error (with and without speed filtering)
        plot_decodingError(decodingError, 'Decoding error (A vs. A\')', save = False)
        plot_decodingError(decodingError[speedIdx], 'Decoding error (A vs. A\')', save = True)
        
        # Plot the decoding error and speed together
        if decodingError.shape[0] == dfTest['Body_speed'][::binFrame].to_numpy().shape[0]:
            stats = pearsonr(decodingError, dfTest['Body_speed'][::binFrame].to_numpy())
        else: 
            nDel = dfTest.shape[0]%binFrame
            dfTest = dfTest.iloc[0:len(dfTest)-nDel]
            stats = pearsonr(decodingError, dfTest['Body_speed'][::binFrame][0:decodingError.shape[0]].to_numpy())
        
        fig, ax = plt.subplots(figsize = (14,2))
        ax.plot(decodingError[0:500], color = palette[100], label = 'Decoding error')
        ax.plot(dfTest['Body_speed'][::binFrame].to_numpy()[0:500], color = contrast[190], label = 'Speed')
        ax.legend(ncol = 1, frameon = False)
        ax.set_title('R = '+str(round(stats[0],2))+', p = '+str(round(stats[1],20)))
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim([0,500])
        
        # Plot the decoding accuracy of x and y separately
        start, stop = 0, 300
        plot_decodingAccuracy(truePos, decodingPos, speedIdx, start, stop, timeBin, 'Decoding accuracy: A vs. A\'', save = True)
        
        # Plot the decoding error and speed together
        stats = pearsonr(decodingError[speedIdx], dfTest['Body_speed'][::binFrame].to_numpy()[speedIdx])
        
        fig, ax = plt.subplots(figsize = (14,2))
        ax.plot(decodingError[speedIdx][0:500], color = palette[100], label = 'Decoding error')
        ax.plot(dfTest['Body_speed'][::binFrame].to_numpy()[speedIdx][0:500], color = contrast[190], label = 'Speed')
        ax.set_xlim([0,500])
        ax.legend(ncol = 2, bbox_to_anchor=(0.5, 0.7, 0.5, 0.5), frameon = False)
        ax.set_title('R = '+str(round(stats[0],2))+', p = '+str(round(stats[1],20)))
        ax.spines[['top', 'right']].set_visible(False)

    return decode_AA2_PC

#%% From A, decode the position in B from all cells

def decoding_A_to_B_nCells(session_dict, decoding_params, **kwargs):
    
    plotter = kwargs.get('plotter', False) 
    
    # Define bins
    binFrame, timeBin = decoding_params['binFrame'], decoding_params['timeBin']

    nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)
    
    # Create the train data, which is the ratemaps from session A, and scale it to the time per bin (ratemaps are in Hz)
    trainData = np.full([nCells, nBins, nBins], np.nan)
    
    keys = list(session_dict['Ratemaps']['dfNAT0'].keys())
    for x in range(len(keys)): trainData[x] = session_dict['Ratemaps']['dfNAT0'][keys[x]]
    
    trainData = trainData * timeBin
        
    # Calculate the activity matrix
    dfTest = session_dict['dfNAT']['dfNAT1']
    testMatrix = dfTest.iloc[:, 15:dfTest.shape[1]:4] # Activity for all cells for test timestamps, to be binned
    
    # Perform the decoding and calculate decoding error
    activityMatrix, likelihoodMatrix, truePos, decodingPos, decodingError, speedIdx = run_decoder(testMatrix, binFrame, trainData, dfTest)
    
    decode_A_to_B_nCells = {'activityMatrix': activityMatrix,
                            'likelihoodMatrix': likelihoodMatrix,
                            'truePos': truePos,
                            'decodingPos': decodingPos,
                            'decodingError': decodingError,
                            'speedIdx': speedIdx}
    
    if plotter == True:
        
        # Plot the decoding results
        timeBinNo = 0
        
        # Plot the log likelihood matrix for this time bin
        y_ind, x_ind = np.unravel_index(np.argmax(likelihoodMatrix[timeBinNo]), (32,32)) # Row, column
        
        fig, ax = plt.subplots()
        ax.imshow(likelihoodMatrix[timeBinNo])
        ax.scatter(x_ind, y_ind, marker = 'x', color = 'r')
        ax.set_title('Likelihood matrix')
        
        # Plot the trajectory for the test data with true and decoded position
        fig, ax = plt.subplots()
        ax.set_title('Decoded vs. true position')
        ax.set_aspect('equal')
        ax.plot(truePos[:,0], truePos[:,1], color = 'grey', zorder = 0)
        ax.scatter(decodingPos[timeBinNo][0], decodingPos[timeBinNo][1], marker = 'x', color = 'r', zorder = 1, label = 'Decoded')
        ax.scatter(truePos[timeBinNo][0], truePos[timeBinNo][1], marker='o', color = 'b', zorder = 1, label = 'True')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon = False)
        
        plt.tight_layout()
        
        # Plot the distribution of the decoding error (with and without speed filtering)
        plot_decodingError(decodingError, 'Decoding error A-B', save = False)
        plot_decodingError(decodingError[speedIdx], 'Decoding error A-B (speed filtered)', save = True)
        
        # Plot the decoding error and speed together
        stats = pearsonr(decodingError, dfTest['Body_speed'][::binFrame].to_numpy())
        
        fig, ax = plt.subplots(figsize = (14,2))
        ax.plot(decodingError[0:500], color = palette[100], label = 'Decoding error')
        ax.plot(dfTest['Body_speed'][::binFrame].to_numpy()[0:500], color = contrast[190], label = 'Speed')
        ax.set_xlim([0,500])
        ax.legend(ncol = 2, bbox_to_anchor=(0.5, 0.6, 0.5, 0.5), frameon = False)
        ax.set_title('R = '+str(round(stats[0],2))+', p = '+str(round(stats[1],20)))
        ax.spines[['top', 'right']].set_visible(False)
        
        # Plot the decoding accuracy of x and y separately
        start, stop = 0, 300
        plot_decodingAccuracy(truePos, decodingPos, speedIdx, start, stop, timeBin, 'Decoding accuracy: A vs. B', save = True)
        
        # Plot the decoding error and speed together
        stats = pearsonr(decodingError[speedIdx], dfTest['Body_speed'][::binFrame].to_numpy()[speedIdx])
        
        fig, ax = plt.subplots(figsize = (14,2))
        ax.plot(decodingError[speedIdx][0:500], color = palette[100], label = 'Decoding error')
        ax.plot(dfTest['Body_speed'][::binFrame].to_numpy()[speedIdx][0:500], color = contrast[190], label = 'Speed')
        ax.set_xlim([0,500])
        ax.legend(ncol = 2, bbox_to_anchor=(0.5, 0.53, 0.54, 0.5), frameon = False)
        ax.set_title('R = '+str(round(stats[0],2))+', p = '+str(round(stats[1],20)))
        ax.spines[['top', 'right']].set_visible(False)
    
    return decode_A_to_B_nCells
    
#%% From A, decode the position in B using only place cells

def decoding_A_to_B_PC(session_dict, decoding_params, **kwargs):
    
    plotter = kwargs.get('plotter', False) 
    
    # Define bins
    binFrame, timeBin = decoding_params['binFrame'], decoding_params['timeBin']

    # Place cells
    placecells = session_dict['Placecell']['NAT0'][0]

    # Create the train data, which is the ratemaps from session A, and scale it to the time per bin (ratemaps are in Hz)
    trainData = np.full([placecells.size, nBins, nBins], np.nan)
    
    for x in range(placecells.size): 
        PC = placecells[x]
        trainData[x] = session_dict['Ratemaps']['dfNAT0']['N'+str(PC)]
    
    trainData = trainData * timeBin
        
    # Calculate the activity matrix
    dfTest = session_dict['dfNAT']['dfNAT1']
    
    keys = []
    for x in placecells: keys.append('Deconvolved, N'+str(x))
    testMatrix = dfTest[keys] # Activity for all cells for test timestamps, to be binned
    
    # Perform the decoding and calculate decoding error
    activityMatrix, likelihoodMatrix, truePos, decodingPos, decodingError, speedIdx = run_decoder(testMatrix, binFrame, trainData, dfTest)
    
    decode_A_to_B_PC = {'activityMatrix': activityMatrix,
                        'likelihoodMatrix': likelihoodMatrix,
                        'truePos': truePos,
                        'decodingPos': decodingPos,
                        'decodingError': decodingError,
                        'speedIdx': speedIdx}
    if plotter == True: 
        
        # Plot the decoding results
        timeBinNo = 0
        
        # Plot the log likelihood matrix for this time bin
        y_ind, x_ind = np.unravel_index(np.argmax(likelihoodMatrix[timeBinNo]), (32,32)) # Row, column
        
        fig, ax = plt.subplots()
        ax.imshow(likelihoodMatrix[timeBinNo])
        ax.scatter(x_ind, y_ind, marker = 'x', color = 'r')
        ax.set_title('Likelihood matrix')
        
        # Plot the trajectory for the test data with true and decoded position
        fig, ax = plt.subplots()
        ax.set_title('Decoded vs. true position')
        ax.set_aspect('equal')
        ax.plot(truePos[:,0], truePos[:,1], color = 'grey', zorder = 0)
        ax.scatter(decodingPos[timeBinNo][0], decodingPos[timeBinNo][1], marker = 'x', color = 'r', zorder = 1, label = 'Decoded')
        ax.scatter(truePos[timeBinNo][0], truePos[timeBinNo][1], marker='o', color = 'b', zorder = 1, label = 'True')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon = False)
        
        plt.tight_layout()
        
        # Plot the distribution of the decoding error (with and without speed filtering)
        plot_decodingError(decodingError, 'Decoding error A-B', save = False)
        plot_decodingError(decodingError[speedIdx], 'Decoding error A-B (speed filtered)', save = True)
        
        # Plot the decoding error and speed together
        stats = pearsonr(decodingError, dfTest['Body_speed'][::binFrame].to_numpy())
        
        fig, ax = plt.subplots(figsize = (14,2))
        ax.plot(decodingError[0:500], color = palette[100], label = 'Decoding error')
        ax.plot(dfTest['Body_speed'][::binFrame].to_numpy()[0:500], color = contrast[190], label = 'Speed')
        ax.set_xlim([0,500])
        ax.legend(ncol = 2, bbox_to_anchor=(0.5, 0.6, 0.5, 0.5), frameon = False)
        ax.set_title('R = '+str(round(stats[0],2))+', p = '+str(round(stats[1],20)))
        ax.spines[['top', 'right']].set_visible(False)
        
        # Plot the decoding accuracy of x and y separately
        start, stop = 0, 300
        plot_decodingAccuracy(truePos, decodingPos, speedIdx, start, stop, timeBin, 'Decoding accuracy: A vs. B', save = True)
        
        # Plot the decoding error and speed together
        stats = pearsonr(decodingError[speedIdx], dfTest['Body_speed'][::binFrame].to_numpy()[speedIdx])
        
        fig, ax = plt.subplots(figsize = (14,2))
        ax.plot(decodingError[speedIdx][0:500], color = palette[100], label = 'Decoding error')
        ax.plot(dfTest['Body_speed'][::binFrame].to_numpy()[speedIdx][0:500], color = contrast[190], label = 'Speed')
        ax.set_xlim([0,500])
        ax.legend(ncol = 2, bbox_to_anchor=(0.5, 0.53, 0.54, 0.5), frameon = False)
        ax.set_title('R = '+str(round(stats[0],2))+', p = '+str(round(stats[1],20)))
        ax.spines[['top', 'right']].set_visible(False)

    return decode_A_to_B_PC

#%% Compare the effect of novelty vs. a stable representation in CA1

def decoding_novelty(session_dict, decoding_params, **kwargs):

    plotter = kwargs.get('plotter', False) 
    
    nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)
    nSessions = len(session_dict['dfNAT'])
    sessionName = ['A','B','A\'']
        
    # Prepare data for the model
    results = {}
    decode_novelty = {}
    
    for flip in range(2):
        
        for sessionNo in range(nSessions-1): # For now, only look at the first two sessions
            
            dfNAT = session_dict['dfNAT']['dfNAT'+str(sessionNo)]
            nFramesSession = int(session_dict['ExperimentInformation']['FrameinEachSession'][sessionNo])   
            binFrame = 3 # Number of frames per time bin
            frameRate = 7.5
            timeBin = binFrame*1/frameRate
        
            # Calculate the headpos for the entire session, set bin_edges for ratemaps from these
            headpos = (dfNAT[['Head_X','Head_Y']]).to_numpy()
            
            x_edges = np.linspace(headpos[:,0].min(), headpos[:,0].max(), nBins+1)
            y_edges = np.linspace(headpos[:,1].min(), headpos[:,1].max(), nBins+1)
            bin_edges = (x_edges, y_edges)
             
            # Split the session in two, first and second halves 
            if flip == 0:
                dfTrain, dfTest = dfNAT.iloc[0:int(nFramesSession/2)], dfNAT.iloc[int(nFramesSession/2):]
            elif flip == 1:
                dfTest, dfTrain = dfNAT.iloc[0:int(nFramesSession/2)], dfNAT.iloc[int(nFramesSession/2):]
                
            # Get occupancy for the train data, and calculate rate maps for this   
            timestampsTrain = (dfTrain.Timestamps).to_numpy()
            headposTrain    = (dfTrain[['Head_X','Head_Y']]).to_numpy()
            
            trainOccupancy = op.analysis.spatial_occupancy(timestampsTrain, np.transpose(headposTrain), boxSize, bin_edges = bin_edges)[0]
           
            # Create a 3D matrix of rate maps: One half --> Training data
            trainData = np.full([nCells, nBins, nBins], np.nan)
            
            for cellNo in range(1, nCells+1):
                signaltracking = get_signaltracking(dfTrain, cellNo, signal = 'deconvolved', speed_threshold = 2.5)
                
                trainTuningmap  = calc_tuningmap(trainOccupancy, bin_edges[0], bin_edges[1], signaltracking, 2.5)['tuningmap']
                trainRatemap  = gaussian_filter(trainTuningmap, 1.5)
                trainData[cellNo-1] = trainRatemap
        
            trainData = trainData * timeBin
        
            # Get the test matrix and calculate the activity matrix
            testMatrix = dfTest.iloc[:, 15:dfTest.shape[1]:4] # Activity for all cells for test timestamps, to be binned
            
            # Perform the decoding and calculate decoding error
            activityMatrix, likelihoodMatrix, truePos, decodingPos, decodingError, speedIdx = run_decoder(testMatrix, binFrame, trainData, dfTest)
            
            decode_novelty[flip] = {'activityMatrix': activityMatrix,
                                    'likelihoodMatrix': likelihoodMatrix,
                                    'truePos': truePos,
                                    'decodingPos': decodingPos,
                                    'decodingError': decodingError,
                                    'speedIdx': speedIdx}
            
            if flip == 0: # Decoding second half by training on first half
                results['1-2-'+str(sessionName[sessionNo])] = truePos, decodingPos, decodingError
            elif flip == 1: # Decoding first half by training on second half
                results['2-1-'+str(sessionName[sessionNo])] = truePos, decodingPos, decodingError
    
        if plotter == True:         
            # Plot the results
            fig, ax = plt.subplots(2,1, figsize = (5,5), sharex = True, sharey = True)
            fig.supxlabel('Decoding error (cm)')
            fig.supylabel('Count')
            ax[0].set_title('A: Familiar')
            ax[0].hist(results['1-2-A'][2],50, color = palette[100], alpha = 0.6, label = 'Frist-Sec')
            ax[0].hist(results['2-1-A'][2],50, color = contrast[190], alpha = 0.6, label = 'Sec-First')
            ax[0].set_xlim(0)
            ax[0].legend(ncol = 2, loc = 'best', frameon = False)
            ax[0].spines[['top', 'right']].set_visible(False)
            
            ax[1].set_title('B: Novel')
            ax[1].hist(results['1-2-B'][2],50, color = palette[100], alpha = 0.6, label = 'Frist-Sec')
            ax[1].hist(results['2-1-B'][2],50, color = contrast[190], alpha = 0.6, label = 'Sec-First')
            ax[1].set_xlim(0)
            ax[1].legend(ncol = 2, loc = 'best', frameon = False)
            ax[1].spines[['top', 'right']].set_visible(False)
            
            plt.tight_layout()
        
            # Plot the decoding results: Decoding first half by training on second half of B
            timeBinNo = 0
            
            # Plot the log likelihood matrix for this time bin
            y_ind, x_ind = np.unravel_index(np.argmax(likelihoodMatrix[timeBinNo]), (32,32)) # Row, column
            
            fig, ax = plt.subplots()
            ax.imshow(likelihoodMatrix[timeBinNo])
            ax.scatter(x_ind, y_ind, marker = 'x', color = 'r')
            ax.set_title('Likelihood matrix')
            
            # Plot the trajectory for the test data with true and decoded position
            fig, ax = plt.subplots()
            ax.set_title('Decoded vs. true position')
            ax.set_aspect('equal')
            ax.plot(truePos[:,0], truePos[:,1], color = 'grey', zorder = 0)
            ax.scatter(decodingPos[timeBinNo][0], decodingPos[timeBinNo][1], marker = 'x', color = 'r', zorder = 1, label = 'Decoded')
            ax.scatter(truePos[timeBinNo][0], truePos[timeBinNo][1], marker='o', color = 'b', zorder = 1, label = 'True')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon = False)
            
            plt.tight_layout()
            
            # Plot the distribution of the decoding error
            plot_decodingError(decodingError, 'Decoding error', save = False)
            plot_decodingError(decodingError[speedIdx], 'Decoding error (speed filter)', save = False)
            
            # Plot the decoding error and speed together
            stats = pearsonr(decodingError, dfTest['Body_speed'][::binFrame].to_numpy())
            
            fig, ax = plt.subplots(figsize = (14,2))
            ax.plot(decodingError[0:500], color = palette[100], label = 'Decoding error')
            ax.plot(dfTest['Body_speed'][::binFrame].to_numpy()[0:500], color = contrast[190], label = 'Speed')
            ax.legend(ncol = 2, frameon = False)
            ax.set_title('R = '+str(round(stats[0],2))+', p = '+str(round(stats[1],20)))
            ax.set_xlim([0,500])
            ax.spines[['top', 'right']].set_visible(False)
            
            # Plot the decoding accuracy of x and y separately
            start, stop, titleStr = 0, 300, 'Decoding accuracy '+sessionName[sessionNo]+' vs. '+ sessionName[sessionNo]
            plot_decodingAccuracy(truePos, decodingPos, speedIdx, start, stop, timeBin, titleStr, save = False)
            
            # Plot the decoding error and speed together
            stats = pearsonr(decodingError[speedIdx], dfTest['Body_speed'][::binFrame].to_numpy()[speedIdx])
            
            fig, ax = plt.subplots(figsize = (14,2))
            ax.plot(decodingError[speedIdx][0:500], color = palette[100], label = 'Decoding error')
            ax.plot(dfTest['Body_speed'][::binFrame].to_numpy()[speedIdx][0:500], color = contrast[190], label = 'Speed')
            ax.set_xlim([0,500])
            ax.legend(ncol = 2, bbox_to_anchor=(0.5, 0.75, 0.5, 0.5), frameon = False)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_title('R = '+str(round(stats[0],2))+', p = '+str(round(stats[1],20)))
        
    return decode_novelty, results

#%% Initiate data and variables

scale = 1.1801 # Pixels/µm --> 1 pixel = 0.847.. µm
binning = 2.5 # cm/bin in ratemap
boxSize = 80 # cm
nBins = int(boxSize/binning)

decoding_params = {'binFrame': 3, # Number of frames per time bin
                   'frameRate': 7.5, # Sampling frame rate
                   'timeBin': 3*1/7.5} # decoding_params['binFrame']*1/decoding_params['frameRate']}

palette = sns.color_palette("viridis", 256).as_hex()
contrast = sns.color_palette('OrRd', 256).as_hex()

#%% Loop over sessions

results_folder = r'C:\Users\torstsl\Projects\axon2pmini\results'

with open(results_folder+'/sessions_overview.txt') as f:
    sessions = f.read().splitlines() 
f.close()

decoding_dict = {}

if __name__ == "__main__":
    
    for session in sessions: 
        
        # Load session_dict
        session_dict = pickle.load(open(session+'\session_dict.pickle','rb'))
        print('Successfully loaded session_dict from: '+str(session))
        
        key = session_dict['Animal_ID'] + '-' + session_dict['Date']
       
        # Test decoding from all cells and place cells in A
        decode_allCell_A, decode_PC_A = decoding_tester_A(session_dict, decoding_params, plotter = False)
        
        # Decode from 25 random PCs in A, not usefull if 'decode_from_nPCs' is ran
        # decode_25PC = decoding_from_25_placecells(session_dict, decoding_params, plotter = False)
        
        # Decode from a range of random PCs
        decode_nPC, medians, means = decoding_from_nPCs(session_dict, decoding_params, plotter = False)
        
        # Decoding error by n neares PCs, with distance given for them in distIter
        mediansD, meansD, distIter, anatomicalDist = decoding_nearby_PC(session_dict, decoding_params, scale, plotter = False)
        
        # Decoding error by n random PCs, with distance given for them in distIterrand
        mediansDrand, meansDrand, distIterrand, tStats = decoding_nearby_PC_control(session_dict, decoding_params, anatomicalDist, scale, plotter = False)
        
        # Decode based only on PC witht a rate map of correlation > 0.5 (based on one reference PC (selfPC))
        selfPC = 3 # NATEX ID
        decode_ratemapcorr, summedMap = decoding_by_ratemapscorr(session_dict, decoding_params, selfPC, plotter = False)
        
        # From data in A, decode position in A' based on all cells
        decode_AA2_nCells = decoding_A_to_A2_nCells(session_dict, decoding_params, plotter = False)
        
        #  From data in A, decode position in A' based on only place cells
        decode_AA2_PC = decoding_A_to_A2_PC(session_dict, decoding_params, plotter = False)
        
        # From data in A, decode position in B based on all cells 
        decode_A_to_B_nCells = decoding_A_to_B_nCells(session_dict, decoding_params, plotter = False)
        
        #  From data in A, decode position in B based on only place cells
        decode_A_to_B_PC = decoding_A_to_B_PC(session_dict, decoding_params, plotter = False)
   
        # Put the results in a dict
        decoding_dict[key] = {'decode_allCell_A': decode_allCell_A,
                              'decode_PC_A': decode_PC_A,
                              'decode_nPC': decode_nPC,
                              'medians': medians,
                              'means': means,
                              'mediansD': mediansD,
                              'meansD': meansD,
                              'distIter': distIter,
                              'mediansDrand': mediansDrand,
                              'meansDrand': meansDrand,
                              'distIterrand': distIterrand, 
                              'tStats': tStats,
                              'decode_ratemapcorr': decode_ratemapcorr,
                              'summedMap': summedMap,
                              'decode_AA2_nCells': decode_AA2_nCells,
                              'decode_AA2_PC': decode_AA2_PC,
                              'decode_A_to_B_nCells': decode_A_to_B_nCells,
                              'decode_A_to_B_PC': decode_A_to_B_PC}
        
        # Store the output
        with open(results_folder+'/decoding_dict.pickle','wb') as handle:
            pickle.dump(decoding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Successfully saved results_dict in '+ results_folder)
  
#%% Decode novelty

results_folder = r'C:\Users\torstsl\Projects\axon2pmini\results'

# Novelty in B for these sessions
dirs = [r'N:/axon2pmini/Recordings/100867/190822',
        r'N:/axon2pmini/Recordings/101258/180822',
        r'N:/axon2pmini/Recordings/102121/260922',        
        r'N:/axon2pmini/Recordings/102123/051022', 
        r'N:/axon2pmini/Recordings/102124/280922']

decoding_novelty_dict = {}

if __name__ == "__main__":
    
    for session in sessions: 
        
        # Load session_dict
        session_dict = pickle.load(open(session+'\session_dict.pickle','rb'))
        print('Successfully loaded session_dict from: '+str(session))
        
        key = session_dict['Animal_ID'] + '-' + session_dict['Date']
       
        decode_novelty, results = decoding_novelty(session_dict, decoding_params, plotter = False)
        
        # Put the results in a dict
        decoding_novelty_dict[key] = {'decode_novelty': decode_novelty,
                                      'results': results}
        
        # Store the output
        with open(results_folder+'/decoding_novelty_dict.pickle','wb') as handle:
            pickle.dump(decoding_novelty_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Successfully saved results_dict in '+ results_folder)     

