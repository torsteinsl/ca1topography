# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:30:54 2023

@author: torstsl

Bayesian decoding of position, performed on CA1 place cell data. Used to see
whether or not the decoding is better or worse when looking at anatomically 
close by place cells compared to random place cells. 

This script loads a session, and then prepares it for running the decoder. The 
sessions are split in half by odds and even bins, trained on one half and
tested on the other half. 

The parameters for the decoding are set: distances, nRefCells and nShuffles

For the actual decoding, first a random PC from that session is chosen. Then, 
for a given anatomical distance, all PCs within that distance are picked out. 
Decoding is performed on this population of cells. As a control, as many cells
that were used to decode are picked randomly together with the set reference 
cell (first picked cell). For each iteration, the decoding error is stored. 
At last, the decoding errors between the true distribution and the control 
shuffled distribution are compared. The results are plotted. 

"""
    
import pickle
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import opexebo as op
import pandas as pd
import concurrent.futures as cf
from tqdm import tqdm
from src.cell_activity_analysis import get_signaltracking, calc_tuningmap, calc_activityMatrix, run_decoder, ttest2
from scipy.ndimage import gaussian_filter
    
#%% Function to perform the decoding for all sessions

def decode_topography(session_dict, params):    
    
    placecell_dict = session_dict['Placecell']
    
    nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)
    nSessions = len(session_dict['dfNAT'])
    
    # Prepare an anatomical distance matrix
    cell_centre = np.full([nCells, 2], np.nan)
    for cellNo in range(nCells): cell_centre[cellNo] = session_dict['ExperimentInformation']['CellStat'][cellNo]['med'][::-1] # x, y
    anatomicalDist = np.linalg.norm(cell_centre - cell_centre[:,None], axis=-1)/scale # Scaled by µm/pixel
    
    # Prepare decoding: Calculate all train data rate maps once, so that they can be indexed directly later
    
    # Set decoding variables
    binFrame = 3 # Number of frames per time bin
    frameRate = 7.5
    timeBin = binFrame*1/frameRate
    
    speedThreshold = 2.5 #cm/s
    
    trainData_dict = {} # To be indexed later
    activityMatrix_dict = {} # To be indexed later
    
    dfHDs = {}
    speeds = {}
    
    for sessionNo in range(nSessions-1): # Only do box A and B
        
        dfNAT = session_dict['dfNAT']['dfNAT'+str(sessionNo)]
        
        # Calculate the headpos for the entire session, set bin_edges for ratemaps from these
        headpos = (dfNAT[['Head_X','Head_Y']]).to_numpy()
        
        x_edges = np.linspace(headpos[:,0].min(), headpos[:,0].max(), nBins+1)
        y_edges = np.linspace(headpos[:,1].min(), headpos[:,1].max(), nBins+1)
        bin_edges = (x_edges, y_edges)
        
        # Every other binFrame datapoints go to train and tests data
        trainIdx = np.arange(len(dfNAT) - len(dfNAT)%binFrame).reshape((len(dfNAT)//binFrame, binFrame))[::2].flatten() # Index of every two 3-by-3 datapoint
        testIdx = np.arange(len(dfNAT) - len(dfNAT)%binFrame).reshape((len(dfNAT)//binFrame, binFrame))[1::2].flatten() # As above, but starting from the second 3-by-3 bin
        
        dfTrain, dfTest = dfNAT.iloc[trainIdx].copy(), dfNAT.iloc[testIdx].copy()
        dfHDs['NAT'+str(sessionNo)] = dfTest[['Head_X', 'Head_Y']]
        
        # Get occupancy for the train data, and calculate rate maps for this   
        timestampsTrain = (dfTrain.Timestamps).to_numpy()
        headposTrain    = (dfTrain[['Head_X','Head_Y']]).to_numpy()
        
        trainOccupancy = op.analysis.spatial_occupancy(timestampsTrain, np.transpose(headposTrain), boxSize, bin_edges = bin_edges)[0]
        
        # calcualte the training data rate maps
        trainData = np.full([nCells, nBins, nBins], np.nan)
        
        for cellNo in range(0, nCells):
            signaltracking = get_signaltracking(dfTrain, cellNo+1, signal = 'deconvolved', speed_threshold = 2.5)
            
            trainTuningmap = calc_tuningmap(trainOccupancy, bin_edges[0], bin_edges[1], signaltracking, 2.5)['tuningmap']
            trainRatemap = gaussian_filter(trainTuningmap, 1.5)
            trainRatemap = np.ma.MaskedArray(trainRatemap, trainTuningmap.mask)
            
            trainData[cellNo] = trainRatemap
    
        # Scale the training data to match the frequency of a time bin    
        trainData = trainData * timeBin
        
        trainData_dict['NAT'+str(sessionNo)] = trainData
        
        # Calculate test data activity matrix
        testMatrix = dfTest.iloc[:, 15:dfTest.shape[1]:4] # Activity for all cells for test timestamps, to be binned
        
        activityMatrix = calc_activityMatrix(testMatrix, binFrame)
        
        activityMatrix_dict['NAT'+str(sessionNo)] = activityMatrix
        
        # Get the instances per time bin that are above a speed threshold
        speeds['NAT'+str(sessionNo)] = dfTest['Body_speed'][::binFrame].to_numpy() > speedThreshold
    
    # Decode on close by place cells, and do a control shuffling for eactested cell

    # Set random PC
    # Find the nearby PCs to that cell by a distance criterion. Not number of neighbours
    # Do decoding and calculate the decoding error for this subset 
    # Shuffle: Pick as many random PCs as you got neighbours (and the ref. cell), and to the decoding again
    # Repeat several times. Note the decoding error. 
    # Compare the decoding error to the true distribution and to the shuffled control distribution
    
    distances = params['distances']
    nRefCells = params['nRefCells']
    nShuffles = params['nShuffles']
   
    # distances = np.linspace(25, 150, 6, dtype = int)
    # nRefCells = 200
    # nShuffles = 30
    
    numWorkers = 14
    
    medians, means, cohort = {}, {}, {}
    mediansShuffle, meansShuffle = {}, {}
    
    for minDist in distances: # Iterate over minimum instances from reference cell
        medians[minDist], means[minDist] = np.full([nRefCells, nSessions-1], np.nan), np.full([nRefCells, nSessions-1], np.nan)
        mediansShuffle[minDist], meansShuffle[minDist] = {}, {}
        cohort[minDist] = np.full([nRefCells, nSessions-1], np.nan)
        
        for k in range(nRefCells): 
            meansShuffle[minDist][k] = np.full([nShuffles, nSessions-1], np.nan)
            mediansShuffle[minDist][k] = np.full([nShuffles, nSessions-1], np.nan)
           
        for sessionNo in range(nSessions-1): # Iterate over sessions A and B
            nPC = placecell_dict['NAT'+str(sessionNo)][0].shape[0]
            dfHD = dfHDs['NAT'+str(sessionNo)]
            speedIdx = speeds['NAT'+str(sessionNo)]
            
            # Start parallelisation of analyses in a cell wise matter
            futures = []
            
            parameters = {'nPC': nPC,
                          'sessionNo': sessionNo,
                          'binFrame': binFrame,
                          'nShuffles': nShuffles,
                          'minDist': minDist}
            
            with cf.ProcessPoolExecutor(max_workers=numWorkers) as pool:
                for cellNo in range(nRefCells):
                    futures.append(pool.submit(
                        run_decoder,            # The function
                        trainData_dict,         # All training data
                        activityMatrix_dict,    # All activity matrices
                        dfHD,                   # DataFrame with the test data (headpos is used)
                        placecell_dict,         # All NATEX indexes for place cells
                        anatomicalDist,         # Matrix of anatomical distance between all cells
                        speedIdx,               # Boolean for each timebin, True if speed is > 2.5
                        parameters,             # Parameter dictionary with the following:
                            #.nPC,              # Number of place cells in this session
                            #.sessionNo,        # Current session numb(iterated over)
                            #.binFrame,         # Number of frames per time bin in decoding
                        cellNo                  # Cell number, the iterable
                    ))     
            
                for future in tqdm(cf.as_completed(futures), total=nRefCells):
                    jobNo = future.result()[0]
                    medians[minDist][jobNo, sessionNo] = future.result()[1]
                    means[minDist][jobNo, sessionNo] = future.result()[2]
                    mediansShuffle[minDist][jobNo][:, sessionNo] = future.result()[3]
                    meansShuffle[minDist][jobNo][:, sessionNo] = future.result()[4]
                    cohort[minDist][jobNo, sessionNo] = future.result()[5]
                    
        print('\nCompleted for cells within distance: '+str(minDist)+'\n')
    
    return medians, means, mediansShuffle, meansShuffle, cohort
    
#%% Prepare variables
session_string = ['A', 'B', "A'"]

scale = 1.1801 # Pixels/µm --> 1 pixel = 0.847.. µm
binning = 2.5 # cm/bin in ratemap
boxSize = 80 # cm
nBins = int(boxSize/binning)

palette = sns.color_palette("viridis", 256).as_hex()
contrast = sns.color_palette('OrRd', 256).as_hex()

#%% Loop over sessions

results_folder = r'C:\Users\torstsl\Projects\axon2pmini\results'

with open(results_folder+'/sessions_overview.txt') as f:
    sessions = f.read().splitlines() 
f.close()

decoding_topography_dict = {}

if __name__ == "__main__":
    
    for session in sessions: 
        
        # Load session_dict
        session_dict = pickle.load(open(session+'\session_dict.pickle','rb'))
        print('Successfully loaded session_dict from: '+str(session))
        
        key = session_dict['Animal_ID'] + '-' + session_dict['Date']
        
        params = {'distances': np.linspace(25, 150, 6, dtype = int),
                  'nRefCells': 200,
                  'nShuffles': 25}
        
        medians, means, mediansShuffle, meansShuffle, cohort = decode_topography(session_dict, params)
        
        # Put the results in a dict 
        decoding_topography_dict[key] = {'medians': medians, 
                                         'means': means,
                                         'mediansShuffle': mediansShuffle, 
                                         'meansShuffle': meansShuffle, 
                                         'cohort': cohort,
                                         'parameters': params}
        
        # Store the output
        with open(results_folder + '/decoding_topography_dict.pickle','wb') as handle:
            pickle.dump(decoding_topography_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Successfully saved results_dict in '+ results_folder)
