# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 09:23:59 2022

@author: torstsl

PLACE CELL ANALYSIS

A script to classify place cells. Used on CA1 imaging data. Based on Weijian 
Zong's MATLAB place cell code from MINI2P published in Cell 
(https://doi.org/10.1016/j.cell.2022.02.017). 

The classification is based on a shuffling procedure, where the cells spatial
tuning properties are compared to that of a shuffled distribution. 

To be concidered for further analysis, the cell needs sufficient signal to 
noice-ratio (SNR) and a minimum of deconvolved spikes. Only cells that exheeds 
these quality measures are concidered for further analysis and hence possible 
place cells. 

Place cells are defined as follows:
    - Spatial information above 95th percentile of shuffled distribution
    - Stability above 95th percentile of shuffled distribution 
        (Pearson correlation of first and second halves of the recording)
    - At least one detactable place field, using opexebo's field detection 
        with 'sep'

The script uses three supporting functions, for (1) calculating the SI and 
stability og a cell,(2) shuffling the data from that cell nShuffle number of 
times, and (3) calculating the SIand stability for each shuffle. These are 
located in src.cell_activity_analysis as part of this package, and have their
own documentation. 
 
OUTPUT:
    placecell_dict (dict):  Defines cells per session that fulfil the above 
                            criteria in a session wise matter. Gives NATEX ID 
                            of cells that matched dfNAT. This dictionary is by
                            default put into the session_dict.pickle loaded in  
                            the beginning under the key 'Placecell'.

RUNTIME:
These calculations are computatiaonally heavy, and thus the code takes a while. 
This is reduced to some degree by parallelising the calculations. The 
efficiency gain of this is dependent on the computer hardware, and must be 
specified by the user in the script (defalult: numWorkers = 18).
    
Previously: 07.10.22:   Approx. 200 cells in 5 hours = 1.5 min per cell 
                        (500 shuffles); before changes in calc_tuningmap
            10.10.22:   Approx. 420 cells in 1 hour = 8,5 sec per cell
                        (500 shuffles); after changes in calc_tuningmap

PERFORMANCE: 31.10.22 with minEvents = 40
    Sensitivity: 0,89
    Spesificity: 0,48
    Positive predictive value: 0,67
    Negative predictive value: 0,79
"""

import numpy as np
import opexebo as op
import pandas as pd
import concurrent.futures as cf
import pickle
#import time 
from tqdm import tqdm
from src.loading_data import load_session_dict
from src.cell_activity_analysis import place_cell_classifier

#%% Initialize variables and get data
if __name__ == "__main__":
   
    # Make a parameter dictionary 
    param_dict = {'boxSize': 80,   # Square
                  'nBins': 32,     # boxSize/mapBinSize = 80/2.5 = 32   To be added in occupancy calculations
                  'speedThreshold': 2.5,
                  'minSNR': 3,
                  'minEvents': 40,
                  'minPeakField': 0.2,
                  'nShuffle': 500,
                  'minShufflingInt': 30}
    
    # Get data
    session_dict = load_session_dict()
    nSessions = int(session_dict['ExperimentInformation']['Session'])
    nCells = int(session_dict['ExperimentInformation']['TotalCell'])
    cellSNR = session_dict['ExperimentInformation']['CellSNR']
    
    print('Found '+str(nSessions)+' sessions, each with '+str(nCells)+' cells')
    #%% Filter out good cells, analyses cells for spatial information, fields 
    #   and stability, shuffle, classify cells that are place cells
    
    # Variables
    spatialInformation = np.full([nSessions,nCells],np.nan)
    correlation_r      = np.full([nSessions,nCells],np.nan)
    placefields        = [[np.nan]*nCells]*nSessions
    shuffleSIPrc       = [[np.nan]]*nSessions
    shuffleCorrPrc     = [[np.nan]]*nSessions
    placecell_dict     = {}
    
    #t = time.process_time()
    
    # Filter by 1) SNR, 2) speed, 3) events
    for sessionNo in range(nSessions):
        
        print('Analysing session '+str(sessionNo+1)+':')
        
        #session_t = time.process_time()
        
        nFramesSession = int(session_dict['ExperimentInformation']['FrameinEachSession'][sessionNo])   
        candidatePC = []
        truePC = []
        dfNAT = session_dict['dfNAT']['dfNAT'+str(sessionNo)]
        
        signalShuffle = np.full([nFramesSession,param_dict['nShuffle']],np.nan)
        shuffleSI     = np.full([param_dict['nShuffle'],nCells],np.nan)
        shuffleCorr   = np.full([param_dict['nShuffle'],nCells],np.nan)
        
        # Get occupancy per session, and calculate rate maps per cell from filtered cells
        timestamps = (dfNAT.Timestamps).to_numpy()
        headpos    = (dfNAT[['Head_X','Head_Y']]).to_numpy()
        
        occupancy, coverage_prc, bin_edges = op.analysis.spatial_occupancy(timestamps, np.transpose(headpos), 
                                                                           param_dict['boxSize'], bin_number = param_dict['nBins'])
        
        firstOccupancy = op.analysis.spatial_occupancy(timestamps[0:int(nFramesSession/2)], 
                                                       np.transpose(headpos[0:int(nFramesSession/2),:]),
                                                       param_dict['boxSize'], bin_edges = bin_edges)[0]
        secondOccupancy = op.analysis.spatial_occupancy(timestamps[int(nFramesSession/2):nFramesSession+1], 
                                                        np.transpose(headpos[int(nFramesSession/2):nFramesSession+1,:]),
                                                        param_dict['boxSize'], bin_edges = bin_edges)[0]
        
        # Filter out cells with too low SNR    
        SNR_Cells = np.where(cellSNR > param_dict['minSNR'])[0]+1 # Gives array with NATEX output cell ID
        
        # Initiate dataframe for cell, add tracking
        dfCell = pd.DataFrame()
        dfCell['Head_X'], dfCell['Head_Y'] = headpos[:,0], headpos[:,1]
               
        # Start parallelisation of analyses in a cell wise matter
        numWorkers = 12
        futures = []
        
        with cf.ProcessPoolExecutor(max_workers=numWorkers) as pool:
            for cellNo in range(1,nCells+1):
                futures.append(pool.submit(
                    place_cell_classifier,  # The function
                    dfNAT,                  # The tracking and signal for all cells
                    dfCell,                 # The tracking without signal
                    nCells,                 # The number of cells
                    SNR_Cells,              # Cells with sufficient SNR (NATEX index)
                    occupancy,              # Occupancy for all tracking
                    firstOccupancy,         # Occupancy for first half of tracking
                    secondOccupancy,        # Occupancy for second half of tracking
                    bin_edges,              # Bin edges for tracking
                    nFramesSession,         # Number of frames in the session for splitting
                    param_dict,             # Parameter dictionary with the following:
                        #.minEvents,            # Minimum number of events
                        #.speedThreshold,       # Lower speed threshold for analysing data
                        #.minPeakField,         # Minimum peak rate to be called a place field
                        #.minShufflingInt,      # Minimum number of frames to shuffle the data
                        #.nShuffle,             # Number of times to perform the shuffling
                    cellNo                  # Cell number, the iterable
                ))     
        
            for future in tqdm(cf.as_completed(futures), total=nCells):
                jobNo = future.result()[0]
                spatialInformation[sessionNo,jobNo] = future.result()[1]
                correlation_r[sessionNo,jobNo] = future.result()[2]
                placefields[sessionNo][jobNo] = future.result()[3]
                shuffleSI[:,jobNo] = future.result()[4]
                shuffleCorr[:,jobNo] = future.result()[5]
                if not np.isnan(future.result()[6]):    # If candidatePC is nan, it means the cell had either too low SNR or too few events 
                    candidatePC.append(future.result()[6])                                
                      
        # Get percentile of shuffled data
        shuffleSIPrc[sessionNo] = np.percentile(shuffleSI,95,axis=0)
        shuffleCorrPrc[sessionNo] = np.percentile(shuffleCorr,95,axis=0)
        
        candidatePC = np.sort(np.asarray(candidatePC, dtype=int)) # Make into array and sort in ascending order
        
        fieldList = []
        for i in (candidatePC-1): 
            fieldList.append(type(placefields[sessionNo][i])==list)   # Returns False if there is no detected place fields (type would be float, nan)
        
        # Sort out candidate cells that reach all place cell criteria  
        true_SI = spatialInformation[sessionNo][candidatePC-1] > shuffleSIPrc[sessionNo][candidatePC-1] # SI over shuffled distribution
        true_stability = correlation_r[sessionNo][candidatePC-1] > shuffleCorrPrc[sessionNo][candidatePC-1] # Correlation over shuffled distribution
        
        truePC.append(candidatePC[true_SI  # SI over shuffled distribution
                        & true_stability   # Correlation over shuffled distribution  
                        & fieldList])      # At least one detectable place field, the value is False if there is none 
        
        placecell_dict['NAT'+str(sessionNo)] = truePC
        
        #elapsed_time_session = time.process_time() - session_t
        #print('Session runtime:'+str(elapsed_time_session)+'seconds')
        
    #elapsed_time = time.process_time() - t
    #print('Total runtime:'+str(elapsed_time) +'seconds')
    
    #%% Save result
    session_dict['Placecell'] = placecell_dict
    
    with open(session_dict['ExperimentInformation']['RawDataAddress']+'/session_dict.pickle','wb') as handle:
        pickle.dump(session_dict,handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    nPC = len(np.unique(np.concatenate([placecell_dict['NAT0'][0],placecell_dict['NAT1'][0],placecell_dict['NAT2'][0]])))
    print('Found '+str(nPC)+' unique placecells ('+str(np.round(nPC/nCells*100,1))+' %)')
    print('Sucessfully stored the placecell_dict with in this session_dict.pickle')