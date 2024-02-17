# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:34:46 2023

@author: torstsl
"""
import pickle
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import opexebo as op
from src.cell_activity_analysis import get_signaltracking, calc_tuningmap, calc_activityMatrix, poisson_model_lg_vector, calc_decodingPos, plot_decodingAccuracy
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr

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

#%% Decode just from PCs with similar rate maps (based on one reference cell) set to r > 0.5, only in session A

def decoding_by_ratemapscorr(session_dict, decoding_params, selfPC, **kwargs):
    
    plotter = kwargs.get('plotter', False) 
    
    PC = session_dict['Placecell']['NAT0'][0]
    
    if plotter == True:
        # Use a reference PC
        fig, ax = plt.subplots()
        ax.imshow(session_dict['Ratemaps']['dfNAT0']['N'+str(selfPC)])

    rVals = np.full(PC.size, np.nan)
    for n, cell in enumerate(PC):
        rVals[n] = pearsonr(session_dict['Ratemaps']['dfNAT0']['N'+str(selfPC)].flatten(), session_dict['Ratemaps']['dfNAT0']['N'+str(cell)].flatten())[0]
    
    minR = 0.4
    boolPC = rVals > minR
    print('Place cells with correlation >' + str(minR) +': ' + str((np.nansum(boolPC))))
    
    # Decode
    dfNAT = session_dict['dfNAT']['dfNAT0']
    
    # Filter out the dfNAT in this session for selected place cells 
    PCs = ['Deconvolved, N'+str(x) for x in PC[boolPC]]
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

    for num, cellNo in enumerate(PC[boolPC]):    
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
                          'decodingError': decodingError,
                          'speedIdx': speedIdx}
            
    # Calculate a "summed" rate map
    ratemaps = np.ma.zeros([len(PCs), 32, 32])
    # ratemaps = np.ma.zeros([len(rPC), 32, 32])
    
    for n, N in enumerate(PC[boolPC]):
        ratemaps[n] = session_dict['Ratemaps']['dfNAT0']['N'+str(N)]
            
    summedMap = np.nansum(ratemaps, axis = 0)

    if plotter == True:
        # Decoding accuracy speed filtered
        start, stop = 0, 300
        plot_decodingAccuracy(truePos, decodingPos, speedIdx, start, stop, timeBin, 'Decoding accuracy', save = False)
        
        # plt.savefig('N:/axon2pmini/Article/Figures/Supplementary/fig5I_correlated_decoding_example1_new.svg', format = 'svg') 
        
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

        # plt.savefig('N:/axon2pmini/Article/Figures/Supplementary/fig5I_correlated_decoding_example2_new.svg', format = 'svg') 

    # Return the decoding results and the "summed" ratemap for the cells used to decode

    return decode_ratemapcorr, summedMap

#%% 

scale = 1.1801 # Pixels/µm --> 1 pixel = 0.847.. µm
binning = 2.5 # cm/bin in ratemap
boxSize = 80 # cm
nBins = int(boxSize/binning)

decoding_params = {'binFrame': 3, # Number of frames per time bin
                   'frameRate': 7.5, # Sampling frame rate
                   'timeBin': 3*1/7.5} # decoding_params['binFrame']*1/decoding_params['frameRate']}

palette = sns.color_palette("viridis", 256).as_hex()
contrast = sns.color_palette('OrRd', 256).as_hex()

ex_session = 'N:\\axon2pmini\\Recordings\\102124\\280922'
session_dict = pickle.load(open(ex_session+'\session_dict.pickle','rb'))

#%% 

decodingRes = []
sumMap = []
for selfPC in session_dict['Placecell']['NAT0'][0]:
    decode_ratemapcorr, summedMap = decoding_by_ratemapscorr(session_dict, decoding_params, selfPC, plotter = False)
    decodingRes.append(decode_ratemapcorr)
    sumMap.append(sumMap)
      
#%%
errorMean = []
numPCs = []
for res in decodingRes:
    errorMean.append(np.nanmean(res['decodingError'][res['speedIdx']]))
    numPCs.append(res['activityMatrix'].shape[1])
    
errorMean, numPCs = np.array(errorMean), np.array(numPCs)   

#%% 

decoding_correlated_PCs = {'decodingRes': decodingRes,
                           'sumMap': sumMap,
                           'errorMean': errorMean,
                           'numPCs': numPCs}

results_folder = r'C:\Users\torstsl\Projects\axon2pmini\results'

with open(results_folder+'/decoding_correlated_PCs.pickle','wb') as handle:
     pickle.dump(decoding_correlated_PCs,handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
#%%  

selfPC = 2
corrCells = rVals>0.7
idx = np.where(corrCells == True)[0]
idxPC = PC[idx] # NATEX INDEX of place cells

rmaps = np.full([idxPC.size, 32, 30], np.nan)
rMask = session_dict['Ratemaps']['dfNAT0']['N1'].mask[:,0:30]     

for n, x in enumerate(idxPC): 
    rmap = session_dict['Ratemaps']['dfNAT0']['N'+str(x)][:,0:30]
    rmaps[n] = rmap
    plt.imshow(rmap)
    plt.show()
    
sumMap = np.nansum(rmaps, axis = 0)
sumMap = np.ma.array(sumMap, mask = rMask)

plt.imshow(np.nanmax(rmaps, axis = 0))

fig, ax = plt.subplots(figsize = (5,5))
sns.heatmap(data = sumMap, ax = ax, square = True, robust = True, mask = rMask,
            xticklabels = False, yticklabels = False, cmap = 'viridis', cbar = False) 
ax.set_title('Summed tuning map, n = '+str(idxPC.size)+' place cells')

plt.savefig('N:/axon2pmini/Article/Figures/Supplementary/fig5_supplementary_sumTuningMap.svg', format = 'svg') 

fig, ax = plt.subplots(6,8, figsize = (10,10))
for x, axes in zip(range(idxPC.size), ax.flatten()):
    sns.heatmap(data = rmaps[x], mask = rMask, ax = axes, square = True, robust = True, 
                xticklabels = False, yticklabels = False, cmap = 'viridis', cbar = False, rasterized = True)
    axes.set_title('N'+str(idxPC[x]))
plt.tight_layout()

plt.savefig('N:/axon2pmini/Article/Figures/Supplementary/fig5_supplementary_corrTuningMaps.svg', format = 'svg') 
