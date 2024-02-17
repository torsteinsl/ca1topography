# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 16:20:02 2023

@author: torstsl

Script that calculates and plots the centric tuning maps of given cells
from a spesific session. Made to match the cells in Figure 6. Manual inputs
that can be changed. 

Basicly, it creates a rate map with the object in the centre rather than just
the 2D coordinates of the box. When the object is moved, the object tuning
should remain (angle and distance). 

For the open field sessions, the position in the first object trial is used. 

"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.filters import gaussian

#%% Define constants and variables, load path to all object sessions

scale = 1.1801 # Pixels/µm --> 1 pixel = 0.847.. µm 
binning = 2.5 # cm/spatial bin in rate map

palette = list(sns.color_palette("viridis", 256).as_hex())
contrast = list(sns.color_palette('OrRd', 256).as_hex())

sessionName = ['Open field','Object','Object moved', 'Open field']

# Load data paths
results_folder = r'C:\Users\torstsl\Projects\axon2pmini\results'

with open(results_folder+'/ObjectSessions_overview.txt') as f:
    objectSessions = f.read().splitlines() 
f.close()

#%% Load primary session

session = objectSessions[0]

session_dict = pickle.load(open(session+'\session_dict.pickle','rb'))
print('Successfully loaded session_dict from: '+str(session))

if not 'ObjectData' in session_dict.keys(): 
    print('No object data in given data')
else: 
    print('Found object data within session_dict')
    print('Found '+str(session_dict['ObjectData']['ObjectTunedCells'].size)+' object tuned cells')

# Define session constants and variables
sessionKey = session_dict['Animal_ID']+'-'+session_dict['Date']

nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)
nSessions = len(session_dict['dfNAT'])
SNR = session_dict['ExperimentInformation']['CellSNR']

objectCells = session_dict['ObjectData']['ObjectTunedCells']
OVC = session_dict['ObjectData']['ObjectVectorCells']

sessionName = ['Open field','Object','Object moved', 'Open field']

plotCells = np.array([352, 11, 200]) # NATEX index

dfNATs = session_dict['dfNAT']
objectData = session_dict['ObjectData']

objectPos = np.array([objectData['ObjectPosX'], objectData['ObjectPosY']])
objectMovedPos = np.array([objectData['ObjectMovedPosX'], objectData['ObjectMovedPosY']])
            
#%% Calculate centric ratemap
def centric_ratemap(signal, angle2object, distance2object, speed, distanceExtremes = (2, 40), 
                    angularBinSize = 6, minTimeinBin = 3, distanceBin = 2, smoothing = 3,
                    speedThreshold = 2.5):

    """
    FROM HANNA ENEQVIST (sligtly adapted)
    
    Given cell signal, distance and angle to object (calculated from animal), it generates a centric ratemap.

    LocalDF is a session specific dataframe containing headX, headY, HD as well as angle2object and distance2object
    signal is session specific activity (deconvolved)
    Angle2object is egocentric angle to object in degrees not radians
    Bin sizes are given in degrees and cm respectively.
    
    """

    # Data binning  
    numAngularBins = round((360/angularBinSize))
    angularBinEdges = np.linspace(0, 2*np.pi, numAngularBins+1)

    numDistanceBins = round((distanceExtremes[1]-distanceExtremes[0])/distanceBin)
    distanceBinEdges = np.linspace(distanceExtremes[0], (distanceExtremes[1] + distanceBin), numDistanceBins+1)

    # Calculate occupancy and spike amplitude in bin
    angle2object = np.radians(angle2object)
    occupancyMap = np.zeros([numDistanceBins, numAngularBins], np.int16)
    signalMap = np.zeros([numDistanceBins, numAngularBins], np.float64)
    
    # Speed filter
    speedFilter = speed >= speedThreshold # Boolean, True if speed >= threshold
    
    angle2object, distance2object, signal = angle2object[speedFilter], distance2object[speedFilter], signal[speedFilter]

    # Calculate the occupancy and rate for each angular-distance bin    
    for angle, dist, signal in zip(angle2object, distance2object, signal):

        #Constrain the data to only look at the distances from we want
        #If all distances are wanted then use distance2object max as the distance boundary
        if distanceExtremes[0] <= dist <= distanceExtremes[1]:

            angle_ind = np.where(angle <= angularBinEdges)[0][0] - 1
            dist_ind = np.where(dist <= distanceBinEdges)[0][0] - 1

            if dist_ind == -1:
                dist_ind = 0

            occupancyMap[dist_ind, angle_ind] += 1
            signalMap[dist_ind, angle_ind] += signal       
    
    # Divide by occupancy, leave out if under minTimeBin
    centric_ratemap = np.zeros_like(signalMap)

    for j in range(numDistanceBins):
        for i in range(numAngularBins):
            if occupancyMap[j, i] > minTimeinBin:
                centric_ratemap[j, i] = signalMap[j, i] / occupancyMap[j, i]

            else:
                centric_ratemap[j, i] = 0

    # Apply a gaussial smoothing on the wrapped data (circular boundaries)
    centric_ratemap_stacked = np.matlib.repmat(centric_ratemap, 1, 3)
    centric_ratemap_stacked_smooth = gaussian(centric_ratemap_stacked, sigma=smoothing)
    centric_ratemap_stacked_smooth = centric_ratemap_stacked_smooth[:,numAngularBins-1:numAngularBins*2-1]
 
    return centric_ratemap_stacked_smooth, angularBinEdges, distanceBinEdges 


#%% getThetaR
def getThetaR(angularBinSize = 6, distanceBin = 2, distanceExtremes =(2, 40)): 
    numAngularBins = round((360/angularBinSize))
    angularBins = np.linspace(0, 2*np.pi, numAngularBins)

    numDistanceBins = round((distanceExtremes[1]-distanceExtremes[0])/distanceBin)
    distanceBins = np.linspace(distanceExtremes[0], distanceExtremes[1], numDistanceBins)

    theta, r = np.meshgrid(angularBins, distanceBins)
    
    return theta, r
#%% Get signal, angle2object and distance2object

debugging = False

fig, ax = plt.subplots(plotCells.size,4, figsize = (15,10), subplot_kw = {'projection': 'polar', 'rasterized': False})

# For polar plotting
theta, r = getThetaR()

# Loop over NATs
for NAT, name in zip(dfNATs.keys(), sessionName):
        dfNAT = dfNATs[NAT]
        headpos = dfNAT[['Head_X','Head_Y']]
        speed = dfNAT['Body_speed'].to_numpy() 
        
        column = int(NAT[-1]) # For plotting
        
        # Define the object position
        if name == 'Object moved': oPos = objectMovedPos
        else: oPos = objectPos
            
        # Calculate distance to object at all timestamps
        distance = np.linalg.norm(headpos.to_numpy() - oPos, axis = -1)
        
        # Calculate angle to object at all timestamps
        x, y = headpos.Head_X - oPos[0], headpos.Head_Y - oPos[1]
        angle = np.arctan2(y, x).to_numpy() # [-pi, pi>
        
        angleDeg = (np.rad2deg(angle)+360)%360 # Convert to deg and wrap to 0-360
        angleRad = np.deg2rad(angleDeg)
        
        if debugging == True: 
            fig, ax = plt.subplots(3,5, figsize = (10,6))
            for n, axs in enumerate(ax.flatten()): 
                
                axs.scatter(headpos.Head_X[n], headpos.Head_Y[n], color = palette[80])
                axs.scatter(objectPos[0], objectPos[1], color = contrast[180], marker = 's')
                axs.set(xlim = ([-45,45]), ylim = ([-45,45]), aspect = 'equal')
                
                axs.plot([objectPos[0], objectPos[0] + np.cos(angleRad[n])*distance[n]], 
                        [objectPos[1], objectPos[1] + np.sin(angleRad[n])*distance[n]], 
                        color = 'forestgreen')
            plt.tight_layout()    

        # Calculate the tuning per cell for this session
        for cellNo, OC in enumerate(plotCells): #cellNo is also row, OC is NATEX index
            signal = dfNAT['Deconvolved, N'+str(OC)].to_numpy()
            angle2object = angleDeg
            distance2object = distance
            
            centric_ratemap_stacked_smooth, angularBinEdges, distanceBinEdges = centric_ratemap(signal, angle2object, distance2object, speed)
            # print(centric_ratemap_stacked_smooth.max())

            ax[cellNo, column].contourf(theta, r, centric_ratemap_stacked_smooth, 100, vmin = 0, vmax = 0.4)
            ax[cellNo, column].grid(alpha = 0.25)
            ax[cellNo, column].set_xticks(np.linspace(0,1.5*np.pi,4))
            ax[cellNo, column].set_yticks([5,10,15,20,25,30])
            ax[cellNo, column].tick_params(axis='y', colors='white')

plt.tight_layout()            

plt.savefig('N:/axon2pmini/Article/Figures/Figure 6/centric_tuningMaps.svg', format = 'svg')  
