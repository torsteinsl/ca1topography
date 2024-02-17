# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:33:21 2023

@author: torstsl

Object analysis

Loads and analyses topography amongst object tuned cells in CA1.

"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from src.cell_activity_analysis import plot_maxproj_feature, get_cell_masks, get_signaltracking
from matplotlib.colors import Normalize 
from scipy.stats import pearsonr

#%% Helper functions for anatomical distances

#% Get anatomical distance between all cell pairs
def get_cellCentre(session_dict):
    
    nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)
    cellCentre = np.full([nCells,2], np.nan) 
    
    for cellNo in range(nCells): 
        cellCentre[cellNo] = session_dict['ExperimentInformation']['CellStat'][cellNo]['med'][::-1] # Python index

    return cellCentre

# Get anatomical distance between only object cell pairs
def calc_anatomicalDistance(cellCentre, **kwargs): 
    
    scale = kwargs.get('scale', 1) # If not provided, scaled to 1 (=not scaled)
    cellIdx = kwargs.get('cellIdx', np.arange(cellCentre.shape[0]))  # If not provided, do distance between all coordinates
    
    anatomicalDistance = np.linalg.norm(cellCentre[cellIdx] - cellCentre[cellIdx][:,None], axis=-1)/scale # Scaled by µm/pixel
    
    return anatomicalDistance

#%% Plot the mask of just the OC

def get_FOV_object(session_dict, objectCells, OVC, SNR, **kwargs):
    
    saving = kwargs.get('saving', False)
    plotter = kwargs.get('plotter', False)
    
    maxprojection, maxprojection_mask = get_cell_masks(session_dict['ExperimentInformation'])
    
    if plotter == True:
        plot_maxproj_feature(maxprojection, maxprojection_mask[objectCells-1], SNR[objectCells-1], 
                             'Object tuned cells', background = 'maxint', vmin = 0, vmax = max(SNR[objectCells-1]))  
        
        if saving == True: plt.savefig('N:/axon2pmini/Article/Figures/Figure 6/maxproj_mask_OC.svg', format = 'svg') 
        
        # Plot the mask of just the OVC
        plot_maxproj_feature(maxprojection, maxprojection_mask[OVC-1], SNR[OVC-1], 
                             'Object vector cells', background = 'maxint', vmin = 0, vmax = max(SNR[OVC-1]))  
        
        if saving == True: plt.savefig('N:/axon2pmini/Article/Figures/Figure 6/maxproj_mask_OVC.svg', format = 'svg') 
    
    return maxprojection, maxprojection_mask

#%% Plot activity maps of example cells

def plot_activityMaps(dfNATs, objectData, objectCells, **kwargs):
    
    saving = kwargs.get('saving', False)
        
    objectPos = (objectData['ObjectPosX'], objectData['ObjectPosY'])
    objectMovedPos = (objectData['ObjectMovedPosX'], objectData['ObjectMovedPosY'])
    
    sessionName = ['Open field','Object','Object moved', 'Open field']
    
    # For session on mouse 102124
    if session_dict['Animal_ID'] == '102124': saveCells = 11, 15, 27, 37, 200, 220, 225, 256, 352, 494, 584, 599, 812 
    
    for cellNo, OC in enumerate(objectCells):
            
        fig, ax = plt.subplots(1,len(dfNATs.keys()), figsize = (10,10))
        fig.supylabel('N'+str(OC))
        
        for NAT, axes, name in zip(dfNATs.keys(), ax.flatten(), sessionName):
            dfNAT = dfNATs[NAT]
            headpos = dfNAT[['Head_X','Head_Y']]
            
            signaltracking = get_signaltracking(dfNAT, objectCells[cellNo], signal='deconvolved', speed_threshold=2.5)
        
            axes.set_title(name)
            axes.plot(headpos['Head_X'],headpos['Head_Y'], color = 'darkgrey', zorder=1)
            axes.scatter(signaltracking.Head_X, signaltracking.Head_Y, s=(signaltracking.Signal*2), color=contrast[200], alpha = 0.8, zorder=2)
            if name == 'Object':
                axes.scatter(objectPos[0], objectPos[1], color = palette[80], marker = 's', s = 100, zorder = 1)
            
            elif name == 'Object moved':
                axes.scatter(objectMovedPos[0], objectMovedPos[1], color = palette[80], marker = 's', s = 100, zorder = 1)
            axes.axis('off')
            axes.set_aspect('equal')
            
        plt.tight_layout()
        
        if saving == True: 
            if OC in saveCells: plt.savefig('N:/axon2pmini/Article/Figures/Figure 6/activityMaps_'+str(OC)+'.svg', format = 'svg') 

#%% Get the angular tuning and vector length of object cells

def calc_objectTuning(dfNATs, objectData, objectCells, **kwargs): 
    
    testing = kwargs.get('testing', False)
        
    objectPos = (objectData['ObjectPosX'], objectData['ObjectPosY'])
    objectMovedPos = (objectData['ObjectMovedPosX'], objectData['ObjectMovedPosY'])
    objectCellStats = objectData['ObjectCellStats']
    
    angle2object = np.array([])
    vlength2object = np.array([])
    angle2objectOC = np.array([])
    
    for cellNo, OC in enumerate(objectCellStats.keys()):
        if objectCellStats[OC]['fieldAngle2object'].size == 1:
            angle2object = np.append(angle2object, objectCellStats[OC]['fieldAngle2object'])
            vlength2object = np.append(vlength2object, objectCellStats[OC]['fieldDist2object'])
            angle2objectOC = np.append(angle2objectOC, objectCells[cellNo])
    
        # If there is more object fields, take the one closest to the object
        elif objectCellStats[OC]['fieldAngle2object'].size > 1: 
            idx = np.where(objectCellStats[OC]['fieldDist2object'] == np.min(objectCellStats[OC]['fieldDist2object']))[0][0]
           
            angle2object = np.append(angle2object, objectCellStats[OC]['fieldAngle2object'][idx])
            vlength2object = np.append(vlength2object, objectCellStats[OC]['fieldDist2object'][idx])
            angle2objectOC = np.append(angle2objectOC, [objectCells[cellNo]])
            
            print(str(OC)+' had more than 1 new field in object trial')
            
            if testing == True: 
                # Plot the activity maps with objects and vector to all object fields
                angle = np.deg2rad(objectCellStats[OC]['fieldAngle2object'])
                vlength = objectCellStats[OC]['fieldDist2object']
                
                fig, ax = plt.subplots(1,len(dfNATs.keys()), figsize = (10,10))
                fig.supylabel(OC)
                for NAT, axes, name in zip(dfNATs.keys(), ax.flatten(), sessionName):
                    dfNAT = dfNATs[NAT]
                    headpos = dfNAT[['Head_X','Head_Y']]
                    
                    signaltracking = get_signaltracking(dfNAT, objectCells[cellNo], signal='deconvolved', speed_threshold=2.5)
                
                    axes.set_title(name)
                    axes.plot(headpos['Head_X'],headpos['Head_Y'], color = 'darkgrey', zorder=1)
                    axes.scatter(signaltracking.Head_X, signaltracking.Head_Y, s=(signaltracking.Signal*2), color=contrast[200], alpha = 0.8, zorder=2)
                    if name == 'Object':
                        axes.scatter(objectPos[0], objectPos[1], color = palette[80], marker = 's', s = 100, zorder = 1)
                        
                        for a,v in zip(angle, vlength):
                            axes.plot((objectPos[0], objectPos[0] + np.cos(a)*v), # x-coordinate start,stop
                                      (objectPos[1], objectPos[1] + np.sin(a)*v), # y-coordinate start,stop
                                      linewidth = 3, color = palette[180], alpha = 0.75, zorder = 3)
                            
                    elif name == 'Object moved':
                        axes.scatter(objectMovedPos[0], objectMovedPos[1], color = palette[80], marker = 's', s = 100, zorder = 1)
                    axes.set_aspect('equal')
                    axes.axis('off')
                plt.tight_layout()
            
    # Pairwise angular and vector length difference
    angularDifference = abs(angle2object - angle2object[:,None])%180   
    vlengthDifference = abs(vlength2object - vlength2object[:,None])

    return angle2object, vlength2object, angle2objectOC, angularDifference, vlengthDifference

#%% Polar plot with angle and vector length for all object fields
def plot_polarscatter_vector(angle2object, vlength2object, **kwargs):
    
    saving = kwargs.get('saving', False)
    
    # Plot the figure
    circular_palette = np.array(sns.color_palette('twilight_shifted', 360).as_hex())
    colors = circular_palette[angle2object.astype(int)]
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    ax.scatter(np.deg2rad(angle2object), vlength2object, color = colors, alpha=0.8, zorder = 2)
    
    if saving == True: plt.savefig('N:/axon2pmini/Article/Figures/Figure 6/polarScatter_cmap.svg', format = 'svg')  
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    ax.scatter(np.deg2rad(angle2object), vlength2object, color = palette[80], alpha=0.75, zorder = 2)
    
    if saving == True: plt.savefig('N:/axon2pmini/Article/Figures/Figure 6/polarScatter.svg', format = 'svg')  

#%% Get anatomical differences and angular differences in vectors

def get_differences(angle2objectOC, cellCentre, angularDifference, vlengthDifference, scale):
    
    angleIdx = angle2objectOC.astype(int)-1 # Convert from NATEX to Python and change to int

    anatomicalAngularCells = calc_anatomicalDistance(cellCentre, cellIdx = angleIdx, scale = scale) # Scaled by µm/pixel

    upperMask = np.triu(np.full([angleIdx.size,angleIdx.size], True), 1)

    anatomicalDiff = anatomicalAngularCells[upperMask].flatten()
    angularDiff = angularDifference[upperMask].flatten()
    vlengthDiff = vlengthDifference[upperMask].flatten()
    
    return angleIdx, anatomicalDiff, angularDiff, vlengthDiff

#%% Plot the FOV with colouring by angle

def plot_FOV_angular_tuning(maxprojection, maxprojection_mask, angleIdx, angle2object, **kwargs):
    
    saving = kwargs.get('saving', False)
    cmap = kwargs.get('cmap', False)
    
    plot_maxproj_feature(maxprojection, maxprojection_mask[angleIdx], angle2object, 
                         'Object cells', background = 'maxint', palette = 'twilight_shifted') 
    
    if saving == True: plt.savefig('N:/axon2pmini/Article/Figures/Figure 6/FOV_objectCells.svg', format = 'svg') 
    
    if cmap == True:
        
        # Plot a circle with the cmap
        fg = plt.figure(figsize=(8,8))
        ax = fg.add_axes([0.1,0.1,0.8,0.8], projection='polar')
        
        # Define colormap normalization for 0 to 2*pi
        norm = Normalize(0, 2*np.pi) 
        
        # Plot a color mesh on the polar plot with the color set by the angle
        n = 200  #the number of secants for the mesh
        t = np.linspace(0,2*np.pi,n)   #theta values
        r = np.linspace(.75,1,2)        #radius values change 0.6 to 0 for full circle
        rg, tg = np.meshgrid(r,t)      #create a r,theta meshgrid
        c = tg                         #define color values as theta value
        ax.pcolormesh(t, r, c.T,norm=norm, cmap = 'twilight_shifted')  #plot the colormesh on axis with colormap
        ax.set_yticklabels([])                   #turn of radial tick labels (yticks)
        ax.tick_params(pad=15,labelsize=24)      #cosmetic changes to tick labels
        ax.spines['polar'].set_visible(False)    #turn off the axis spine.
        
        plt.savefig('N:/axon2pmini/Article/Figures/Figure 6/circularCmap.svg', format = 'svg') 

#%% Plot the pairwise anatomical distance and angular difference 

def plot_pairwise(anatomicalDiff, angularDiff, **kwargs):
    
    saving = kwargs.get('saving', False)
    
    palette = list(sns.color_palette("viridis", 256).as_hex())
    contrast = list(sns.color_palette('OrRd', 256).as_hex())
    
    # Plot heatmap using Seaborn
    dfPlot = pd.DataFrame({'Anatomical difference': anatomicalDiff, 'Angular difference': angularDiff})
    
    s = sns.jointplot(data = dfPlot, x='Anatomical difference', y='Angular difference', kind='hex', cmap = 'OrRd',
                      marginal_kws = {'bins': 25, 'color': '#d83221', 'alpha': 0.9}, xlim = (0,450), ylim = (0,180))
    s.ax_joint.grid(False)
    s.ax_marg_y.grid(False)
    s.set_axis_labels('Anatomical distance (µm)', 'Angular difference (deg)')
    s.fig.suptitle('Pairwise anatomical distance and angular difference', y=1.02)
    
    # Plot joined scatter using Seaborn
    s = sns.jointplot(data = dfPlot, x='Anatomical difference', y='Angular difference', kind='scatter', 
                      color = contrast[180], cmap = 'OrRd', alpha = 0.75,
                      marginal_kws = {'bins': 25, 'color': '#d83221', 'alpha': 0.9})#, xlim = (-5,455), ylim = (-2,182))
    s.ax_joint.grid(False)
    s.ax_marg_y.grid(False)
    s.set_axis_labels('Anatomical distance (µm)', 'Angular difference (deg)')
    s.fig.suptitle('Pairwise anatomical and angular distance', y=1.02)
    
    if saving == True: plt.savefig('N:/axon2pmini/Article/Figures/Figure 6/jointmap.svg', format = 'svg')    
    
    # Plot regression plot using Seaborn
    fig, ax = plt.subplots(figsize = (5,5))
    sns.regplot(data = dfPlot, x = 'Anatomical difference', y = 'Angular difference', ax = ax, 
                    color = palette[80], line_kws = {'color': contrast[180], 'alpha': 1}, scatter_kws = {'s': 10, 'alpha': 0.75})
    plt.setp(ax.collections[1], alpha=0.35)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set(xlabel = 'Anatomical difference (µm)', ylabel = 'Angular difference (deg)', title = 'n = '+str(len(angularDiff))+' cell pairs')
    
    if saving == True: plt.savefig('N:/axon2pmini/Article/Figures/Figure 6/scatterReg.svg', format = 'svg')  

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

#%% Loop over all object sessions and save all data needed to pool and visaualize the results

# Predefine results dictionary
objectResults_dict = pickle.load(open(results_folder+'\objectResults_dict.pickle','rb'))
save_results = True

# Loop over all paths, load data and analyse it
for session in objectSessions:

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

    # Plot FOV of object cells 
    maxprojection, maxprojection_mask = get_FOV_object(session_dict, objectCells, OVC, SNR, saving = False, plotter = False)
    
    # Plot activity maps of object cells
    dfNATs = session_dict['dfNAT']
    objectData = session_dict['ObjectData']
    
    plot_activityMaps(dfNATs, objectData, objectCells, saving = False)
    
    # Get cell centres and anatomical distances
    cellCentre = get_cellCentre(session_dict)

    # Calcualte angle to object and angular differences
    objectCellStats = objectData['ObjectCellStats']
    angle2object, vlength2object, angle2objectOC, angularDifference, vlengthDifference = calc_objectTuning(dfNATs, objectData, objectCells, saving = False) 
    
    # Plot a scatter of object cells' vectorial tuning
    plot_polarscatter_vector(angle2object, vlength2object, saving = False)
    
    # Get anatomical differences and angular differences in vectors
    angleIdx, anatomicalDiff, angularDiff, vlengthDiff = get_differences(angle2objectOC, cellCentre, angularDifference, vlengthDifference, scale)
    
    # Plot FOV with ROIs colour coded by preferred angle
    plot_FOV_angular_tuning(maxprojection, maxprojection_mask, angleIdx, angle2object, saving = False, cmap = False)
    
    # Plot the pairwise relations between anatomical distances and angular distances
    plot_pairwise(anatomicalDiff, angularDiff, saving = True)
    
    # Put together the results in a dictionary    
    objectResults_dict[sessionKey] = {'angle2object': angle2object, 
                                      'vlength2object': vlength2object,
                                      'anatomicalDiff': anatomicalDiff,
                                      'angularDiff': angularDiff,
                                      'vlengthDiff': vlengthDiff}
    
    if save_results == True: 
        with open(results_folder+'/objectResults_dict.pickle','wb') as handle:
            pickle.dump(objectResults_dict,handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Successfully saved objectResults_dict in '+ results_folder)

#%% Pool all data across sessions, and analyse them

# Load results
objectResults_dict = pickle.load(open(results_folder+'\objectResults_dict.pickle','rb'))

tempAnat, tempAng, tempvDist = [], [], []
tempang2obj, tempdist2obj = [], []

for key in objectResults_dict.keys():
    tempAnat.append(objectResults_dict[key]['anatomicalDiff'])
    tempAng.append(objectResults_dict[key]['angularDiff'])
    tempvDist.append(objectResults_dict[key]['vlengthDiff'])
    tempang2obj.append(objectResults_dict[key]['angle2object'])
    tempdist2obj.append(objectResults_dict[key]['vlength2object'])
    
anatDiff, angDiff, vlengthDiff = np.concatenate(tempAnat), np.concatenate(tempAng), np.concatenate(tempvDist)
ang2obj, dist2obj = np.concatenate(tempang2obj), np.concatenate(tempdist2obj)

# Plot all in a heat map/scatter with regression
dfPlot = pd.DataFrame({'Anatomical difference': anatDiff, 'Angular difference': angDiff})

s = sns.jointplot(data = dfPlot, x='Anatomical difference', y='Angular difference', kind='hex', cmap = 'OrRd',
                  marginal_kws = {'bins': 25, 'color': '#d83221', 'alpha': 0.9}, xlim = (0,450), ylim = (0,180))
s.ax_joint.grid(False)
s.ax_marg_y.grid(False)
s.set_axis_labels('Anatomical distance (µm)', 'Angular difference (deg)')
s.fig.suptitle('Pairwise anatomical distance and angular difference', y=1.02)

plt.savefig('N:/axon2pmini/Article/Figures/Figure 6/jointmap_pooled.svg', format = 'svg')   

r, p = pearsonr(dfPlot['Anatomical difference'], dfPlot['Angular difference'])
print(r**2)

# Linear regression on just one session
from scipy.stats import linregress
a1 = objectResults_dict['102124-081222']['anatomicalDiff']
a2 = objectResults_dict['102124-081222']['angularDiff']
res1 = pearsonr(a1, a2)
res2 = linregress(a1, a2)
res3 = linregress(a2, a1)

# Plot joined scatter using Seaborn
s = sns.jointplot(data = dfPlot, x='Anatomical difference', y='Angular difference', kind='scatter', 
                  color = contrast[180], cmap = 'OrRd', alpha = 0.75, joint_kws = {'s': 15},
                  marginal_kws = {'bins': 25, 'color': '#d83221', 'alpha': 0.9})#, xlim = (-5,455), ylim = (-2,182))
s.ax_joint.grid(False)
s.ax_marg_y.grid(False)
s.set_axis_labels('Anatomical distance (µm)', 'Angular difference (deg)')
s.fig.suptitle('Pairwise anatomical and angular distance', y=1.02)

plt.savefig('N:/axon2pmini/Article/Figures/Figure 6/jointmap_scatter_pool.svg', format = 'svg')    

# Plot regression plot using Seaborn
fig, ax = plt.subplots(figsize = (5,5))
sns.regplot(data = dfPlot, x = 'Anatomical difference', y = 'Angular difference', ax = ax, 
                color = palette[80], line_kws = {'color': contrast[180], 'alpha': 1}, scatter_kws = {'s': 10, 'alpha': 0.5})
plt.setp(ax.collections[1], alpha=0.35)
ax.spines[['top', 'right']].set_visible(False)
ax.set(xlabel = 'Anatomical difference (µm)', ylabel = 'Angular difference (deg)', title = 'n = '+str(len(dfPlot))+' cell pairs')

plt.savefig('N:/axon2pmini/Article/Figures/Figure 6/scatterReg_pool.svg', format = 'svg')  

r, p = pearsonr(dfPlot['Anatomical difference'], dfPlot['Angular difference'])
print(r**2)

# Plot a polar scatter plot of vector tuning
circular_palette = np.array(sns.color_palette('twilight_shifted', 360).as_hex())
colors = circular_palette[ang2obj.astype(int)]

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
ax.scatter(np.deg2rad(ang2obj), dist2obj, color = colors, s = 15, alpha=0.80, zorder = 2)
ax.set_title('Distribution of object tuning (n = '+str(ang2obj.size)+' cells)')

plt.savefig('N:/axon2pmini/Article/Figures/Figure 6/polarScatter_cmap_pool.svg', format = 'svg')  

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
ax.scatter(np.deg2rad(ang2obj), dist2obj, color = palette[80], s = 15, alpha=0.7, zorder = 2)
ax.set_title('Distribution of object tuning (n = '+str(ang2obj.size)+' cells)')

plt.savefig('N:/axon2pmini/Article/Figures/Figure 6/polarScatter_pool.svg', format = 'svg')

# Descriptive statistics on vector lengths
medianD2O = np.median(dist2obj)
iqrD2O = np.percentile(dist2obj,25), np.percentile(dist2obj,75)

# Calclate circular uniformity: 
from scipy.stats import circmean, rayleigh
from astropy.stats import rayleightest, circstd    
rtest = rayleightest(np.deg2rad(ang2obj))

rstat = rayleigh(np.deg2rad(ang2obj))

# Manual Rayleigh test
n = ang2obj.size 
r = np.sqrt(np.sum(np.cos(np.deg2rad(ang2obj)))**2 + np.sum(np.sin(np.deg2rad(ang2obj)))**2) / n

R = n*r # Rayleigh's R
z = R**2 / n # Rayleigh's z 
pval = np.exp(np.sqrt(1+4*n+4*(n**2-R**2))-(1+2*n))


cmean= circmean(np.deg2rad(ang2obj))
cstd = circstd(np.deg2rad(ang2obj))
cstd/np.sqrt(ang2obj.size)
np.rad2deg(cmean), np.rad2deg(cstd)


# Plot vector lengths in a heat map/scatter and regression
dfAnatVSvlength = pd.DataFrame({'Anatomical difference': anatDiff, 'vLength difference': vlengthDiff})
r, p = pearsonr(dfAnatVSvlength['Anatomical difference'], dfAnatVSvlength['vLength difference'])
print(r**2)

s = sns.jointplot(data = dfAnatVSvlength, x='Anatomical difference', y='vLength difference', 
                  kind='scatter', color = contrast[200], alpha = 0.5, joint_kws = {'s': 15},
                  marginal_kws = {'bins': 25, 'color': '#d83221', 'alpha': 0.9}, xlim = (0,450), ylim = (-1,70))
s.ax_joint.grid(False)
s.ax_marg_y.grid(False)
s.set_axis_labels('Anatomical distance (µm)', 'Vector length difference (cm)')
s.fig.suptitle('Pairwise anatomical distance and vector length difference', y=1.02)

plt.savefig('N:/axon2pmini/Article/Figures/Figure 6/joint_vlength_pool.svg', format = 'svg')

s = sns.jointplot(data = dfAnatVSvlength, x='Anatomical difference', y='vLength difference', kind='hex', cmap = 'OrRd',
                  marginal_kws = {'bins': 25, 'color': '#d83221', 'alpha': 0.9}, xlim = (0,450), ylim = (-1,50))
s.ax_joint.grid(False)
s.ax_marg_y.grid(False)
s.set_axis_labels('Anatomical distance (µm)', 'Vector length difference (cm)')
s.fig.suptitle('Pairwise anatomical distance and vector length difference', y=1.02)

plt.savefig('N:/axon2pmini/Article/Figures/Figure 6/jointHex_vlength_pool.svg', format = 'svg')

fig, ax = plt.subplots(figsize = (6,6))
sns.regplot(data = dfAnatVSvlength, x = 'Anatomical difference', y = 'vLength difference', ax = ax, 
                color = palette[80], line_kws = {'color': contrast[180], 'alpha': 1}, scatter_kws = {'s': 8, 'alpha': 0.5})
plt.setp(ax.collections[1], alpha=0.5)
ax.spines[['top', 'right']].set_visible(False)
ax.set(xlabel = 'Anatomical difference (µm)', ylabel = 'Vector length difference (cm)', title = 'n = '+str(len(dfAnatVSvlength))+' cell pairs')

plt.savefig('N:/axon2pmini/Article/Figures/Figure 6/reg_vlength_pool.svg', format = 'svg')

#%% What object tuned cells are place cells (and not)?

nOC = []
nOPC = []

for session in objectSessions:

    session_dict = pickle.load(open(session+'\session_dict.pickle','rb'))
    print('Successfully loaded session_dict from: '+str(session))
    
    if not 'ObjectData' in session_dict.keys(): 
        print('No object data in given data')
    else: 
        print('Found object data within session_dict')
        print('Found '+str(session_dict['ObjectData']['ObjectTunedCells'].size)+' object tuned cells')
    
    # Are object tuned? Are place cell?
    pc = session_dict['Placecell']['NAT0'][0]
    oc = session_dict['ObjectData']['ObjectTunedCells']
    
    nOC.append(oc.size)
    nOPC.append(np.sum(np.isin(pc,oc)))
    
numOC = np.sum(nOC)
numOPC = np.sum(nOPC)   
nonPC = numOC - numOPC
    
#%% PLAYGROUND

# Calculate R**2 for objectSession[0]
key = '102124-081222'
x = objectResults_dict[key]['anatomicalDiff']
y = objectResults_dict[key]['angularDiff']
r,_ = pearsonr(x,y)
print('R**2 = '+str(r**2))

dfData = pd.DataFrame({'ang2obj': ang2obj, 'dist2obj': dist2obj})

# Plot regression plot
# NOTE: This is a linear form of the polar plot. It assumes lineraity on the regression, which doesn't make sense on this circular data
fig, ax = plt.subplots(figsize = (5,5))
sns.regplot(data = dfData, x = 'ang2obj', y = 'dist2obj', ax = ax, color = palette[80], 
            line_kws = {'color': contrast[180], 'alpha': 1}, scatter_kws = {'s': 15, 'alpha': 0.75})
plt.setp(ax.collections[1], alpha=0.35)
ax.spines[['top', 'right']].set_visible(False)
ax.set(xlabel = 'Angle to object (deg)', ylabel = 'Distance to object (cm)', title = 'Angle vs. distance (n = '+str(len(dfData))+' cells)')

plt.savefig('N:/axon2pmini/Article/Figures/Figure 6/ang2obj_vs_dist2obj.svg', format = 'svg')

fig, ax = plt.subplots()
sns.histplot(data = dfData, x = 'ang2obj', bins = 30, color = palette[80])

# Distribution of angles in all sessions
fig, ax = plt.subplots(2,2, figsize = (7,6))
plt.suptitle('Distribution of angle to object for each session')
for key, axes in zip(objectResults_dict.keys(), ax.flatten()):
    ang2obj_temp = objectResults_dict[key]['angle2object']  
    sns.histplot(data = ang2obj_temp, ax = axes, bins = 20, color = palette[80])
    axes.set_title('n = '+str(ang2obj_temp.size)+' cells')
    axes.set(xlabel = 'Angle (deg)', ylabel = 'Count', xlim =(0))
    axes.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

plt.savefig('N:/axon2pmini/Article/Figures/Figure 6/ang2obj_dist_hist_session.svg', format = 'svg')
