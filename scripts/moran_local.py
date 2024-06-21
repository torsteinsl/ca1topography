# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 09:57:00 2024

@author: torstsl

Local Moran's I

Load calculated Moran's I and look at the data

Documentation on the calulcated esda.class data: https://pysal.org/esda/generated/esda.Moran_Local.html

Local Moran's I are plotted together with the bin correlations used to calculate
the local Is. Further, the p-values are plotted per bin. These are corrected with
false discovery rate correction, see https://mgimond.github.io/Spatial/spatial-autocorrelation.html#local-morans-i

The adjusted p-values are used for determining which bins display significant
local Moran's I. The summary of this is plotted as: 
    Top left: Bin correlation values
    Top right: Local Moran's I for each bin
    Bottom left: Adjusted p-values (white if significant, grey if not, alpha = 0.05)
    Bottom right: The local Moran's Is in which the p-values are significantly low
    
The output is stored as .svg    

"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# from scipy.stats import false_discovery_control
from statsmodels.stats.multitest import fdrcorrection

#%% Initialte variables and constants
sessionName = ['A','B','A\'']
NATs = ['NAT0','NAT1','NAT2']

palette = sns.color_palette("viridis", 256).as_hex()
contrast = sns.color_palette('OrRd', 256).as_hex()

results_folder = r'C:\Users\torstsl\Projects\axon2pmini\results'

with open(results_folder+'\sessions_overview.txt') as f:
    sessions = f.read().splitlines() 
f.close()

plot3 = pickle.load(open(r'C:\Users\torstsl\Projects\axon2pmini\results\grids_anatomy_dict.pickle','rb'))

# Save path for figures
savepath = 'N:/axon2pmini/Article - Topography/Figures/Local Moran/'

#%% Local Moran's I 

for key in plot3.keys():
    data = plot3[key]
    
    for NAT in NATs[0:2]:
        moranLoc = data['moran_loc'][NAT]

        moranLoc.Is.max()
        
        (moranLoc.p_sim)<0.05
        (moranLoc.p_z_sim*2)<0.05
        
        # Prepare values for plotting 
        coords = data['coords'][NAT]
        feature = data['feature'][NAT]
        moran = data['moran'][NAT]
        maxproj = data['maxproj']
        x_edges = data['x_edges']
        y_edges =  data['y_edges']
        
        # Plot the figure of bin correlations and local Moran's Is
        fig, ax = plt.subplots(2,2, figsize = (8,5.4))
        for axes in ax.flatten():
            sns.heatmap(maxproj, ax = axes, cmap = 'gist_gray', square = True, 
                            cbar = False, yticklabels = False, xticklabels = False, alpha = 0.65, rasterized = True)
            axes.set_aspect('equal')
            
            for ii in range(1,len(x_edges)): 
                axes.vlines(x_edges[ii], 0, y_edges[-1]+y_edges[1]/2, color = 'darkgray', linewidth = 1.5)
                axes.hlines(y_edges[ii], 0, x_edges[-1]+x_edges[1]/2, color = 'darkgray', linewidth = 1.5)   
        
        # Scatter bin correlations
        vEx = abs(feature).max()    
        norm = plt.Normalize(-vEx, vEx)
        sm = plt.cm.ScalarMappable(cmap="vlag", norm=norm)    
            
        ax[0,0].scatter(coords[:,1], coords[:,0], s = 50, c = feature, cmap = 'vlag', vmin = -vEx, vmax = vEx)
        ax[0,0].figure.colorbar(sm, ax = ax[0,0], shrink = 0.95, label = 'Bin correlation (r)')
        ax[0,0].set_title('Bin correlation (r)')
        
        # Scatter local Moran's I
        vEx = abs(moranLoc.Is).max()    
        norm = plt.Normalize(-vEx, vEx)
        sm = plt.cm.ScalarMappable(cmap="vlag", norm=norm)    
            
        ax[0,1].scatter(coords[:,1], coords[:,0], s = 50, c = moranLoc.Is, cmap = 'vlag', vmin = -vEx, vmax = vEx)
        ax[0,1].figure.colorbar(sm, ax = ax[0,1], shrink = 0.95, label = 'Local Moran\'''s I')
        ax[0,1].set_title('Local Moran\'''s I')
        
        # Scatter p-values
        # pAdjust = false_discovery_control(moranLoc.p_sim, axis=0, method='by')
        localM_sig, pAdjust = fdrcorrection(moranLoc.p_sim, 0.05, 'indep', False)
        # localM_sig = (moranLoc.p_sim)<0.05 # Boolean
        
        # vEx = abs(moranLoc.p_sim).max() 
        # norm = plt.Normalize(0, vEx)
        # sm = plt.cm.ScalarMappable(cmap='OrRd_r', norm=norm)
        
        # ax[1,0].scatter(coords[:,1], coords[:,0], s = 50, c = moranLoc.p_sim, cmap = 'OrRd_r', vmin = 0, vmax = vEx)
        # ax[1,0].figure.colorbar(sm, ax = ax[1,0], shrink = 0.95, label = 'p-value')
        # ax[1,0].set_title('p-value')
        
        ax[1,0].scatter(coords[:,1][localM_sig], coords[:,0][localM_sig], s = 50, c = contrast[220], label = 'p < 0.05')
        ax[1,0].scatter(coords[:,1][~localM_sig], coords[:,0][~localM_sig], s = 50, c = 'darkgrey', label = 'p ≥ 0.05')
        # ax[1,0].figure.colorbar(sm, ax = ax[1,0], shrink = 0.95, label = 'p-value')
        ax[1,0].legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        ax[1,0].set_title('p-value')
        
        # Scatter of significant values
        # vEx = abs(moranLoc.Is).max()    
        # norm = plt.Normalize(-vEx, vEx)
        # sm = plt.cm.ScalarMappable(cmap="vlag", norm=norm)    
            
        ax[1,1].scatter(coords[:,1][localM_sig], coords[:,0][localM_sig], s = 50, c = moranLoc.Is[localM_sig], cmap = 'vlag', vmin = -vEx, vmax = vEx)
        # ax[1,1].figure.colorbar(sm, ax = ax[1,1], shrink = 0.95, label = 'Local Moran\'''s I')
        ax[1,1].set_title('Significant local Moran\'''s I')
        
        plt.tight_layout()
        
        plt.savefig(savepath+key+'-'+NAT+'.svg', format = 'svg')   

