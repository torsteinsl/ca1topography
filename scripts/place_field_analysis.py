# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: torstsl

This script loads a session_dict, and then visualises the place field detection
for the cells. It compares different minimum values to accept a field. The
field detection is based on sep, and is similar to the one used in BNT. The
minPeak value is originally given in Hz (made fro ephys data). Thus, the value
is somewhat arbitrary for imaging data.

From testing, it  appears that a minPeak of about 0.2 makes sense. 

"""

import numpy as np
import opexebo as op
from src.cell_activity_analysis import get_signaltracking, calc_tuningmap
from src.loading_data import load_session_dict
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
#%% Load and initiate
session_dict = load_session_dict()

nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)

placecells = session_dict['Placecell']
placecells = np.unique(np.concatenate([placecells['dfNAT0'][0],placecells['dfNAT1'][0],placecells['dfNAT2'][0]]))

placefield = {}

#%% Calculate place fields

minPeak = 0.01
altPeak1, altPeak2, altPeak3, altPeak4 = 0.1, 0.15, 0.2, 0.25

for sess in range(0,len(session_dict['dfNAT'])):
    placefield['dfNAT'+str(sess)]=[]
    
    timestamps  = (session_dict['dfNAT']['dfNAT'+str(sess)].Timestamps).to_numpy()
    headpos     = (session_dict['dfNAT']['dfNAT'+str(sess)][['Head_X','Head_Y']]).to_numpy()

    occupancy, coverage_prc, bin_edges = op.analysis.spatial_occupancy(timestamps, np.transpose(headpos),80, bin_number = 32)

    x_edges, y_edges = bin_edges[0], bin_edges[1]
    
    for cellNo in placecells:
        
        signaltracking = get_signaltracking(session_dict['dfNAT']['dfNAT'+str(sess)], cellNo,signal='deconvolved',speed_threshold = 2.5)

        tuningmap_dict = calc_tuningmap(occupancy,bin_edges[0],bin_edges[1],signaltracking,2.5)
        ratemap_smooth = gaussian_filter(tuningmap_dict['tuningmap'], 1.5)
        ratemap_smooth = np.ma.masked_array(ratemap_smooth,tuningmap_dict['tuningmap'].mask)
        
        fields,fields_map = op.analysis.place_field(ratemap_smooth,search_method='sep',min_peak = minPeak) # For testing
        fields2,fields_map2 = op.analysis.place_field(ratemap_smooth,search_method='sep',min_peak = altPeak1) # For testing
        fields3,fields_map3 = op.analysis.place_field(ratemap_smooth,search_method='sep',min_peak = altPeak2) # For testing
        fields4,fields_map4 = op.analysis.place_field(ratemap_smooth,search_method='sep',min_peak = altPeak3) # For testing
        fields5,fields_map5 = op.analysis.place_field(ratemap_smooth,search_method='sep',min_peak = altPeak4) # For testing

        placefield['dfNAT'+str(sess)].append([fields, fields_map])
        
        if cellNo < 25:# and sess == 0:
            fig, ([ax1,ax2,ax3],[ax4,ax5,ax6]) = plt.subplots(2,3)
            fig.suptitle('S'+str(sess)+' C'+str(cellNo))
            im1 = ax1.imshow(ratemap_smooth)
            im2 = ax2.imshow(fields_map)
            ax2.set_title('min_peak = '+str(minPeak))
            im3 = ax3.imshow(fields_map2)
            ax3.set_title('min_peak = '+str(altPeak1))
            im4 = ax4.imshow(fields_map3)
            ax4.set_title('min_peak = '+str(altPeak2))
            im5 = ax5.imshow(fields_map4)
            ax5.set_title('min_peak = '+str(altPeak3))
            im6 = ax6.imshow(fields_map5)
            ax6.set_title('min_peak = '+str(altPeak4))
            plt.tight_layout()