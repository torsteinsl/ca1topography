# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 11:28:38 2023

@author: torstsl

Store place fields in the session dict to make future analysis easier. 
Also, make them more similar. Where nans occur for defined place cells, there 
is something fishy. Plotting these cells reveal that some are indeed nice 
place cells, others are more on the edge (litterarily and figuratively <-- 
did not spell those correct) and could be false positives. It is just an issue 
for 16/6617 cells , so I guess it will have to just go.

    - Create list of all session paths
    - Load session_dict directly from session paths
    - Calculate all place fields for all cells per session
    - If it can't be found, make it a nan
    - Store place fields in session_dict
    - Index in dict corresponds to cell number of total cells (NATEX ID)
    - Save new session_dict

"""
import numpy as np
import opexebo as op
import pickle
import matplotlib.pyplot as plt
from os import path

#%% Load sessions to run through
basefolder = 'N:\\axon2pmini\\Recordings\\'

session_folder = ['100867\\190822','100867\\020922','100867\\090922','100867\\160922',
                  '101258\\180822','101258\\250822','101258\\010922','101258\\080922','101258\\150922',
                  '102121\\260922','102121\\031022','102121\\101022','102121\\181022','102121\\241022',
                  '102123\\051022','102123\\121022','102123\\191022','102123\\261022','102123\\021122',
                  '102123\\090123_ML000_AP-400','102123\\110123_ML-400_AP-400','102123\\130123_ML-400_AP000',
                  '102124\\280922','102124\\051022','102124\\121022','102124\\191022','102124\\261022',
                  '102124\\191222_ML000_AP+400','102124\\201222_ML+400_AP000','102124\\211222_ML+400_AP+400','102124\\221222_ML+400_AP+800']
dir_name = []    
for x in session_folder: dir_name.append(basefolder+x)   
 
#%% Run through all sessions   
lostPC = 0
   
for x in dir_name: 
    
    if path.isfile(x+'/session_dict.pickle')==False:
        raise TypeError('No session_dict located in this folder: '+str(x))
    else: 
        session_dict = pickle.load(open(x+'/session_dict.pickle','rb'))
        print('Successfully loaded session_dict from: '+str(x))

    # Initiate variables
    nSessions = len(session_dict['dfNAT'])
    placecell_dict = session_dict['Placecell']
    ratemaps_dict = session_dict['Ratemaps']
    nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)
    
    minPeak = 0.2
    placefield = {}
    
    # Get the place fields for all place cells per session
    for sessionNo in range(nSessions):
        key = 'NAT'+str(sessionNo)
        placefield[key]={}
        PC = placecell_dict[key][0]
        
        for cellNo in range(1, nCells+1): # NATEX index    
           
            # Calculate the place field stats
            ratemap = ratemaps_dict['df'+key]['N'+str(cellNo)]
            
            calc_fields = op.analysis.place_field(ratemap, search_method = 'sep', min_peak = minPeak)
            
            # Check there were no calculated field(s), if so: Try again with a boost
            if calc_fields[0]:
                placefield[key]['N'+str(cellNo)] = calc_fields 
            
            if not calc_fields[0]:
                calc_fields = op.analysis.place_field(ratemap*1/np.max(ratemap), search_method = 'sep', min_peak = minPeak)
                
                if calc_fields[0]:
                    placefield[key]['N'+str(cellNo)] = calc_fields 
           
                elif not calc_fields[0]:
                    placefield[key]['N'+str(cellNo)] = np.nan
                    
                    if cellNo in PC:
                        print('Could not find a place field in this session: '+key+', cell '+str(cellNo))
                        plt.figure(), plt.imshow(ratemap), plt.colorbar(), plt.title(str(key)+' N'+str(cellNo)), plt.axis(False)
                        lostPC += 1 

    # Store results
    session_dict['Placefields'] = placefield
         
    with open(x+'/session_dict.pickle','wb') as handle:
        pickle.dump(session_dict,handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Successfully saved place fields in session_dict')
    
print('Total loss of place cells: '+str(lostPC))    
