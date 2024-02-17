# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:52:17 2023

@author: torstsl

Calculate how many cells are in each session, how many are PCs and in which 
condition. Filter by events and SNR.

"""

import numpy as np
import pickle

#%% 
def count_PCs(session_dict):
    
    minSNR = 3
    minEvents = 40
    
    SNR = session_dict['ExperimentInformation']['CellSNR']
    events = session_dict['ExperimentInformation']['EventCount_raw']
     
    nCells = np.sum((SNR >= minSNR) & (events >= minEvents))
    
    placecell_dict = session_dict['Placecell']
    placecells = np.unique(np.concatenate([placecell_dict['NAT0'][0],np.unique(np.concatenate([placecell_dict['NAT0'][0],placecell_dict['NAT1'][0],placecell_dict['NAT2'][0]]))    ,placecell_dict['NAT2'][0]]))    
    
    PC_A = placecell_dict['NAT0'][0]
    PC_A2 = placecell_dict['NAT2'][0]
    PC_AA2 = np.unique(np.concatenate([placecell_dict['NAT0'][0],placecell_dict['NAT2'][0]]))    
    PC_B = placecell_dict['NAT1'][0]
    
    results_dict = {'nCells': nCells, 'PCs': placecells, 'PC_A': PC_A, 'PC_B': PC_B, 'PC_A2': PC_A2, 'PC_AA2': PC_AA2}

    return  results_dict

#%%
results_folder = r'C:\Users\torstsl\Projects\axon2pmini\results'

with open(results_folder+'/sessions_overview.txt') as f:
    sessions = f.read().splitlines() 
f.close()

# Iterate over sessions and count all PCs
count_PC_dict = {}  
  
for session in sessions: 
    
    # Load session_dict
    session_dict = pickle.load(open(session+'\session_dict.pickle','rb'))
    print('Successfully loaded session_dict from: '+str(session))
    
    # Perform the analysis and return the output
    key = session_dict['Animal_ID'] + '-' + session_dict['Date']
    count_PC_dict[key] = count_PCs(session_dict)
    
    # Store the output
    with open(results_folder+'/count_PCs.pickle','wb') as handle:
        pickle.dump(count_PC_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Successfully saved results_dict in '+results_folder)