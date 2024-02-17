# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:14:57 2023

@author: torstsl

For merging object information from Hanna into session_dict

"""

import pickle
from src.loading_data import generate_session_dict, load_session_dict

#%%
animal_id = '102124'
date = '281122'
newDate = '20221128'

generate_session_dict(animal_id, date, save_dict = True, ratemaps = True)

session_dict = load_session_dict()

print(session_dict['ExperimentInformation']['RawDataAddress'])

newPath = r'N:/axon2pmini/Recordings/CA1 - Objects/'+animal_id+'/'+newDate
print(newPath)

session_dict['ExperimentInformation']['RawDataAddress'] = newPath

with open(session_dict['ExperimentInformation']['RawDataAddress']+'/session_dict.pickle','wb') as handle:
    pickle.dump(session_dict,handle, protocol=pickle.HIGHEST_PROTOCOL)

print('Saved session_dict with new path at:\n'+newPath)

#%% Add object information to dict

# Load data paths
results_folder = r'C:\Users\torstsl\Projects\axon2pmini\results'

with open(results_folder+'/ObjectSessions_overview.txt') as f:
    objectSessions = f.read().splitlines() 
f.close()

#%% Add object data into session_dict

for session in objectSessions:
    
    print('Loading '+session)
    session_dict = pickle.load(open(session+'/session_dict.pickle','rb'))   
    
    if not  'ObjectData' in session_dict.keys(): 
        print('No object data in given session_dict')
    
        objectInfo = pickle.load(open(session+'\objectCellAnalysis.pkl','rb'))
        objectCellStats = {}
        
        for n in objectInfo['objectTunedCells']:
            objectCellStats['N'+str(n)] = objectInfo['objectCellInfo']['Cell_'+str(n)]
        
        ObjectData = {'ObjectPosX': objectInfo['objectPos'][1],
                      'ObjectPosY': objectInfo['objectPos'][0],
                      'ObjectMovedPosX': objectInfo['objectMovedPos'][1],
                      'ObjectMovedPosY': objectInfo['objectMovedPos'][0],
                      'ObjectTunedCells': objectInfo['objectTunedCells'],
                      'ObjectVectorCells': objectInfo['objectVectorCells'],
                      'ObjectCellStats': objectCellStats
                      }
        
        session_dict['ObjectData'] = ObjectData
    
        with open(session+'/session_dict.pickle','wb') as handle:
            pickle.dump(session_dict,handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saved session_dict with object information')
        
    else: print('Object data is already in this session_dict: '+session)
    
