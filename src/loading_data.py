# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 13:43:03 2022

@author: torstsl

This file contains functions used to load and extract data from 2PMINI
and saving the wanted data in usable files for later analysis.

"""    
import numpy as np
import pandas as pd
import pickle
import scipy.io as sio
import tkinter
from tkinter import filedialog
from src.cell_activity_analysis import calc_session_ratemaps
from os import path
from mat73 import loadmat

#%%

def GUI_choose_directory():
  
    """
    # Opens the file explorer GUI and lets user pick a directory - name of which is returned 
    
    # @author: Online, reviewed from Hanna Eneqvist
    """
        
    root = tkinter.Tk()
    root.withdraw()
    dirname = filedialog.askdirectory(parent=root, initialdir="N:/axon2pmini/Recordings/", title='Please select a directory')

    if len(dirname ) > 0:
        print ("You chose directory %s" % dirname)

    if len(dirname) == 0:
        print ("Error; directory not found")

    root.destroy()    
    
    return dirname

#%%
def GUI_choose_files():
  
    """
    # Opens the file explorer GUI and lets user pick files - path of which are returned 
    
    # @author: Online, reviewed from Hanna Eneqvist, adapted by Torstein Slettmoen
    """
    
    root = tkinter.Tk()
    root.withdraw()
    filenames = tkinter.filedialog.askopenfilenames(parent=root, initialdir="N:/axon2pmini/Recordings/", title='Please select a directory')

    if len(filenames) == 0:
        print("Error; file not found")

    root.destroy()    
    
    return filenames

#%%
def suite2p_ROI_ID_converter(**kwargs):
    
    """
    Created on Wed Jun 15 19:17:46 2022

    @author: torstsl

    Function in which you write the current idx of your ROI, and it gives you 
    the original ROI from suite2p (before filtering for true ROIs). This is handy
    to easily check data and code between raw and filtered.

    13.12.22: Function call does not work. Logic in ids should still be working.

    ids = suite2p_ROI_ID_converter(iscell)

    OUTPUT:
    ids
        Column 0: ROI ID for true cells when filtering by iscell.npy from suite2p
                  Return nan if the ROI is not classified as a cell in iscell.npy
                  This number is corresponds to the NATEX ID.   
        Column 1: ROI ID directly from suite2p
        
        Ex.: ids[10,0] is the filtered ID (NATEX) of the ROI corresponding
             to ROI ids[10,1] in suite2p
             
             Finding NATEX cell 8's suite2p ID:
             ids[np.where(ids[:,0]==8)] 
      
    INPUT:
        **kwargs: If none provided, a GUI opens to locate the iscell.npy file
            iscell: The suite2p output file called iscell.npy for a certain dataset
            session_dict: Session dictionary with iscell stored within
    """
    if len(kwargs) == 0:             
        iscell_path = GUI_choose_files()[0]
        iscell = np.load(iscell_path, allow_pickle=True)
    
    elif len(kwargs) > 0:
        if "iscell" in kwargs:
            iscell = kwargs['iscell']
        elif 'session_dict' in kwargs: 
            iscell = kwargs['session_dict']['ExperimentInformation']['IsCell']
            print(iscell)
            iscell[:,1] = 0
    
    ids=np.zeros([len(iscell[:,0]),2])
    count = 0

    for k in range(0,len(iscell[:,0])):
        if iscell[k,0] == 1:
            ids[k,0] = k-count+1    #k - count = filtered ID (NATEX starts at 1)
            ids[k,1] = k            #k = original ID
        elif iscell[k,0] == 0:
                ids[k,0] = np.nan   #this ROI is not filtered as a cell
                ids[k,1] = k        #k = original ID
                count+=1
        
    return ids

#%%
def generate_session_dict(animal_id, date, **kwargs):   
    
    """
    Created on Fri Jul  8 09:23:10 2022

    @author: torstsl

    Generates a dictionary (session_dict) that contains all the information 
    elsewise stored in NAT.mat and ExperimentInformation.mat as generated by 
    NATEX. Path is defined by a GUI, and the dictionary can be saved in the 
    same folder as a picklefor later loadings and analyses     

    session_dict = generate_session_dict(animal_ID, date, save_dict, **kwargs)

    OUTPUT:
        session_dict: Dictionary containing:
                ExperimentInformation (dict): All info in ExperimentInformation.mat,
                    including session raw path, planes, SNR, cellStat, iscell,
                    eventcount, total cell number etc.
                    
                dfNAT (dict/df): All info from NAT.mat, incuding timestamps, tracking
                    and cell activity. For sessions with several subsessions the 
                    variable is a dictionary with df for each subsession's NAT.
                Animal_ID (str): mLims animal ID
                Date (str): Date of recording, as format ddmmyy
                Ratemaps (dict): Ratemaps for all cells per subsession in this session.
                Place_cell (dict): Array with each cell ID that is concidered 
                    a place cell after shufffling (MATLAB function).
    INPUT:
        animal_id (str): mLims animal ID/number
        date (str): Date of recording, ddmmyy
        *kwargs:  
            save_dict (boolean): If True, the session_dict is saved as a 
                pickle in the path folder.
            ratemaps (boolean): If True, the function calls up get_session_ratemaps, 
                calculates ratemaps, and stores them within the session_dict
      
    """
       
    print('Please choose session directory')
    dir_name = GUI_choose_directory()

    file_dict, file_NAT, file_expInf = path.isfile(dir_name+'/session_dict.pickle'),path.isfile(dir_name+'/NAT.mat'),path.isfile(dir_name+'/ExperimentInformation.mat')
     
    if file_dict == True and kwargs['save_dict'] == True: 
        raise TypeError('Given path already has a session_dict, please delete this file before rerunning')
    elif file_NAT == False or file_expInf == False:
        raise TypeError('Given path does not have a NAT.mat or ExperimentInformation.mat')

    NAT = loadmat(dir_name+'/NAT.mat')['NAT']
    experimentInfo = loadmat(dir_name+'/ExperimentInformation.mat')

      # NAT-key (from MATLAB):
      # Column 1:  timestamp
      # Column 2:3 headposition
      # Column 4:  headdirection
      # Column 5:  headspeed
      # Column 6:  headvilid
      # Column 7:8 bodyposition (x,y)
      # Column 9:  bodydirection
      # Column 10: bodyspeed
      # Column 11: bodyvilid
      # Column 12: reserved
      # Column 13:4:13+N*4-1+neuron 
      #   Column 13+N*4-1+neuron: dF/F raw, corrected by neuropil (Fcorr) and baseline fluorescence (F0(t)) 
      #   Column 14+N*4-1+neuron: dF/F filtered by significant traces (above signal filtered by below boolean)
      #   Column 15+N*4-1+neuron: Boolean for wether or not this frame has a significant transient. 
      #                           (1) Significant traces are defined as: dF/F exceeding 2x the local std of baseline fluorescence   
      #                           (2) This lasts for >0.75 s
      #   Column 16+N*4-1+neuron: Deconvolved spikes, filtered by significant traces
      #   See: https://www.sciencedirect.com/science/article/pii/S0092867422001970#sec5.5.2
      
    
    if len(NAT) > 1: # Generates session_dict if there are subsession within this session, stores all NATs
        print('Found '+str(len(NAT))+' subsessions within this session. Generating DataFrames for all within session_dict')
        
        dfNAT_sess = {}
        
        for sess in range(0,len(NAT)):
            dfNAT = pd.DataFrame(NAT[sess])
            dfNAT = pd.DataFrame.rename(dfNAT,columns={0:'Timestamps',1:'Head_X', 2: 'Head_Y', 
                                             3: 'HD',4:'Head_speed',5:'DLCHeadConf',
                                             6:'Body_X', 7:'Body_Y', 8:'Body_dir', 
                                             9:'Body_speed',10:'DLCBodyConf',11:'Reserved'})

            uniques = np.arange(12,len(dfNAT.columns)-1,4)
            neuron = 1

            for k in uniques:
                dfNAT = pd.DataFrame.rename(dfNAT,columns={k:'dF raw, N'+str(neuron),
                                                 k+1:'dF filtered, N'+str(neuron),
                                                 k+2: 'Significant trace, N'+str(neuron),
                                                 k+3: 'Deconvolved, N'+str(neuron)})
                neuron +=1                                         
            
                dfNAT_sess['dfNAT'+str(sess)] = dfNAT
            
        session_dict = {'ExperimentInformation': experimentInfo['ExperimentInformation'],
                            'dfNAT': dfNAT_sess,
                            'Animal_ID':animal_id,
                            'Date':date} 
                 
      
    elif len(NAT) == 1: # If just one session
        dfNAT = pd.DataFrame(NAT[0])
        dfNAT = pd.DataFrame.rename(dfNAT,columns={0:'Timestamps',1:'Head_X', 2: 'Head_Y', 
                                     3: 'HD',4:'Head_speed',5:'DLCHeadConf',
                                     6:'Body_X', 7:'Body_Y', 8:'Body_dir', 
                                     9:'Body_speed',10:'DLCBodyConf',11:'Reserved'})

        uniques = np.arange(12,len(dfNAT.columns)-1,4)
        neuron = 1

        for k in uniques:
            dfNAT = pd.DataFrame.rename(dfNAT,columns={k:'dF raw, N'+str(neuron),
                                         k+1:'dF filtered, N'+str(neuron),
                                         k+2: 'Significant trace, N'+str(neuron),
                                         k+3: 'Deconvolved, N'+str(neuron)})
            neuron +=1                                         
    
        session_dict = {'ExperimentInformation': experimentInfo['ExperimentInformation'],
                    'dfNAT': dfNAT,
                    'Animal_ID':animal_id,
                    'Date':date} 
    
    # Load and add place cell info if available
    if path.isfile(dir_name+'/IsPCCell.mat') == True:
        mat_dict = sio.loadmat(dir_name+'/IsPCCell.mat')
        
        placecell = {}
        
        for sess in range(0,len(NAT)):
            placecell['NAT'+str(sess)] = mat_dict['IsPCCell'][sess][0][0]
            
        session_dict['Place_cell'] = placecell      
    elif path.isfile(dir_name+'/IsPCCell.mat') == False:
        print('No Matlab place cell data saved at given directory, not added to dict')
    
        
    # Deal with **kwargs    
    if kwargs['ratemaps'] == True:
        print('Calculates ratemaps')
        session_ratemaps = calc_session_ratemaps(session_dict)
        session_dict['Ratemaps'] = session_ratemaps
        
    if kwargs['save_dict'] == True:
        with open(dir_name+'/session_dict.pickle','wb') as handle:
            pickle.dump(session_dict,handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Successfully saved session_dict in '+dir_name)
    elif kwargs['save_dict'] == False:
        print('Output session_dict was generated, but not saved')
   
    return session_dict         

#%%
def load_session_dict():
    """
    Created on Fri Jul  8 17:15:44 2022

    @author: torstsl
    
    Loads session_dict which is stored as a pickle from generate_session_dict
    
    session_dict = load_session_dict()
    
    OUTPUT:
        session_dict: See generate_session_dict       
    
    """
      
    dir_name = GUI_choose_directory()

    if path.isfile(dir_name+'/session_dict.pickle')==False:
        raise TypeError('No session_dict located in this folder')
    else: 
        session_dict = pickle.load(open(dir_name+'/session_dict.pickle','rb'))
        print('Successfully loaded session_dict from pickle')
    return session_dict

#%%
def load_results_dict():
    """
    Created on Thurs Feb  2 12:16:48 2023

    @author: torstsl
    
    Loads results_dict which is stored as a pickle.
    
    results_dict = load_results_dict()     
    
    """
      
    dir_name = r'C:\Users\torstsl\Projects\axon2pmini\results'

    if path.isfile(dir_name+'/results_dict.pickle')==False:
        raise TypeError('No results_dict located in this folder')
    else: 
        results_dict = pickle.load(open(dir_name+'/results_dict.pickle','rb'))
        print('Successfully loaded results_dict from pickle')
    return results_dict
#%%
def append_session_overview(session_dict):
    
    """
    Created on Fri Jul  9 14:42:15 2022

    @author: Internet, adapted by torstsl
    
    append_session_overview(session_str)
    
        A function that is meant to keep track of all sessions that are put
        into downstreams analyzes. The file is hard coded in the results-folder
        of this package. The function writes the raw data folder into the end 
        of the file. If not existing, this file generates a 
        sessions_overview.txt in the results-folder of this package.
        
        The sole purpose of this is to append data that is used for actual
        publications, so that it is easy to find the data used for the paper
        when it comes to that. You're doing your future self a big service!
    
        INPUT: session_dict (dict): Session dictionary as created in this 
                                    package. The function grabs the raw data
                                    path from the dictionary and appends this.
    
    """
    path = 'C:/Users/torstsl/Projects/axon2pmini/results/sessions_overview.txt'
    session_str = session_dict['ExperimentInformation']['RawDataAddress']
    
    # Open the file in append & read mode ('a+')
    with open(path, "a+") as file_object:
        
        # Move read cursor to the start of file.
        file_object.seek(0)
        
        # If file contains session_str already, return error
        if session_str in file_object.read():
            raise TypeError('This sessions is already in sessions_overview.txt')
        
        # Move read cursor to the start of file.
        file_object.seek(0)
        
        # If file is not empty then append '\n'
        if len(file_object.read(100)) > 0:
            file_object.write("\n")
            
        # Append text at the end of file
        file_object.write(session_str)
        print('Successfully added '+str(session_str)+' to '+path)

#%%        
"""
Created on Mon Sep  5 12:05:44 2022

@author: torstsl

    merge_tracking_csv(filename)

    This function is meant to help merge tracking .csv files output from DLC.
    NATEX does not tolerate several tracking files if there are more sessions
    per experiment. Since it sometimes is necessary to abort the recording, 
    and this generates a tracking lag - it is sometimes necessary to change 
    the tracking to synchonize with the imaging data. 
    
    This function does not deal with excess tracking datapoints (meaning the 
    misalignment of tracking and imaging that occurs when aborting). This must
    be fixed manually (by deleting excess rows from csv that does not have 
    matching tiffs) or be adddded to the code. 
    
    So, if you have control of this - you can merge the files using this 
    function. Viel glück!

    INPUT: filename (str):  Name of the new tracking.csv to be saved.

"""
def merge_tracking_csv(filename):
         
     # Get files
     print('Please choose directory to save the new csv')
     dirname = GUI_choose_directory()
     
     file_dir = path.isfile(dirname+'/'+filename)
     if file_dir == True: 
         raise TypeError('Given path already has a file '+filename+', please delete this file before rerunning')
     
     print('Please choose the csv files you want to merge')
     filenames = GUI_choose_files()
     df1 = pd.read_csv(filenames[0])
     df2 = pd.read_csv(filenames[1])
     df2 = df2.drop([0,1])
     df2 = df2.reset_index(drop=True)

    # Change the second csv before merging it with the first, and save
     df2['scorer'] = pd.to_numeric(df2['scorer'])
     df2['scorer'] = df2['scorer']+len(df1['scorer'])-2 # -2 because bodyparts and coords are two
     df2['scorer'] = df2['scorer'].apply(str)

     new_tracking = pd.concat([df1,df2])
     new_tracking = new_tracking.reset_index(drop=True)
     
     new_tracking.to_csv(dirname+'/'+filename+'.csv',index=False)
     print('Sucessfully saved '+filename+'.csv to directory')       
     
     