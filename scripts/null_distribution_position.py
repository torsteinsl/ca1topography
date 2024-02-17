# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:45:19 2023

@author: torstsl

Quick script that generates a null distribution to say whether a decoder finds
a random or not point of position. 

Generates two distributions: A null for shuffles position data, and a null
for just random coordinates within the same grid as the position data. 

The shufffled data takes into account the behaviour of the animal, with both
occupancy and being a continuous distribution (no teleportation)

"""
 
import pickle
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

#%% Initiate data and variables

dirs = {'102124': [r'N:/axon2pmini/Recordings/102124/280922',
                   r'N:/axon2pmini/Recordings/102124/051022',
                   r'N:/axon2pmini/Recordings/102124/121022',
                   r'N:/axon2pmini/Recordings/102124/191022',
                   r'N:/axon2pmini/Recordings/102124/261022',
                   r'N:\axon2pmini\Recordings\102124\191222_ML000_AP+400',
                   r'N:\axon2pmini\Recordings\102124\201222_ML+400_AP000',
                   r'N:\axon2pmini\Recordings\102124\211222_ML+400_AP+400',
                   r'N:\axon2pmini\Recordings\102124\221222_ML+400_AP+800'],
        '102123': [r'N:/axon2pmini/Recordings/102123/051022',
                   r'N:/axon2pmini/Recordings/102123/121022',
                   r'N:/axon2pmini/Recordings/102123/191022',
                   r'N:/axon2pmini/Recordings/102123/261022',
                   r'N:/axon2pmini/Recordings/102123/021122',
                   r'N:\axon2pmini\Recordings\102123\090123_ML000_AP-400',
                   r'N:\axon2pmini\Recordings\102123\110123_ML-400_AP-400',
                   r'N:\axon2pmini\Recordings\102123\130123_ML-400_AP000'],
        '102121': [r'N:/axon2pmini/Recordings/102121/241022'],
        '101258': [r'N:/axon2pmini/Recordings/101258/010922'],
        '100867': [r'N:/axon2pmini/Recordings/100867/090922']
        }

dir_name = dirs['102124'][3]

session_dict = pickle.load(open(dir_name + '/session_dict.pickle','rb'))
placecell_dict = session_dict['Placecell']

nCells = session_dict['ExperimentInformation']['TotalCell'].astype(int)
nSessions = len(session_dict['dfNAT'])
session_string = ['A', 'B', "A'"]

scale = 1.1801 # Pixels/µm --> 1 pixel = 0.847.. µm
binning = 2.5 # cm/bin in ratemap
boxSize = 80 # cm
nBins = int(boxSize/binning)

palette = sns.color_palette("viridis", 256).as_hex()
contrast = sns.color_palette('OrRd', 256).as_hex()

#%% Create a null distribution of positional decoding by shuffling the positions based on one session

# Shuffle the headpos train
for sessionNo in range(nSessions):
    
    dfNAT = session_dict['dfNAT']['dfNAT'+str(sessionNo)]
    
    # Calculate the headpos for the entire session, set bin_edges for ratemaps from these
    headpos = (dfNAT[['Head_X','Head_Y']]).to_numpy()
    
    shuffles = np.random.randint(10, len(dfNAT)-10, 200)
    
    dist = np.full([len(dfNAT), len(shuffles)], np.nan) 
   
    for ii in range(len(shuffles)): 
        shift = shuffles[ii]
        headposShuffle = np.roll(headpos, shift, axis = 0)
    
        dist[:,ii] = np.linalg.norm(headpos - headposShuffle, axis=-1) 

# Plot the true and shuffled train: Should overlap perfectly
fig, ax = plt.subplots()    
ax.plot(headpos[:,0], headpos[:,1], c = 'blue', alpha = 0.5)
ax.plot(headposShuffle[:,0], headposShuffle[:,1], c = 'red', alpha = 0.5) 
ax.set_aspect('equal')

# Generate a true random distribution by taking random points
randPos = np.full([len(headpos), 2], np.nan)
randDist = np.full([len(dfNAT), len(shuffles)], np.nan) 

for jj in range(len(shuffles)):
    randPos[:,0] = np.random.uniform(headpos[:,0].min(), headpos[:,0].max(), len(headpos))
    randPos[:,1] = np.random.uniform(headpos[:,1].min(), headpos[:,1].max(), len(headpos))    
    
    randDist[:, jj] = np.linalg.norm(headpos - randPos, axis=-1) 
 
np.mean(dist.flatten())
np.median(dist.flatten())    
 
np.mean(randDist.flatten())
np.median(randDist.flatten())  

#%% Plot the results
fig, ax = plt.subplots()
sns.histplot(data = dist.flatten(), bins = 1000, color = palette[80])
ax.vlines(np.median(dist.flatten()), 0, 5000, color = contrast[200], alpha = 0.75, linewidth = 2,
          label = 'Median:' + str(round(np.median(dist.flatten()),2)) + 'cm')
ax.set_xlim(0)
ax.set(xlabel = 'Error (cm)', title = 'Chance level estimating position')
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon = False)

plt.savefig('N:/axon2pmini/Article/Figures/Figure 5/supp_chance_level.svg', format = 'svg')   

fig, ax = plt.subplots(1, 2, sharey = True) 
ax[0].hist(dist.flatten(),1000, color = palette[80])
ax[0].set_title('200 shuffles')
ax[0].set_xlim(0)
ax[0].set_ylabel('Counts')
    
ax[1].hist(randDist.flatten(),1000, color = palette[80])
ax[1].set_title('200 rand points')
ax[1].set_xlim(0)

plt.suptitle('Distance from headpos and shuffle/random headpos')
fig.supxlabel('Distance (cm)')
plt.tight_layout()
