# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 09:42:18 2023

@author: torstsl

Moran's I is a statistic used to measure spatial autocorrelation, which refers 
to the relationship between the values of variables in space. To calculate 
Moran's I, you will need:

1)  A set of spatial data, such as a raster or a point dataset, that represents 
    the variable of interest
2)  A spatial weight matrix, which represents the spatial relationships
    between the observations in the data set
3)  The mean of the variable of interest

   The formula for calculating Moran's I: 

                 n     sum(wij * (xi - xmean) * (xj - xmean))
    Moran's I = --- x ------------------------------------------ 
                 W              sum(xi - xmean)^2)
    
   where:

    n       is the number of observations in the data set
    i, j    is the indecies for the points in the data set (back and forth whether i == j is valid or not)
    wij     is the weight associated with the relationship between observations 
            i and j
    W       is the sum of all weights (sum(wij) for all i and j)
    xi      is the value of the variable of interest for observation i
    xmean   is the mean of the variable of interest
    
Note that the interpretation of Moran's I is dependent on the spatial weight 
matrix used, so it is important to choose a weight matrix that accurately 
represents the spatial relationships in the data.


This script contains the following: 
    - A function to calculate global Moran's I, moran(x,y,featyre,weights,nRep)
    - Two helper functions to the Moran's I calculation, where one does the 
        actual calcualtion and the other does bootstrapping to estimate a 
        p-value and significance of the results. 
    - An example created from random data to demonstrate the function, with
        particular emphasis on the calculation of the spatial weights
    - A final usage controlling the above using the package esda.Moran to 
        calculate both theglobal and local Moran's I. Both examples uses
        bootstrapping and visualizes the results. 
    - The manual calculation does not match the esda.Moran perfectly, but they
        seem to agree on the rejection of H0 - just with somewhat different Is. 
        For future usage, sticking to the package of esda is probably wise.

"""

import numpy as np
import matplotlib.pyplot as plt
import esda
import libpysal.weights

#%% Moran I's statistics by "hand"

def moran(x, y, feature, weights, nRep):
    """

    Parameters
    ----------
    x : x-coordinates of the data points. 
    y : y-coordinates of the data points. 
    feature : The feature which is to be tested; hence is this feature randomly 
              distributed among the population or not. 
    weights : The weights matrix, size NxN where N is the number of data points. 
              Can't contain NaN values. The example weights are calculated as
              1/distance for each single data point in the scatter. 
              The diagonal (wij, i==j) is set to 0.
    nRep : Number of permutations to perform in bootstrapping.

    Returns
    -------
    moranStats : Global Moran's I test statistic:
        [0] = Moran's I
        [1] = Expected value (E[I])
        [2] = Variance of I (var[I])
        [3] = z-statistic of I
        [4] = p-value under permutation bootstrapping
    replicates : I values from the permutation bootstrapping.

    """
    # Error if there are missing values in the spatial weights
    if np.sum(np.isnan(weights)) > 0:
        raise TypeError('NaN values in weights for cell pairs')
    
    # Do test statistics
    morani, exp, var, stat = moran_stats(x, y, feature, weights)
    
    # Do bootstrapping permutations, estimate p-value
    replicates = moran_boot(x, y, feature, weights, nRep)
    pval = ((np.sum(replicates > morani)+ 1) / (nRep+1)) 
    
    # Return statistics
    moranStats = morani, exp, var, stat, pval
    
    return moranStats, replicates
        
def moran_stats(x, y, feature, weights):
    
    """
    This uses the following formula, which is not fully correct as it 
    is missing the sum of the spatial weights in the denominator:
        
    (n * sum(wij * (xi - xmean) * (xj - xmean))) / (sum((xi - xmean)^2))
    
    When adding the sum of the weights, the I's get really low. Is is just
    scaled, but it doesn't look right. It is therefore not added to the 
    function, as it appears more stable without that part. The resulting I 
    is still not quite similar to esda.Moran (not sure why), but they seem to
    agree with resuls and rejection H0, it just that the actual I values are 
    not exacytly the same.
    
    Also, this omits cases where i==j, so that the self values are not used
    in the actual calcualtion. These should not be given in the weights 
    matrix either, or should be set to 0 (not NaN) as they influence the sum
    of all weights potentially used in the denominator.
    
    """
    # set up variables to calculated the global Moran's I
    n = len(x)
    fmean = np.nanmean(feature)
    fsum = 0
    denomsum = 0

    for i in range(n):
        for j in range(n):
            if not i == j: # Omits self values completely (even if w[i,j]=0)
                fsum += weights[i,j]*((feature[i]-fmean)*(feature[j]-fmean))
                denomsum += (feature[i]-fmean)**2           
    
    # sumW = np.nansum(w) # If to divide by the sum of the weights?
    
    # Get Moran's I
    morani = (n*fsum)/(denomsum)# *sumW) # Add *sumW in denominator here?
    
    # Get test statistics: Expected value, variance and test statistic
    exp = -1/(n-1)
    var = 1/(n*(n-3))
    stat = (morani - exp) / np.sqrt(var)
    
    return morani, exp, var, stat   

def moran_boot(x, y, feature, weights, nRep):
    
    # Does bootstrapping permutation to estaimate a p-value, returns the permuted Is    
    replicates = np.empty(nRep)
    
    for i in range(nRep):
        resampled_data = np.random.choice(feature, size=len(feature), replace=False)
        replicates[i] = moran_stats(x, y, resampled_data, weights)[0]
        
    return replicates      

#%% Generate random data
x = np.random.randint(0,100,100) 
y = np.random.randint(0,100,100) 
f = np.random.rand(100) 

# Calculate the weights matrix: Defined by 1/distance between points in scatter
coords = np.array([x, y]).T
dist = np.linalg.norm(coords - coords[:,None], axis = -1)
dist[np.diag_indices_from(dist)] = np.nan # To omit devide by 0 warning

# Calculate the weights, set diagonal to 0 (wij, i==j --> 0).
w = 1/dist
w[np.diag_indices_from(w)] = 0

# Because these are random point, they sometimes become the same, not doable
if np.sum(np.isinf(w))>0:
    print('Divide by 0 in weights, infinite value occurs')
#%% Calculate Moran's I and plot results
moranStats, shuffles = moran(x, y, f, w, nRep=999)

fig, ax = plt.subplots(1,2, figsize = (10,5))
plt.suptitle('Moran\'s I: Random data')

ax[0].scatter(x, y, s = f) 
ax[0].set_aspect('equal')

ax[1].hist(shuffles, 25, color = 'darkgrey')
plt.axvline(moranStats[0], color = 'crimson')
plt.axvline(np.percentile(shuffles,95), color = 'grey')
ax[1].set_title('I: '+str(round(moranStats[0],2))+'; p: '+str(round(moranStats[4],2)))
plt.tight_layout()

#%% Generate clustered data
f_cluster = f.copy()
f_cluster[np.where((x>40) & (x<60) & (y>20) & (y<80))] = 10

moranStats_cluster, shuffles_cluster = moran(x, y, f_cluster, w, nRep=500)

fig, ax = plt.subplots(1,2, figsize = (10,5))
plt.suptitle('Moran\'s I: Clustered data')

ax[0].scatter(x, y, s = f_cluster) 
ax[0].set_aspect('equal')

ax[1].hist(shuffles_cluster, 25, color = 'darkgrey')
plt.axvline(moranStats_cluster[0], color = 'crimson')
plt.axvline(np.percentile(shuffles_cluster,95), color = 'grey')
ax[1].set_title('I: '+str(round(moranStats_cluster[0],2))+'; p: '+str(round(moranStats_cluster[4],2)))
plt.tight_layout()

#%% Control using esda.moran

"""
The package esda.Moran allows fast calculations of global and local Moran's I.
It is most smoothly used with a matrix without missing data and where there 
are defined borders between neighbours (mostly used in geographical data 
analysis). It can also be used on scatter data, with a little work around the 
spatial weights, which must be on the format given by:
    
    W = libpysal.weights.W(neighbours, weights)

This function basically provides a dictionary with each element, in which the 
neighbours for each element is listed as well as the spatial weight between 
them. To make this work for scatter data, I make every data point neighbour to 
all other data points, and give them a weight according to 1/distance between 
the data points. With the parameters neighbours and weights the weights object
W is calculated, and fed into the esda.moran together with the feature to be 
analysed. This provides the given statistics with varibales and permutations 
for an estimated p-va
s
"""

"""
# Alternative weight calculation accepting i==j

keys = []
for i in range(len(f)): keys.append(i)
keys = np.array(keys)

neighbours, weights = {}, {}
for i in range(len(f)): 
    neighbours[keys[i]] = keys
    weights[keys[i]] = w[i, keys]
"""

# Calculate weight where instances of i==j are not accepted
keys = []
for i in range(len(f)): keys.append(i)
keys = np.array(keys)

neighbours, weights = {}, {}
for i in range(len(f)): 
    others = keys[~np.isin(keys,i)]
    neighbours[keys[i]] = others
    weights[keys[i]] = w[i, others]

W = libpysal.weights.W(neighbours, weights)

# Calculate Moran's I 
moran_esda = esda.moran.Moran(f, W) # Random data
moran_esda_cluster = esda.moran.Moran(f_cluster, W)
moran_loc_esda = esda.Moran_Local(f,W)

# Plot bootstrap permutations
fig, ax = plt.subplots(1,2)
plt.suptitle('Moran\'s I: esda.Moran')

ax[0].hist(moran_esda.sim, 25, color = 'darkgrey')
ax[0].axvline(moran_esda.I, color = 'crimson')
ax[0].axvline(np.percentile(moran_esda.sim,95), color = 'grey')
ax[0].set_title('Random: I: '+str(round(moran_esda.I,2))+'; p: '+str(round(moran_esda.p_sim,2)))

ax[1].hist(moran_esda_cluster.sim, 25, color = 'darkgrey')
ax[1].axvline(moran_esda_cluster.I, color = 'crimson')
ax[1].axvline(np.percentile(moran_esda_cluster.sim,95), color = 'grey')
ax[1].set_title('Clustered: I: '+str(round(moran_esda_cluster.I,2))+'; p: '+str(round(moran_esda_cluster.p_sim,2)))

plt.tight_layout()

print(moranStats[0::4])
print(moran_esda.I, moran_esda.p_sim)

# Calculate spatial lags
lags = libpysal.weights.lag_spatial(W, f)
