from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp

dataframe = read_csv('Data set/using data/dataset of nearing stations for PCA.csv', engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

table = np.empty([dataset.shape[0]-1,5],dtype = float)
table[:,:] = np.nan

for i in range(1,dataset.shape[0]):
    fit = sp.linregress(dataset[0,:],dataset[i,:])
    corr = np.corrcoef(dataset[0,:],dataset[i,:])
    table[i-1,0] = corr[0,1]
    table[i-1,1] = fit.slope
    table[i-1,2] = fit.intercept
    table[i-1,3] = fit.rvalue
    table[i-1,4] = fit.stderr

