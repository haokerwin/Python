"""
@author haomj
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from pearson import multipl, corrcoef
from sklearn.metrics import mean_squared_error

mdir = './'
df = pd.read_csv(mdir+'data1.csv',header=None)
dh = df.dropna()
"""add data; note the 0-date, 1-obs_wind, 2-fst_wind
   3-fst_dir; 4-fst_pre; 5-fst_t2m; 6-fst_rh, 7-pre_wind
"""
obw0 = dh.ix[:,[1]] 
fcw0 = dh.ix[:,[2]] 
prw0 = dh.ix[:,[7]] 
obw1 = obw0.values
fcw1 = fcw0.values
prw1 = prw0.values
"""devide the train and test data sets using the time series
"""
l_total = len(dh)
l_train = int(0.7*l_total)
obw_train = obw1[0:l_train,:]
obw_test  = obw1[l_train:l_total,:]
fcw_train = fcw1[0:l_train,:]
fcw_test  = fcw1[l_train:l_total,:]
prw_train = prw1[0:l_train,:]
prw_test  = prw1[l_train:l_total,:]
time = list(range(len(obw_test)))
"""statistics
"""
rmse1 = np.sqrt(mean_squared_error(fcw_test,obw_test))
rmse2 = np.sqrt(mean_squared_error(prw_test,obw_test))
corr1 = corrcoef(fcw_test,obw_test)
corr2 = corrcoef(prw_test,obw_test)
print('the origional rmse is '+repr(rmse1))
print('the origional correlation coefficient is '+repr(corr1))
print('the learned rmse is '+repr(rmse2))
print('the learned correlation coefficient is '+repr(corr2))
"""plot curve
"""
sns.set_style ('darkgrid')
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_title('Wind Velocity') 
plt.xlabel('Time') 
plt.ylabel('Wind speed (m/s)')
plt.axis([0,len(obw_test),-5,35]) 
fig.set_size_inches(20.5,5.5) 
ax1.plot(time,obw_test,color='k',label='Observational',linestyle='-',linewidth=1)
ax1.plot(time,fcw_test,color='r',label='Origional',linestyle='-',linewidth=1)
ax1.plot(time,prw_test,color='c',label='Optimized',linestyle='-',linewidth=1)
plt.legend(loc='upper left',frameon=True)
plt.savefig('./Test.png',bbox_inches='tight')

