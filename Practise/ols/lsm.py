"""
@@@@
"""

import numpy as np
np.random.seed(1337)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from jdcal import gcal2jd
import time

ratio = 0.8 #ratio of the training data
prop  = [0.,0.2,0.8,1.] #divide data into different sets based on time sequency
mdir = '/home/haomj/python/practise/'
df = pd.read_table(mdir+'70m.txt',header=0,delim_whitespace=True)
"""add the Julian day in case of annual and daily cycle
"""
R = np.zeros((len(df),3))
for ii in list(range(len(df))):
    t  = time.strptime(repr(df.at[ii,'ftime']), '%Y%m%d%H%M')
    jd = gcal2jd(t[0],t[1],t[2])
    R[ii][0] = jd[1] 
    R[ii][1] = t[3]  
    R[ii][2] = t[4]  
    del t
    del jd
df[u'MJD']  = np.array(list(R[:,0]))
df[u'hour'] = np.array(list(R[:,1]))
df[u'min']  = np.array(list(R[:,2]))
dh   = df.dropna(axis=0,how='any')
data = dh.loc[range(int(len(dh)*prop[0]),int(len(dh)*prop[2]))].copy() 
feature = ['ori_fcst','dir_fcst','pre_fcst','t2m_fcst','rh2m_fcst','MJD','hour','min'] 
nf = len(feature)
label = ['obs_wind']
"""set training set and test set
"""
l_total = len(data) 
l_train = int(ratio*l_total)
data_train = data.loc[range(0,l_train)].copy()
data_test  = data.loc[range(l_train,l_total)].copy()
X_train = data_train[feature].as_matrix() 
y_train = data_train[label].as_matrix() 
X_test  = data_test[feature].as_matrix()
y_test  = data_test[feature].as_matrix()
scx = MinMaxScaler()
scx.fit(X_train)
X_train_std = scx.transform(X_train)
X_test_std  = scx.transform(X_test)
scy = MinMaxScaler()
scy.fit(y_train)
y_train_std = scy.transform(y_train)
"""create a linear model
"""
slr = LinearRegression()
t0 = time.time()
slr.fit(X_train_std,y_train_std)
tf = time.time() - t0
y_train_std_pred = slr.predict(X_train_std)
y_test_std_pred = slr.predict(X_test_std)
y_train_pred = scy.inverse_transform(y_train_std_pred)
y_test_pred  = scy.inverse_transform(y_test_std_pred)
data[u'pre_wind'] = np.array(list(y_train_pred) + list(y_test_pred))
"""save and plot
"""
np.savetxt('./data1.csv', data, delimiter = ',')
print('The time used to fit the model is '+repr(tf))

