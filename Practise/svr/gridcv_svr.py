"""
@@@@
"""

import numpy as np
np.random.seed(1337)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from jdcal import gcal2jd
import time

ratio = 0.8 
prop  = [0.,0.2,0.8,1.] 
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
"""create a svr model
"""
svr = SVR(kernel='rbf',gamma=0.1,C=1.0,epsilon=0.2)
C = [1e0,1e1,1e2,1e3]
gamma = np.logspace(-2,2,5)
param_grid = dict(C=C,gamma=gamma)
grid = GridSearchCV(estimator=svr,cv=5,param_grid=param_grid,n_jobs=1,scoring='neg_mean_squared_error')
t0 = time.time()
grid_result = grid.fit(X_train_std, y_train_std)
tf = time.time() - t0
print('The time used to fit the model is '+repr(tf))
"""summarize results
"""
R  = np.zeros((len(C)*len(gamma),3))
nn = 0
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
    R[nn][0] = scores.mean()
    R[nn][1] = params['C']
    R[nn][2] = params['gamma']
    nn = nn + 1 
np.savetxt('./result.csv', R, delimiter = ',')

