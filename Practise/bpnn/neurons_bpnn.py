"""
@@@@
"""

import numpy as np
np.random.seed(1337)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import r2_score
from keras import optimizers
from keras.models import Sequential
from keras.layers.core import Dense, Activation, ActivityRegularization, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.constraints import maxnorm
from jdcal import gcal2jd
import time

"""API
"""
ratio = 0.8 #ratio of the training data
prop  = [0.,0.2,0.8,1.] #divide data into different sets based on time sequency
rg1 = 0.000 #L1 regulization
rg2 = 0.0   #L2 regulization
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
"""create a BP model
"""
def create_model(neurons=1):
    model = Sequential() 
    model.add(Dense(int(neurons),input_dim=nf,kernel_initializer='uniform',W_constraint=maxnorm(4)))
    model.add(Activation('relu'))
    model.add(Dense(int(neurons/2),input_dim=int(neurons),kernel_initializer='uniform',W_constraint=maxnorm(4)))
    model.add(Activation('relu'))
    model.add(Dense(int(neurons/4),input_dim=int(neurons/2),kernel_initializer='uniform',W_constraint=maxnorm(4)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1,input_dim=int(neurons/4),kernel_initializer='uniform',W_constraint=maxnorm(4))) 
    rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    model.compile(loss='mse',optimizer='rmsprop') 
    return model
model = KerasRegressor(build_fn=create_model,epochs=100,batch_size=6,verbose=0)
neurons = [4,16,28,40,52,64,76]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1,scoring='neg_mean_squared_error') 
t0 = time.time()
grid_result = grid.fit(X_train_std, y_train_std)
tf = time.time() - t0
"""summarize results
"""
R  = np.zeros((len(neurons),3))
nn = 0
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
    R[nn][0] = scores.mean()
    R[nn][1] = scores.std()
    R[nn][2] = params['neurons']
    nn = nn + 1
np.savetxt('./result.csv', R, delimiter = ',')

