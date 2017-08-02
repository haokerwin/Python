"""
@author haomj
On 20170802
"""

import numpy as np
np.random.seed(1337)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import r2_score
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, ActivityRegularization, Dropout
from keras.callbacks import EarlyStopping
from keras.constraints import maxnorm
import h5py
from jdcal import gcal2jd
import time
import matplotlib.pyplot as plt 

"""API
"""
ratio = 0.8 #ratio of the training data
prop  = [0.,0.2,0.8,1.] #divide data into different sets based on time sequency

#model
numad = 8 #an arbitrary number to specify the process
dropout_rate = 0.6 #the regularization term
weight_constraint = 2
init_mode = 'uniform' #the initial weight distribution
activation = 'linear' #'relu' the activation function
neurons = 64 #the number of neurons in the input layers
epochs = 50 #the number of iteactions
batch_size = 180 #how many samples to be handle once in a time
rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06) #the loss function options
optimizer = 'rmsprop'
#dagrad = optimizers.Adagrad(lr=0.2, epsilon=1e-06) #lr=0.001
#ptimizer = 'adagrad'
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
model = Sequential() 
model.add(Dense(int(neurons),input_dim=nf,kernel_initializer=init_mode,kernel_constraint=maxnorm(weight_constraint)))
model.add(Activation(activation))
model.add(Dense(int(neurons/2),input_dim=int(neurons),kernel_initializer=init_mode,kernel_constraint=maxnorm(weight_constraint)))
model.add(Activation(activation))
model.add(Dense(int(neurons/4),input_dim=int(neurons/2),kernel_initializer=init_mode,kernel_constraint=maxnorm(weight_constraint)))
model.add(Activation(activation))
model.add(Dropout(dropout_rate))
model.add(Dense(1,input_dim=int(neurons/4),kernel_initializer=init_mode,kernel_constraint=maxnorm(weight_constraint))) 
model.compile(loss='mse', optimizer=optimizer) 
early_stopping = EarlyStopping(monitor='loss', patience=10) 
t0 = time.time()
hist = model.fit(X_train_std, y_train_std, epochs = epochs, batch_size = batch_size,callbacks=[early_stopping]) 
tf = time.time() - t0
model.save_weights(repr(numad)+'modelweight.h5') 
y_train_std_pred = model.predict(X_train_std)
y_test_std_pred = model.predict(X_test_std)
y_train_pred = scy.inverse_transform(y_train_std_pred)
y_test_pred  = scy.inverse_transform(y_test_std_pred)
data[u'pre_wind'] = np.array(list(y_train_pred) + list(y_test_pred))
"""save and plot
"""
np.savetxt('./'+repr(numad)+'data.csv', data, delimiter = ',')
np.savetxt('./'+repr(numad)+'hist.csv', np.array(hist.history['loss']),delimiter = ',')
print('The time used to fit the model is '+repr(tf))

