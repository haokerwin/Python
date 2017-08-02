"""
@@@@
"""


import numpy as np
np.random.seed(1337)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Embedding, GlobalMaxPooling1D
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import EarlyStopping
from jdcal import gcal2jd
import time
import h5py
import matplotlib.pyplot as plt

# set parameters:
max_features = 5000 #mask words with lower frequency of occurence
maxlen = 8 #the number of features / variables
batch_size = 6
embedding_dims = 50#20000#50
filters = 250 #objectively the number of feature maps
kernel_size = 8 #the length of window
hidden_dims = 250 #the number of neurons in the full join layer
epochs = 2
ratio = 0.5 #ratio of the training data
prop  = [0.,0.2,0.8,1.] #divide data into different sets based on time sequency
"""add data
"""
mdir = '/home/haomj/python/practise/'
df = pd.read_table(mdir+'70m.txt',header=0,delim_whitespace=True)
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
np.savetxt('./used_data.csv', df, delimiter = ',')
dh   = df.dropna(axis=0,how='any')
data = dh.loc[range(int(len(dh)*prop[0]),int(len(dh)*prop[2]))].copy() 
feature = ['ori_fcst','dir_fcst','pre_fcst','t2m_fcst','rh2m_fcst','MJD','hour','min'] 
nf = len(feature)
label = ['obs_wind'] 
"""set training set and test set
"""
l_total = len(data) 
l_train = int(ratio*l_total)
l_test  = l_total - l_train
data_train = data.loc[range(0,l_train)].copy()
data_test  = data.loc[range(l_train,l_total)].copy()
X_train = data_train[feature].as_matrix() 
y_train = data_train[label].as_matrix() 
X_test  = data_test[feature].as_matrix()
y_test  = data_test[label].as_matrix()
scx = MinMaxScaler()
scx.fit(X_train)
X_train_std = scx.transform(X_train)
X_test_std  = scx.transform(X_test)
scy = MinMaxScaler()
scy.fit(y_train)
y_train_std = scy.transform(y_train)


model = Sequential()
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))
model.add(Convolution1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='linear',
                 strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('linear'))
model.add(Dense(1))
model.add(Activation('linear'))

msprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
model.compile(loss='mse', optimizer='rmsprop')
hist = model.fit(X_train_std, y_train_std,batch_size=batch_size,epochs=epochs)
model.save_weights('modelweight.h5') 
y_train_std_pred = model.predict(X_train_std)
y_test_std_pred  = model.predict(X_test_std)
"""save the y label
"""
R = np.zeros((len(y_train_std),2))
R[:,0] = y_train_std[:,0]
R[:,1] = y_train_std_pred[:,0]
np.savetxt('./test.csv', R, delimiter = ',')

y_train_std_pred = y_train_std_pred.reshape(l_train)
y_test_std_pred  = y_test_std_pred.reshape(l_test)
y_train_pred = scy.inverse_transform(y_train_std_pred)
y_test_pred  = scy.inverse_transform(y_test_std_pred)
data[u'pre_wind'] = np.array(list(y_train_pred) + list(y_test_pred))
"""save and plot
"""
np.savetxt('./data1.csv', data, delimiter = ',')
np.savetxt('./hist.csv', np.array(hist.history['loss']),delimiter = ',')

