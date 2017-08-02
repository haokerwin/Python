"""
@author haomj
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def find_all_index(arr,item):
    return [i for i,a in enumerate(arr) if a==item]

mdir = '../'
df = pd.read_csv(mdir+'result.csv',header=None)
scor = df.ix[:,[0]].values 
btch = [int(i) for i in df.ix[:,[1]].values] 
epoc = [int(i) for i in df.ix[:,[2]].values] 
"""creat metgrid
"""
x = [50,100,150,200,250,300] 
y = [60,100,140,180,220] 
X, Y = np.meshgrid(x, y)
Z = np.zeros((len(y),len(x)))
for ii in list(range(len(y))):
    for jj in list(range(len(x))):
        yy = list(find_all_index(btch,int(y[ii])))
        xx = list(find_all_index(epoc,int(x[jj])))
        nn = [l for l in xx if l in yy]
        Z[ii][jj] = scor[nn[0]]
        del xx
        del yy
        del nn
"""plot
"""
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
levels = np.linspace(Z.min(),Z.max(),20)
CS = ax1.contourf(X, Y, Z, levels, cmap=plt.cm.Blues, origin='lower')
#plt.title('test')
plt.xlabel('Epoch')
plt.ylabel('Batch Size')
cbar = plt.colorbar(CS)
cbar.ax.set_ylabel('NMSE')
plt.savefig('./fig.png',bbox_inches='tight')

