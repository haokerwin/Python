"""
@author haomj
"""

import csv
import time
import numpy as np
"""set some parameters                                        
"""
year = 2016
mtds = [31,29,31,30,31,30,31,31,30,31,30,31]
hour = list(range(24))
mnte = [0,15,30,45]
dtlt = sum(mtds[0:6])*24*4 
R = np.zeros((dtlt,7))
for i in range(dtlt): 
    for j in range(7):
            R[i,j] = np.nan 
"""v1-year, v2-month, v3-day, v4-hour, v5-minute, v6-at wind, v7-sim wind
"""
nn = 0
for ii in list(range(6)): 
    for jj in list(range(mtds[ii])): 
        for kk in list(range(24)): 
            R[nn:nn+4,0] = int(year)
            R[nn:nn+4,1] = int(ii+1) 
            R[nn:nn+4,2] = int(jj+1) 
            R[nn:nn+4,3] = int(kk) 
            R[nn,4] = int(0)
            R[nn+1,4] = int(15)
            R[nn+2,4] = int(30)
            R[nn+3,4] = int(45) 
            nn = nn+4
"""read and write the at data to volumn 5(6)
"""
mdir = '/home/haomj/python/practise/test_site/anemometer_tower/'
wnd1 = []
wnd2 = [] 
with open(mdir+'2901at.csv','r') as csvf:
    reader = csv.reader(csvf)
    for row in reader:
        wnd1.append(row)
wnd2 = wnd1[1:] 
cont = 0
for nn in list(range(len(wnd2))): 
    t = time.strptime(wnd2[nn][2], '%Y/%m/%d %H:%M')
    for ii in list(range(cont,dtlt,1)): 
        if t[1] == R[ii,1] and t[2] == R[ii,2] and t[3] == R[ii,3] and t[4] == R[ii,4]:
            R[ii,5] = wnd2[nn][3]
            cont = ii 
            break 
    del t
"""fetch data at a certain level
"""
ndir = '/home/haomj/python/practise/test_site/forecast_data/'
wnd3 = []
with open(ndir+'EEE2901-00fcst-20160101.txt') as txtf:
    data = txtf.readlines()
    for line in data:
        wnd3.append(line.split(','))
"""select all the 90m wind
"""
nwnd = 0
for ii in list(range(len(wnd3)-1)):
    if int(wnd3[ii][3]) == 90:
        nwnd = nwnd + 1
wnd4 = wnd3[0:nwnd]
nn = 0
for ii in list(range(len(wnd3)-1)):
    if int(wnd3[ii][3]) == 90:
        wnd4[nn] = wnd3[ii]
        nn = nn + 1
"""read and write the model data to volumn 6(7)
"""
cont = 0
for nn in list(range(len(wnd4))): 
    t = time.strptime(wnd4[nn][2], '%Y-%m-%d %H:%M:%S')
    for ii in list(range(cont,dtlt,1)):
       if t[1] == R[ii,1] and t[2] == R[ii,2] and t[3] == R[ii,3] and t[4] == R[ii,4]:
            R[ii,6] = wnd4[nn][5]
            cont = ii
            break
    del t
"""save data as csv format file
"""
np.savetxt('./2901.csv', R, delimiter = ',')



