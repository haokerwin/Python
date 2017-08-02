"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
n_groups = len(init_mode)

mdir = '../'
df = pd.read_csv(mdir+'result.csv',header=None)
print(df.head)
sor = [l*(-1.) for l in df.ix[:,[0]].values]
std = df.ix[:,[1]].values
ode = list(range(n_groups))

index = np.arange(n_groups)
bar_width = 0.85

opacity = 0.8
error_config = {'ecolor': '0.3'}

sns.set_style ('darkgrid')
fig = plt.figure()
ax  = fig.add_subplot(1,1,1)
rects1 = plt.bar(index, sor, bar_width,
                alpha=opacity,
                color='teal',
                yerr=std,
                error_kw=error_config)
plt.xlabel('Initial Weight')
plt.ylabel('Scores')
#plt.title('Scores by group and gender')
plt.xticks(index, tuple(init_mode), rotation=17)
plt.legend()
fig.set_size_inches(7.5, 7.5)
#plt.tight_layout()
#plt.show()
plt.savefig('./fig.png',bbox_inches='tight')
