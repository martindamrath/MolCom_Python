# PBS BER calculation of OOK DBMC transmission, using spherical transparent receiver with single sample and fixed threshold

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

#* parameters
# transmission parameters
N     = 5e3             # number of release particles for bit-1
# detection threshold values
thre  = np.arange(0, N+1)
# filename
fname = 'PBS_OOK_N5000'

#* load PBS results
df = pd.read_csv(fname+'.csv', header=None)
# grouping by first column label
grouped = df.groupby(df[0])
# extract data into matrix [run x bits]
bits = []
y = []
for name, group in grouped:
    if name=='b':
        bits.append(group.iloc[:, 1:-1].to_numpy())
    elif name=='y':
        y.append(group.iloc[:, 1:-1].to_numpy())
bits = np.vstack(bits)
y = np.vstack(y)


#* BER calculation
BER = np.zeros(thre.size)
R, K = bits.shape
for r in range(R):
    if r%10000 == 0:
        print(r)
    bit = bits[r,:]
    rec = y[r,:]
    ber = (rec[:, np.newaxis] > thre)
    BER += np.sum((ber.T != bit), axis=1)

BER /= K * R

#* plotting
fig, ax = plt.subplots()
ax.loglog(thre, BER)
ax.set_xlabel('Threshold')
ax.set_ylabel('BER')
ax.grid()
plt.show()    

#* saving
savemat(fname+'.mat', {'threshold': thre, 'BER': BER})