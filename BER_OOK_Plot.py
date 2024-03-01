# BER Result PBS and theory plot

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

#* result file names
fname_PBS = 'PBS_OOK_N5000.mat'
fname_THE = 'THE_OOK_N5000.mat'

#* load files
data = loadmat(fname_PBS)
threshold = data['threshold'].flatten()
BER_PBS = data['BER'].flatten()
data = loadmat(fname_THE)
BER_THE = data['BER'].flatten()

#* plotting
fig, ax = plt.subplots()
ax.loglog(threshold, BER_PBS, BER_THE)
ax.set_xlabel('Threshold')
ax.set_ylabel('BER')
ax.grid()
ax.legend(['PBS', 'NUM'])
plt.show()    