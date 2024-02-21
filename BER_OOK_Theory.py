# Theoretical BER calculation of OOK DBMC transmission, using spherical transparent receiver with single sample and fixed threshold
#! based on Poisson approximation or Gaussian approximation

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.stats import norm
from scipy.io import savemat

#* parameters
# channel parameter
diffc = 1e-11           # diffusion coefficient in m^2/s
dist  = 3e-6            # distance between Tx and center of Rx
rad   = 1e-6;           # receiver radius in m
# transmission parameters
N     = 5e3             # number of release particles for bit-1
T     = 5e-1            # symbol duration in s
Ts    = 0.15            # sampling time in s
K     = 1e2             # number of bits
# detection threshold values
thre  = np.arange(0, N+1)
# numerical calculation parameters
L     = 12              # considered channel memory length
met   = 'Gaussian'      # method for BER calculation ['Gaussian' or 'Poisson' (default)]
fname = 'BER_Num_OOK'   # name of result file

#* observation probabilities
t = np.arange(Ts, K*T+Ts, T)
h = 4/3 * np.pi * np.power(rad, 3) / np.power(4*np.pi*diffc*t, 3/2) * np.exp(-np.power(dist, 2) / (4*diffc*t))

#* BER calculation
BER = np.zeros(thre.size)
factor = 1
for k in range(0, min(int(K), L+1)):
    print(k)
    
    if k == L:
        factor = K - L
    
    # number of states
    STATES = np.power(2, k+1)
    
    for state in range(0, STATES):
        # create bit sequence
        u = np.flip(np.array([int(bit) for bit in np.binary_repr(state, min(k+1, L+1))]))
        
        # get expected number of observations
        ymean = N * sum(u * h[0:u.size])
        
        # evaluate BER
        match met:
            case 'Gaussian':
                if ymean > 0:
                    yvar = N * sum(u * h[0:u.size] * (1 - h[0:u.size]))
                    yprob = norm.cdf(thre, ymean, np.sqrt(yvar))
                else:
                    yprob = np.ones(thre.size)
                
            case _:
                yprob = poisson.cdf(thre, ymean)
                
        if u[0]:
            BER += factor * (yprob) / STATES
        else:
            BER += factor * (1 - yprob) / STATES
        
BER /= K

#* plotting
fig, ax = plt.subplots()
ax.loglog(thre, BER)
ax.set_xlabel('Threshold')
ax.set_ylabel('BER')
ax.grid()
plt.show()

#* saving
savemat(fname+'.mat', {'threshold': thre, 'BER': BER})