import numpy as np
import matplotlib.pyplot as plt
from Harmonic_MC import bin, MCAverage, DeltaE,update,getG

a = 0.5 #lattice spacing
N = 20 #lattice sites
T = a*N #total time
t= np.linspace(0,T,N) #time axis
epsilon = 1.4 #range for updating x
N_cor = 1 #Number of discarded updates before any evaluation of G, we run a highly-correlated one to see if binning helps
N_cf = 10000 #total number of configurations
binsize = 20 #we choose the bin size so that we have about 50 bins

#The following function computes the MC average of G, once it has been binned. It takes as input a 2D array G,
#the number of sites N, the update range epsilon, the number of correlated configurations N_cor
#and the total number of configurations N_cf. It returns two arrays, avg_G contains the MC averaged values
#of G_n and G_errors contains the corresponding statistical errors computed in the binned array as G_err = (<G_b^2>-<G_b>^2)/N_bin.

def MCAverageBin (G,N,epsilon,N_cor, N_cf, binsize,a=0.5):
    N_bin = int(N_cf/binsize)
    x=np.zeros(N)
    avg_G =np.zeros(N)
    G_errors = np.zeros(N)
    G_bin = np.zeros((N_bin,N))
    for j in range(0,5*N_cor): #lattice thermalization
        update(x,epsilon,a=a)
    for alpha in range(0,N_cf):
        for j in range(0,N_cor):
            update(x,epsilon,a=a)
        for n in range (0,N):
            G[alpha][n]=getG(x,n) ##for each configuration x_aplha computes G_n
    for n in range(N):
        G_bin[:,n]= bin(G[:,n], binsize) #binning G wrt index alpha
        for beta in range (len(G_bin)):
            avg_G[n]+=G_bin[beta][n]  #compute the binned average
            G_errors[n] += (G_bin[beta][n])**2/N_bin #compute the binned error
        avg_G[n] = avg_G[n]/N_bin
        G_errors[n] = np.sqrt((G_errors[n]-avg_G[n]**2)/N_bin)
    return avg_G, G_errors

#We compare non-binned and binned averages
gbin = np.zeros((N_cf,N))
g=np.zeros((N_cf,N))
Gbin, Gbin_errors = MCAverageBin(gbin,N,epsilon,N_cor,N_cf,binsize)
G, G_errors = MCAverage(g,N,epsilon,N_cor,N_cf)

dE_bin, deltaE_bin  = DeltaE(Gbin,Gbin_errors) #compute dE and statistical error in binned case
dE, deltaE =DeltaE(G,G_errors) #compute dE and statistical error in non-binned case


fig = plt.figure()
l1 = plt.errorbar(t,dE_bin,yerr= deltaE_bin, ecolor = 'r', capsize=5, label=r'$\Delta E (t)$ binned', marker = '.', linestyle = 'none') #plot the binned numerical propagator
l2 = plt.plot(t,np.ones(N), label = r'$\Delta E(\infty)$', linestyle = 'dotted') #asymptotic value of dE
l3 = plt.errorbar(t,dE,yerr= deltaE, ecolor = 'b', capsize=5, label=r'$\Delta E (t)$ non-binned', marker = '.', linestyle = 'none') #plot the non-binned numerical propagator
plt.axis([-0.1,3,0,2])
plt.xlabel('t')
plt.ylabel('$\Delta E$')
plt.legend(loc='upper right')
plt.show()
