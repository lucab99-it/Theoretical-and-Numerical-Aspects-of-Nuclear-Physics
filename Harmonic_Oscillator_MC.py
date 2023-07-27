import numpy as np
import matplotlib.pyplot as plt
from Harmonic_MC import  MCAverage, DeltaE

a = 0.5 #lattice spacing
N = 20 #lattice sites
T = a*N #total time
t= np.linspace(0,T,N) #time axis
epsilon = 1.4 #range for updating x
N_cor =20 #Number of discarded updates before any evaluation of G
N_cf = 10000 #total number of configurations averaged
g = np.zeros((N_cf,N))
G, G_errors =MCAverage(g,N,epsilon,N_cor,N_cf) # MC average and statistical error of G

#Compute the energy gap at time tn for each n and the corresponding error
dE, deltaE, deltaE_graph = DeltaE(G,G_errors,N_cf)

fig = plt.figure()
l1 = plt.errorbar(t,dE,yerr= deltaE_graph, ecolor = 'r', capsize=5, label=r'$\Delta E (t)$', marker = '.', linestyle = 'none') #plot the numerical values
l2 = plt.plot(t,np.ones(N), label = r'$\Delta E(\infty)$', linestyle = 'dotted') #asymptotic value of dE
plt.axis([-0.1,3,0,2])
plt.xlabel('t')
plt.ylabel('$\Delta E$')
plt.legend(loc='upper right')
plt.show()
