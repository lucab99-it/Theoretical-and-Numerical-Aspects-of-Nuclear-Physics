import numpy as np
import matplotlib.pyplot as plt
from Harmonic_MC import MCAverage, DeltaE

a = 0.5 #lattice spacing
N = 20 #lattice sites
T = a*N #total time
t= np.linspace(0,T,N) #time axis
epsilon = 1.4#range for updating x
N_cor =20 #Number of discarded updates before any evaluation of G3
N_cf = 10000 #total number of configurations averaged

#Computes the function G3 which has cubic sources and wells
def getG3 (x,n):
    G=0
    for j in range (N):
        G+=(x[j])**3*(x[(j+n)%N])**3
    return G/N


g = np.zeros((N_cf,N))
G3, G3_errors= MCAverage(g,N,epsilon,N_cor,N_cf,f=getG3)# MC average and statistical error of G3

#Compute the energy gap at time tn for each n and the corresponding statistical error
dE3,deltaE3, deltaE3_graph = DeltaE(G3,G3_errors,N_cf) #energy gap


fig = plt.figure()
l1 = plt.errorbar(t,dE3, yerr= deltaE3_graph, ecolor = 'r', capsize=5, label=r'$\Delta E (t)$', marker = '.', linestyle = 'none') #numerical value
l2 = plt.plot(t,np.ones(N), label = r'$\Delta E(\infty)$', linestyle = 'dotted') #exact asymptotic value
plt.axis([-0.1,3,0,2])
plt.xlabel('t')
plt.ylabel('$\Delta E$')
plt.legend(loc='upper right')
plt.show()
