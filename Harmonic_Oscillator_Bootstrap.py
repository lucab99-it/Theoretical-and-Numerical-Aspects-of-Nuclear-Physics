import numpy as np
from Harmonic_MC import getG, update,bootstrap
from matplotlib import pyplot as plt

a = 0.5 #lattice spacing
N = 20 #lattice sites
T = a*N #total time
t= np.linspace(0,T,N) #time axis
epsilon = 1.4 #range for updating x
N_cor = 20 #Number of discarded updates before any evaluation of G, we run a highly-correlated one to see if binning helps
N_cf = 1000 #total number of configurations
N_boot = 100 #number of bootstrap copies of G

#The following function performs the bootstrap MC average of a function G. It takes as input a 2D array G,
#the number of sites N, the update range epsilon, the number of correlated configurations N_cor
#the total number of configurations N_cf and the number of copies N_boot of G. it returns a 2D array avg_G
#containing N_boot arrays of MC-averaged G_n.
def MCAverageBoot(G,N,N_boot,N_cf, N_cor, epsilon):
    x = np.zeros(N)
    G=np.zeros((N_cf,N))
    avg_G=np.zeros((N,N_boot))
    G_bootstrap=np.zeros((N_boot,N_cf,N))

    for _ in range(0,5*N_cor): # lattice thermalization
        update(x,epsilon)
    for alpha in range(0,N_cf):
        for _ in range(0,N_cor):
            update(x,epsilon)
        for n in range(0,N):
            G[alpha][n] = getG(x,n) #for each configuration x_aplha computes G_n

    G_bootstrap[0]=G #Keep original G as first bootstrap copy

    for k in range(1,N_boot):
        G_bootstrap[k]=bootstrap(G,N) #N_boot bootstrap copies of G

    for k in range(0,N_boot):
        for n in range(0,N):
            for alpha in range(0,N_cf):
                avg_G[n][k] +=  G_bootstrap[k][alpha][n]
            avg_G[n][k] = avg_G[n][k]/N_cf  #MC average on configurations for each bootstrap copy

    return avg_G

#The following function computes the energy gap and statistical errors for a bootstrap collection of configurations G.
#It takes as inputs a 2D array G and returns the array of energy gap dE_n averaged over the N_boot configurations in G
#and the corresponding statistical errors deltaE_n.
def DeltaE_bootstrap(G):
    dE = np.zeros((N,N_boot))
    dE_avg = np.zeros(N)
    dE_squared = np.zeros(N)
    deltaE = np.zeros(N)
    for k in range(N_boot):
        for n in range(N-1):
            dE[n][k] = np.log(np.abs(G[n][k]/G[n+1][k]))/a #compute dE_n for each bootstrap copy
    for n in range(N):
        for k in range(N_boot):
            dE_avg[n]+=dE[n][k] # compute energy gap as average of a*dE_n =ln(G_n/G_(n+1))
            dE_squared[n]+=(dE[n][k])**2 #sum of squares of dE
        dE_avg[n]=dE_avg[n]/N_boot #bootstrap average of dE
        dE_squared[n] = dE_squared[n]/N_boot #bootstrap average of dE^2
        deltaE[n]= ((dE_squared[n]-(dE_avg[n])**2))**(1/2) #statistical errors computed from bootstrap distribution
    return dE_avg, deltaE

#Main body of the program
g = np.zeros((N_cf,N))
G =MCAverageBoot(g,N,N_boot,N_cf, N_cor, epsilon) # MC average of bootstrapped G

#Compute the energy gap at time tn for each n and the corresponding error
dE_avg, deltaE = DeltaE_bootstrap(G)

fig = plt.figure()
l1 = plt.errorbar(t,dE_avg,yerr= deltaE, ecolor = 'r', capsize=5, label=r'$\Delta E (t)$', marker = '.', linestyle = 'none') #plot the numerical values
l2 = plt.plot(t,np.ones(N), label = r'$\Delta E(\infty)$', linestyle = 'dotted') #asymptotic value of dE
plt.axis([-0.1,3,0,2])
plt.xlabel('t')
plt.ylabel('$\Delta E$')
plt.legend(loc='upper right')
plt.show()
