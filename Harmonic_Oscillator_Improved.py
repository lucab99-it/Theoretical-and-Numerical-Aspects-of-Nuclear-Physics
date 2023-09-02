import numpy as np
import matplotlib.pyplot as plt
from Harmonic_MC import getG,DeltaE,V,bin

a=0.5 #lattice spacing
N = 20 #number of lattice sites
T = a*N #total time
t = np.linspace(0,T,N)#time axis


#Improved Euclidean action for numerical calculations involving x[j] for a m=1 particle. The second derivative is computed numerically
#using the second-order formula for the central difference method.

def S_imp(x,j):
    jp = (j+1)%N
    jp2 = (j+2)%N
    jm = (j-1)%N
    jm2 = (j-2)%N
    return a*V(x[j])-x[j]*(1/(2*a))*((-1/6)*(x[jp2]+x[jm2])+ (8/3)*(x[jp]+x[jm])-(5/2)*x[j])


#Improved update function
def update_imp(x,epsilon): #changes the path x through Metropolis algorithm
    for j in range(len(x)):
        x_old=x[j]
        S_old = S_imp(x,j)
        x[j]+= 2*epsilon*np.random.rand()-epsilon #modifies the position of x(tj)
        dS = S_imp(x,j)-S_old #variation in the action
        if dS>0 and np.exp(-dS)<np.random.rand(): #updates x(tj) only if dS<0 (more probable path) or if e^(-dS)<uniform(0,1)
            x[j] = x_old

#MCAverage function with the improved update function
def MCAverage_imp(G,N,epsilon,N_cor, N_cf,binsize):
    N_bin = int(N_cf/binsize)
    x=np.zeros(N)
    avg_G =np.zeros(N)
    G_errors = np.zeros(N)
    G_bin = np.zeros((N_bin,N))
    for j in range(0,5*N_cor):
        update_imp(x,epsilon)
    for alpha in range(0,N_cf):
        for j in range(0,N_cor):
            update_imp(x,epsilon)
        for n in range (0,N):
            G[alpha][n]=getG(x,n)
    for n in range(N):
        G_bin[:,n]= bin(G[:,n], binsize) #binning G wrt index alpha
        for beta in range (N_bin):
            avg_G[n]+=G_bin[beta][n]  #compute the binned average
            G_errors[n] += (G_bin[beta][n])**2/N_bin #compute the binned error
        avg_G[n] = avg_G[n]/N_bin
        G_errors[n] = np.sqrt((G_errors[n]-avg_G[n]**2)/N_bin)
    return avg_G, G_errors


epsilon = 1.4
N_cor =20
N_cf = 10000
N_bin = 100
binsize = int(N_cf/N_bin)
g = np.zeros((N_cf,N))
G, G_errors =MCAverage_imp(g,N,epsilon,N_cor,N_cf, binsize)

dE,deltaE = DeltaE(G,G_errors)

fig = plt.figure()
l1 = plt.errorbar(t,dE,yerr= deltaE, ecolor = 'r', capsize=5, label=r'$\Delta E (t)$', marker = '.', linestyle = 'none')
l2 = plt.plot(t,np.ones(N), label = r'$\Delta E(\infty)$', linestyle = 'dotted')
plt.axis([-0.2,3.2,0,2])
plt.xlabel('t')
plt.ylabel('$\Delta E$')
plt.legend(loc='upper right')
plt.show()
