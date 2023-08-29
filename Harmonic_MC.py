import numpy as np


#Harmonic potential
def V(x):
    return x**2/2

#Part of the total Euclidean action involving x[j]. It is useful to have so that we don't
#compute the full action each time we update a single site.

def S(x,j,a=0.5):
    jp=(j-1)%len(x) #next site
    jm=(j+1)%len(x) #previous site
    return a*V(x[j]) + x[j]*(x[j]-x[jp]-x[jm])/a

#The update function takes as input the site configuration array x, and the range
#epsilon in which we produce a random number. This function changes x into x+dx with
#dx in [-epsilon, epsilon] according to the Metropolis algorithm: if S[x+dx]<S[x] it means
#that the new configuration is more probable, otherwise we compare e^(-dS) with a random
#number q between 0 and 1, and update x only if e^(-dS)<q.

def update(x,epsilon,a=0.5):
    for j in range(len(x)):
        x_old=x[j]
        S_old = S(x,j,a=a)
        x[j]+= 2*epsilon*np.random.rand()-epsilon #modifies the position of x(tj)
        dS =S(x,j)-S_old #variation of the action
        if dS>0 and np.exp(-dS)<np.random.rand(): #updates x(tj) only if dS<0 (more probable path) or if e^(-dS)<uniform(0,1)
            x[j] = x_old

#This function computes G given a configuration x at a time t_n.
def getG (x,n):
    N = len(x)
    G=0
    for j in range (N):
        G+=(x[j])*(x[(j+n)%N])
    return G/N

#The following function computes the MC average of G. It takes as input a 2D array G,
#the number of sites N, the update range epsilon, the number of correlated configurations N_cor
#and the total number of configurations N_cf. It returns two arrays, avg_G contains the MC averaged values
#of G_n and G_errors contains the corresponding statistical errors computed as G_err = (<G^2>-<G>^2)/N_cf.

def MCAverage(G,N,epsilon,N_cor, N_cf,f=getG,a=0.5):
    x=np.zeros(N)
    avg_G =np.zeros(N)
    G_errors = np.zeros(N)
    for j in range(0,5*N_cor):
        update(x,epsilon,a=a) #thermalization of the lattice, configurations are less biased
    for alpha in range(0,N_cf):
        for j in range(0,N_cor):
            update(x,epsilon,a=a)
        for n in range (0,N):
            G[alpha][n]=f(x,n) #for each configuration x_aplha computes G_n
    for n in range(0,N):
        for alpha in range (0,N_cf):
            avg_G[n]+=G[alpha][n] #computes the average
            G_errors[n] += (G[alpha][n])**2/N_cf  #computes the error
        avg_G[n] = avg_G[n]/N_cf
        G_errors[n] = np.sqrt((G_errors[n]-avg_G[n]**2)/N_cf)
    return avg_G, G_errors

#The following function performs the binning of an array G in bins of size binsize. It returns
#an array of size len(G)/binsize containing the averaged value of the elements of G in each bin.

def bin (G, binsize): #function that does the binning of the array G
    G_binned = []
    for i in range(0,len(G),binsize):
        G_avg = 0
        for j in range (0, binsize):
            if i+j<len(G):
                G_avg += G[i+j]
        G_binned.append(G_avg/binsize)
    return G_binned

#The following function takes as input an array G and performs a "bootstrap copy" of G.
#It returns an array G_bootstrap which contains exactly len(G) elements which are randomly
#sampled from G.

def bootstrap(G):
    N_cf = len(G)
    G_bootstrap = [] # new array of configurations
    for i in range(0,N_cf):
        alpha = int(N_cf * np.random.rand()) # choose random configuration from G
        G_bootstrap.append(G[alpha])
    return G_bootstrap

#The following function computes the energy gap and statistical errors.
#It takes as input the array G of MC averaged values of G,
# array G_errors of statistical errors on G, the total number of configurations N_cf
#and the (optional) lattice spacing a. It returns three arrays: dE of energy gaps for each n,
# deltaE of statistical errors computed as propagation of error from G_errors
#and deltaE_graph of errorbars for a graph which are scaled up for display purposes only and are not reliable.

def DeltaE (G, G_errors,N_cf,a=0.5):
    N = len(G)
    dE = np.zeros(N) #energy gap
    deltaE = np.zeros(N) #errors on dE
    deltaE_graph = np.zeros(N) # erorbars for the graph
    for n in range (N):
        dE[n] =np.log(np.abs(G[n]/G[(n+1)%N]))/a # compute energy gap as a*dE_n =ln(G_n/G_(n+1))
        deltaE[n] = np.abs(G_errors[n]/G[n] - G_errors[(n+1)%N]/G[(n+1)%N])/a #compute error bars on dE
    return dE , deltaE
