

import vegas
import numpy as np
import matplotlib.pyplot as plt


a= 0.5 #Lattice spacing
N = 8 #lattice sites
T = a*N #Total time
N_points = 11 #number of points between 0 and 2 in which we compute the propagator
x0=np.linspace(0.,2.,N_points) #boundary conditions
r=5. #integration range


#Harmonic Potential
def V(x):
    return (x**2)/2


#Euclidean action on a 1D Lattice for potential V with boundary conditions
#x_0 = x_N = x0. When we will use this function, the array we will pass to S_lat
#will only represent the sites x_1,..,x_(N-1) since the boundaries will not be
#integrated on.

def S_lat(xarr,x0):
    S_LAT = (1/(2*a))*(xarr[0]-x0)**2 + a*V(x0)
    S_LAT+= (1/(2*a))*(x0-xarr[-1])**2 +a*V(xarr[-1])
    for j in range(len(xarr)-1):
        S_LAT += (1/(2*a))*(xarr[j+1]-xarr[j])**2 +a*V(xarr[j])
    return S_LAT


#This function takes as input the lattice spacing a, the number of lattice sites N,
#the potential function V, the initial condition array x0 such that for each k: x_0 = x_N = x0[k],
# and the integration range r. It returns an array with the values of the propagator
#computed with the boundary conditions x0.

def propagator_1d (a,N,V,x0,r):
    A=(1/(2*a*np.pi))**(N/2) #normalization constant
    propagator=np.zeros(len(x0))

    for k in range(len(x0)): #repeat for each value of the x we are interested in
        def weight(xarr): #exponential weight
            S_LAT = S_lat(xarr,x0[k])
            return np.exp(-S_LAT)

#Setting up the numerical integration with vegas. We are integrating on (N-1)
#Lattice sites since of the N+1 total sites 2 are fixed by boudary coditions.
#The integration domain is symmetrical and set by r.

        integ=vegas.Integrator((N-1)*[[-r, r]]) #set up integrator
        result = integ(weight, nitn=10, neval=100000) #perform the path integral as N-1-dim integral of the exponential weight
        propagator[k]=A*result.mean #final value of the numerical propagator

    return propagator

#Testing the numerical calculation on our harmonic potential.
#First computing the exact value of the propagator, then comparing graphically
#the numerical propagator to the exact one.

def tr_propagator(x): #exact value of the propagator at boundary condition x
    return np.exp(-T/2)/np.sqrt(np.pi) * np.exp(-x**2)

x = np.linspace(0.,2., 100)
true_propagator = [tr_propagator(y) for y in x]

num_propagator = propagator_1d (a,N,V,x0,r)

#Ploting the results in the same picture
fig = plt.figure()
l1 = plt.plot(x0,num_propagator, label='Numerical Propagator', marker='o')
l2 = plt.plot(x,true_propagator, label='Exact Propagator')
fig.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('Propagator')
plt.show()
