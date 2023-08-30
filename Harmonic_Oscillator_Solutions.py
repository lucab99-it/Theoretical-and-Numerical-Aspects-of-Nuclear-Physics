import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

a=0.5 #lattice spacing
N = 20 #total lattice sites
w0=np.linspace(0,4,100) #known frequency of the oscillator
w_sol = np.zeros(100)
w_exp = np.zeros(100)
w_guess = w0.copy()

#Order-a^2 second derivative: central method numerical differentiation.
def D(x,j):
    jp = (j+1)%N
    jm = (j-1)%N
    return (x[jp]-2*x[j]+x[jm])/(a**2)

#Order a^4 second derivative. In the article it is written as
# D x_j - a^2/12 (D)^2 x_j with D being defined as above.
def D2(x,j):
    Dx = [D(x,k) for k in range(N)]
    return D(x,j)-a**2/12*D(Dx,j)

#Function to find the zero of,  defined using D as second derivative
def f_D(w, w0, j):
    x=[np.exp(-a*k*w) for k in range(N)]
    return D(x,j)-w0**2*x[j]

#Function to find the root of, defined using D2 as a second derivative
def f_D2 (w, w0, j):
    x=[np.exp(-a*k*w) for k in range(N)]
    return D2(x,j)-w0**2*x[j]

#Solves the equation to find w varying w0
for n in range(100):
    w_sol[n] = fsolve(f_D,w_guess[n], args=(w0[n],2)) #solves the equation f=0, with known frequency w0[n] for j=2
    w_exp[n] = w0[n]*np.sqrt(1-(a*w0[n])**2/12)

fig = plt.figure()
l1=plt.plot(w0,w_exp,label = r'$\omega_0\sqrt{1-\frac{(a\omega_0)^2}{12}}$', linestyle = 'dotted')
l2 = plt.plot(w0,w_sol, label = 'Numerical solution', linestyle='none', marker = '.')
plt.xlabel(r'$\omega_0$')
plt.ylabel('$\omega$')
plt.legend(loc='lower right')
plt.show()
