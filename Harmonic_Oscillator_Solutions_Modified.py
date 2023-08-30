import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

a=0.5 #lattice spacing
N = 20 #total lattice sites
w0=np.linspace(0,4,100) #known frequency of the oscillator
w_sol = np.zeros(100)
w_exp = np.zeros(100)
w_guess = 1

#Order-a^2 second derivative: central method numerical differentiation.
def D(x,j):
    jp = (j+1)%N
    jm = (j-1)%N
    return (x[jp]-2*x[j]+x[jm])/(a**2)

def f_mod(w,w0,j):
    x=[np.exp(-a*k*w) for k in range(N)]
    return D(x,j) - w0**2*(1+(a*w0)**2/12)*x[j]

for n in range(100):
    w_sol[n] = fsolve(f_mod,w_guess, args=(w0[n], 2))
    w_exp[n] = w0[n]*np.sqrt(1+(a*w0[n])**4/90)

fig = plt.figure()
l1=plt.plot(w0,w_exp,label = r'$\omega_0 \sqrt{1+\frac{(a\omega_0)^4}{90}}$', linestyle = 'dotted')
l2 = plt.plot(w0,w_sol, label = 'Numerical solution', linestyle='none', marker = '.')
plt.xlabel(r'$\omega_0$')
plt.ylabel('$\omega$')
plt.legend(loc='lower right')
plt.show()
