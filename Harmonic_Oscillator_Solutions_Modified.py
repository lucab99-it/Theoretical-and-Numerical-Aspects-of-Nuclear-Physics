import numpy as np
from scipy import optimize as opt
import matplotlib.pyplot as plt

a=0.5 #lattice spacing
N = 20 #total lattice sites
T = a*N #total time
Npoints = 100
t = np.linspace(0, T, N) #time axis
w0=np.linspace(0.5,4,Npoints)

#The following function defines the matrix associated with the discretization of the
#differential operator of the second derivative. The discretization is done based on the
#central difference formula at first order with transformed anharmonic potential.
# It takes as input the oscillator frequency w0 and returns a NxN array M
def SystemMatrixTransformed(w0):
    M = np.zeros((N, N))
    for j in range(N):
        jm = (j-1)
        jp = (j+1)
        if jm >= 0:
            M[jm][j] = 1
        M[j][j] = -(2 + (a*w0)**2*(1+(a*w0)**2/12))
        if jp < N:
            M[jp][j] = 1
    return M

#The following function solves the linear system Mx = h where M is the matrix of the transformed discretization of the eqs,
# and h is a length-N array, h = [-x0,0...,0,xN] where x0 and xN are passed as inputs, together with the oscillator frequency w0.
#It returns an array x solution to the aforementioned system.
def SolveTransformed(w0,x0, xN):
    h = np.zeros(N)
    h[0] = -x0
    h[-1] = xN
    M = SystemMatrixTransformed(w0)
    x = np.linalg.solve(M,h)
    return x


#Parametric custom-made exponential function for fit purposes
def exp(x,A,w):
    return A*np.exp(-w*x)

#The following function plots the solution in a graph containing both the numerical solution w_sol
#and its expected behaviour w_exp. Alongside these, it takes as input the array of known frequencies w0
def PlotSolution(w0, w_exp, w_sol):
    fig = plt.figure()
    l1=plt.plot(w0,w_exp,label = r'$\omega_0\sqrt{1-\frac{(a\omega_0)^4}{360}}$', linestyle = 'dotted')
    l2 = plt.plot(w0,w_sol, label = 'Numerical solution', linestyle='none', marker = '.')
    plt.xlabel(r'$\omega_0$')
    plt.ylabel('$\omega$')
    plt.legend(loc='lower right')
    plt.show()

#The following function solves the equations of motion of a Harmonic Oscillator
#with frequency varying as in the array w0, passed as input, with the transformed anharmonic potential.
# It also has an optional argument Npoints which lets the user select the number of
#points in the final graph.
#It finally plots the numerical solution with the expected behaviour as a function of w0.
def HarmonicOscillatorSolution(w0, Npoints=100):
    x_sol = np.zeros((Npoints,N))
    w_sol = np.zeros(Npoints)
    w_exp = np.zeros(Npoints)
    x0 = np.exp(0)
    for k in range(Npoints):
        xN = np.exp(-w0[k]*T)
        x_sol[k] = SolveTransformed(w0[k], x0, xN)
        w_exp[k] = w0[k]*np.sqrt(1-((a*w0[k])**4)/360)
#Fitting the solution with the exponential function to retrieve w_sol
        optpar, cov = opt.curve_fit(exp, t, x_sol[k], p0 = (0.1, w0[k]))
        w_sol[k] = optpar[1]
    PlotSolution(w0, w_exp, w_sol)

#Main Body
HarmonicOscillatorSolution(w0)
