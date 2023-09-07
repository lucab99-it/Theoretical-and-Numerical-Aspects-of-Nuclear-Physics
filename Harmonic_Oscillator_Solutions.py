import numpy as np
from scipy import optimize as opt
import matplotlib.pyplot as plt

a=0.5 #lattice spacing
N = 20 #total lattice sites
T = a*N #total time
Npoints = 100
t = np.linspace(0, T, N) #time axis
improved = True
w0=np.linspace(0.2,4,Npoints) #known frequency of the oscillator

#The following function defines the matrix associated with the discretization of the
#differential operator of the equations of motion. The discretization is done based on the
#central difference formula of the second derivative at first order.
#It takes as input the oscillator frequency w0 and returns a NxN array M
def SystemMatrix(w0):
    M = np.zeros((N,N))
    for i in range(N):
        im = (i-1)
        ip = (i+1)
        if i> 0:
            M[im][i] = 1
        M[i][i] = -(2 + (a* w0)**2)
        if i < (N-1):
            M[ip][i] = 1
    return M


#The following function defines the matrix associated with the discretization of the
#differential operator of the equations of motion. The discretization is done based on the
#improved formula for the second derivative (second order formula).
#It takes as input the oscillator frequency w0 and returns a NxN array M
def SystemMatrixImproved(w0):
    M = np.zeros((N,N))
    for j in range(N):
        jm = (j-1)
        jp = (j+1)
        jm2 = (j-2)
        jp2 = (j+2)
        if jm2 >= 0:
            M[jm2][j] = -1/12
        if jm >= 0:
            M[jm][j] = 4/3
        M[j][j] = -(5/2 + (a*w0)**2)
        if jp < N:
            M[jp][j] = 4/3
        if jp2 < N:
            M[jp2][j] = -1/12
    return M

#The following function solves the linear system Mx = h where M is the matrix of the unimproved second derivative,
# and h is a length-N array, h = [-x0,0...,0,xN] where x0 and xN are passed as inputs, together with the oscillator frequency w0.
#It returns an array x solution to the aforementioned system.
def Solve(w0,x0, xN):
    h = np.zeros(N)
    h[0] = -x0
    h[-1] = xN
    M = SystemMatrix(w0)
    x = np.linalg.solve(M,h)
    return x

#The following function solves the linear system Mx = h where M is the matrix of the improved second derivative and
# h is a length-N array, h = [(1/12 x1-4/3 x0),(1/12)x0,0...,0,1/12 xN,(1/12 xNm1-4/3 xN)]
# where x0,x1,xNm1 and xN are passed as inputs, together with the oscillator frequency w0.
#It returns an array x solution to the aforementioned system.
def SolveImproved(w0,x0,x1,xNm1,xN):
    h = np.zeros(N)
    h[0]=(1/12)*x1-(4/3)*x0
    h[1] = x0*(1/12)
    h[N-2] = xN*(1/12)
    h[N-1] = (1/12)*xNm1-(4/3)*xN
    M = SystemMatrixImproved(w0)
    x = np.linalg.solve(M,h)
    return x

#Parametric custom-made exponential function for fit purposes
def exp(x,A,w):
    return A*np.exp(-w*x)

#The following function plots the solution in a graph containing both the numerical solution w_sol
#and its expected behaviour w_exp. Alongside these, it takes as input a boolean switch improved, which selects
#whether to graph an improved or uninproved expectation for w, and the array of frequencies w0
def PlotSolution(improved, w0, w_exp, w_sol):
    fig = plt.figure()
    if improved:
        l1=plt.plot(w0,w_exp,label = r'$\omega_0\sqrt{1+\frac{(a\omega_0)^4}{90}}$', linestyle = 'dotted')
    else:
        l1=plt.plot(w0,w_exp,label = r'$\omega_0\sqrt{1-\frac{(a\omega_0)^2}{12}}$', linestyle = 'dotted')

    l2 = plt.plot(w0,w_sol, label = 'Numerical solution', linestyle='none', marker = '.')
    plt.xlabel(r'$\omega_0$')
    plt.ylabel('$\omega$')
    plt.legend(loc='lower right')
    plt.show()

#The following function solves the equations of motion of a Harmonic Oscillator
#with frequency varying as in the array w0 passed as input. Alongside this, it takes as inputs
#a boolean switch improved, and an optional integer argument Npoints which lets the user select the number of
#points in the final graph.
#Finally, it plots the numerical solution with the expected behaviour as a function of w0.
def HarmonicOscillatorSolution(w0,improved, Npoints=100):
    x_sol = np.zeros((Npoints,N))
    w_sol = np.zeros(Npoints)
    w_exp = np.zeros(Npoints)
    x0 = np.exp(0) #x0 = e^(-w*0), initial point of the lattice
    for k in range(Npoints):
        xN = np.exp(-w0[k]*T)#xN = e^(-w*T) final point of the lattice
        if improved:
            x1 = np.exp(w0[k]*a)#for improved case we consider x1= e^(-w*(-a))
            xNm1 = np.exp(-w0[k]*(T-a))# xNm1 = e^(-w*(T-a))
            x_sol[k] = SolveImproved(w0[k],x0,x1,xNm1,xN)
            w_exp[k] = w0[k]*np.sqrt(1+((a*w0[k])**4)/90)

        else:
            x_sol[k] = Solve(w0[k], x0, xN)
            w_exp[k] = w0[k]*np.sqrt(1-((a*w0[k])**2)/12)

#Fitting the solution with the exponential function to retrieve w_sol
        optpar, cov = opt.curve_fit(exp, t, x_sol[k], p0 = (0.1, w0[k]))
        w_sol[k] = optpar[1]
    PlotSolution(improved,w0, w_exp, w_sol)


#Main body:
HarmonicOscillatorSolution(w0,improved, Npoints = Npoints)
