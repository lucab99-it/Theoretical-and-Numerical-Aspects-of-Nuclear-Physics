import numpy as np
from scipy import optimize as opt
import matplotlib.pyplot as plt

a=0.5 #lattice spacing
N = 20 #total lattice sites
T = a*N #total time
Npoints = 100
t = np.linspace(0, T, N) #time axis
Option = 'Improved'
w0=np.linspace(0,4,Npoints) #known frequency of the oscillator


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


def Solve(w0,x0, xN):
    h = np.zeros(N)
    h[0] = -x0
    h[-1] = xN
    M = SystemMatrix(w0)
    x = np.linalg.solve(M,h)
    return x

def SolveImproved(w0,x0,x1,xNm1,xN):
    h = np.zeros(N)
    h[0]=(1/12)*x1-(4/3)*x0
    h[1] = x0*(1/12)
    h[N-2] = xN*(1/12)
    h[N-1] = (1/12)*xNm1-(4/3)*xN
    M = SystemMatrixImproved(w0)
    x = np.linalg.solve(M,h)
    return x

def exp(x,A,w):
    return A*np.exp(-w*x)


def PlotSolution(Option, w0, w_exp, w_sol):
    fig = plt.figure()
    if Option == 'Std':
        l1=plt.plot(w0,w_exp,label = r'$\omega_0\sqrt{1-\frac{(a\omega_0)^2}{12}}$', linestyle = 'dotted')
        l2 = plt.plot(w0,w_sol, label = 'Numerical solution', linestyle='none', marker = '.')

    if Option == 'Improved':
        l1=plt.plot(w0,w_exp,label = r'$\omega_0\sqrt{1+\frac{(a\omega_0)^4}{90}}$', linestyle = 'dotted')
        l2 = plt.plot(w0,w_sol, label = 'Numerical solution', linestyle='none', marker = '.')

    plt.xlabel(r'$\omega_0$')
    plt.ylabel('$\omega$')
    plt.legend(loc='lower right')
    plt.show()


def HarmonicOscillatorSolution(w0,Option, Npoints=100):
    x_sol = np.zeros((Npoints,N))
    w_sol = np.zeros(Npoints)
    w_exp = np.zeros(Npoints)
    x0 = np.exp(0)
    for k in range(Npoints):
        xN = np.exp(-w0[k]*T)
        if Option == 'Improved':
            x1 = np.exp(w0[k]*a)
            xNm1 = np.exp(-w0[k]*(T-a))
            x_sol[k] = SolveImproved(w0[k],x0,x1,xNm1,xN)
            w_exp[k] = w0[k]*np.sqrt(1+((a*w0[k])**4)/90)

        else:
            x_sol[k] = Solve(w0[k], x0, xN)
            w_exp[k] = w0[k]*np.sqrt(1-((a*w0[k])**2)/12)

        optpar, cov = opt.curve_fit(exp, t, x_sol[k], p0 = (0.1, w0[k]))
        w_sol[k] = optpar[1]
    PlotSolution(Option,w0, w_exp, w_sol)



HarmonicOscillatorSolution(w0,Option, Npoints = Npoints)
