import numpy as np
from scipy.linalg import expm


#Hermitian conjugate of the matrix M
def hc(M):
    Mc = M.copy()
    return np.transpose(Mc.conj())

def SU3Random (Nmat, epsilon):
    I = np.eye(3)
    Marr = np.zeros((2*Nmat, 3, 3), dtype = complex)
    for k in range(Nmat):
        H = np.zeros((3,3),dtype = complex)
        for i in range(3):
            for j in range (3):
                H[i][j] = np.random.uniform(-1,1)+ im* np.random.uniform(-1,1)
        H = (H.copy()+hc(H))/2
        Marr[k] = expm(im*epsilon*H)
        Marr[k] = Marr[k]/(np.linalg.det(Marr[k]))**(1/3)
        Marr[k+Nmat] = hc(Marr[k])
    print('Matrices generated')
    return Marr.copy()

#The following function computes the Gamma factor for the standard QCD WIlson allocation
# containing only square terms. The gamma factor is independente on U_\mi(x), and thus is computed
#only once for each set of successive updates of the same link variable.
#It takes as input the link configuration U, which is a NsitexNsitexNsitexNsitex4x3x3 array, the
#direction mi, an integer between 0 and 3, the site x of the lattice in which we compute the factor
#and the number Nsite of lattice sites per side.
def GammaWilson(U, mi, x, Nsite):
    gamma = 0.
    incmi=np.zeros(4, dtype=int)
    incni= np.zeros(4, dtype=int)
    incmi[mi]=1
    xpmi = (x + incmi.copy()) % Nsite #site x+mi
    for ni in range(4):
        if ni != mi:
            incni[ni] = 1
            xpni = (x+incni.copy())%Nsite
            xmni = (x-incni.copy())%Nsite
            xmnipmi = (x+incmi.copy()-incni.copy())%Nsite
            incni[ni]= 0
            gamma += np.dot(np.dot(U[xpmi[0],xpmi[1],xpmi[2],xpmi[3]][ni], hc(U[xpni[0],xpni[1],xpni[2],xpni[3]][mi])), hc(U[x[0],x[1],x[2],x[3]][ni])) \
                    + np.dot(np.dot(hc(U[xmnipmi[0],xmnipmi[1],xmnipmi[2],xmnipmi[3]][ni]), hc(U[xmni[0],xmni[1],xmni[2],xmni[3]][mi])), U[xmni[0],xmni[1],xmni[2],xmni[3]][ni])
    return gamma.copy()

#The following function computes the Gamma factor for the standard QCD WIlson allocation
# containing only square terms. The gamma factor is independente on U_\mi(x), and thus is computed
#only once for each set of successive updates of the same link variable.
#It takes as input the link configuration U, which is a NsitexNsitexNsitexNsitex4x3x3 array, the
#direction mi, an integer between 0 and 3, the site x of the lattice in which we compute the factor
#and the number Nsite of lattice sites per side.
def GammaImproved(U, mi, x, Nsite):
    gamma = 0.
    incmi=np.zeros(4, dtype=int)
    incni= np.zeros(4, dtype=int)
    incmi[mi]=1
    xpmi = (x + incmi.copy()) % Nsite #site x+mi
    xmmi = (x-incmi.copy())%Nsite
    xp2mi = (x+2*incmi.copy())%Nsite
    for ni in range(4):
        if ni != mi:
            incni[ni] = 1
            xpni = (x+incni.copy())%Nsite
            xmni = (x-incni.copy())%Nsite
            xpmimni = (x+incmi.copy()-incni.copy())%Nsite
            xpmipni = (x+incmi.copy()+incni.copy())%Nsite
            xp2ni = (x+2*incni.copy())%Nsite
            xm2ni = (x-2*incni.copy())%Nsite
            xp2mimni = (x+2*incmi.copy()-incni.copy())%Nsite
            xpmim2ni = (x+incmi.copy()-2*incni.copy())%Nsite
            xmmipni = (x-incmi.copy()+incni.copy())%Nsite
            xmmimni = (x-incmi.copy()-incni.copy())%Nsite
            incni[ni]= 0

            gamma += np.dot(U[xpmi[0],xpmi[1],xpmi[2],xpmi[3]][mi],np.dot(U[xp2mi[0],xp2mi[1],xp2mi[2],xp2mi[3]][ni],np.dot(hc(U[xpmipni[0],xpmipni[1],xpmipni[2],xpmipni[3]][mi]),np.dot(hc(U[xpni[0],xpni[1],xpni[2],xpni[3]][mi]),hc(U[x[0],x[1],x[2],x[3]][ni])))))\
                    + np.dot(U[xpmi[0],xpmi[1],xpmi[2],xpmi[3]][mi],np.dot(hc(U[xp2mimni[0],xp2mimni[1],xp2mimni[2],xp2mimni[3]][ni]),np.dot(hc(U[xpmimni[0],xpmimni[1],xpmimni[2],xpmimni[3]][mi]),np.dot(hc(U[xmni[0],xmni[1],xmni[2],xmni[3]][mi]),U[xmni[0],xmni[1],xmni[2],xmni[3]][ni]))))\
                    + np.dot(U[xpmi[0],xpmi[1],xpmi[2],xpmi[3]][ni],np.dot(hc(U[xpni[0],xpni[1],xpni[2],xpni[3]][mi]), np.dot(hc(U[xmmipni[0],xmmipni[1],xmmipni[2],xmmipni[3]][mi]), np.dot(hc(U[xmmi[0],xmmi[1],xmmi[2],xmmi[3]][ni]), U[xmmi[0],xmmi[1],xmmi[2],xmmi[3]][mi]))))\
                    + np.dot(U[xpmi[0],xpmi[1],xpmi[2],xpmi[3]][ni],np.dot(U[xpmipni[0],xpmipni[1],xpmipni[2],xpmipni[3]][ni],np.dot(hc(U[xp2ni[0],xp2ni[1],xp2ni[2],xp2ni[3]][mi]),np.dot(hc(U[xpni[0],xpni[1],xpni[2],xpni[3]][ni]),hc(U[x[0],x[1],x[2],x[3]][ni])))))\
                    + np.dot(hc(U[xpmimni[0],xpmimni[1],xpmimni[2],xpmimni[3]][ni]),np.dot(hc(U[xpmim2ni[0],xpmim2ni[1],xpmim2ni[2],xpmim2ni[3]][ni]),np.dot(hc(U[xm2ni[0],xm2ni[1],xm2ni[2],xm2ni[3]][mi]),np.dot(U[xm2ni[0],xm2ni[1],xm2ni[2],xm2ni[3]][ni],U[xmni[0],xmni[1],xmni[2],xmni[3]][ni]))))\
                    + np.dot(hc(U[xpmimni[0],xpmimni[1],xpmimni[2],xpmimni[3]][ni]),np.dot(hc(U[xmni[0],xmni[1],xmni[2],xmni[3]][mi]), np.dot(hc(U[xmmimni[0],xmmimni[1],xmmimni[2],xmmimni[3]][mi]), np.dot(U[xmmimni[0],xmmimni[1],xmmimni[2],xmmimni[3]][ni],U[xmmi[0],xmmi[1],xmmi[2],xmmi[3]][mi]))))


    return gamma.copy()

#The following function updates the link variables U following a Metropolis algorithm. It takes as input the set of
#link variables U (a NsitexNsitexNsitexNsitex4x3x3 array), an array of SU(3) matrices M, the number of sites per side
#on the lattice Nsite and the optional boolean switch improved that, if set to True, uses the improved action instead
#of the Wilson action to update the variables.

def LinkUpdate(U, M, Nsite, improved=False):
    x = np.zeros(4, dtype= int) #position vector on the lattice
    if improved:
        beta = 1.719
        u0 = 0.797
    else:
        beta = 5.5
    for x0 in range (Nsite):
        x[0]=x0
        for x1 in range(Nsite):
            x[1]=x1
            for x2 in range(Nsite):
                x[2]=x2
                for x3 in range(Nsite):
                    x[3]= x3
                    for mi in range(4):
                        #old_U = U[x[0], x[1], x[2], x[3]][mi].copy() #save old U
                        gamma = GammaWilson(U,mi,x,Nsite) #compute the gamma factor
                        if improved:
                         gamma_improved = GammaImproved(U,mi,x,Nsite)
                        for _ in range(10): #update the same link 10 times before moving on to thermalize the lattice faster
                            k = np.random.randint(2,len(M))
                            if improved:
                                dS = -(beta/3)*((5/(3*u0**4))*np.real(np.trace(np.dot((np.dot(M[k], U[x[0],x[1],x[2],x[3]][mi])-U[x[0],x[1],x[2],x[3]][mi]),gamma)))-1/(12*u0**6)*np.real(np.trace(np.dot((np.dot(M[k], U[x[0],x[1],x[2],x[3]][mi])-U[x[0],x[1],x[2],x[3]][mi]),gamma_improved))))#change of teh improved action
                            else:
                                dS = -beta/(3)*np.real(np.trace(np.dot((np.dot(M[k].copy(),U[x[0],x[1],x[2],x[3]][mi].copy())-U[x[0],x[1],x[2],x[3],mi].copy()),gamma.copy()))) # change in action
                            if dS<0 or np.exp(-dS)>np.random.rand(): #condition to update link
                                U[x[0],x[1],x[2],x[3]][mi] = np.dot(M[k].copy(),U[x[0],x[1],x[2],x[3]][mi].copy())  # update U

#The following function computes the Wilson line on an axa plaquette starting in point x.
#It takes as input the set of the configurations of the link variables U, the position vector on the lattice x
#and the number of sites per lattice side Nsite, and outputs the value of the Wilson line integral axa.
def Wilson_axa(U,x,Nsite):
    Wilsonaxa = 0.
    incmi = np.zeros(4, dtype=int)
    incni = np.zeros(4, dtype=int)
    for mi in range(4):
        incmi[mi] = 1
        xpmi = (x+incmi.copy())%Nsite
        for ni in range (mi):
            incni[ni]=1
            xpni = (x+incni.copy())%Nsite
            incni[ni]=0
            Wilsonaxa += np.trace(np.dot(U[x[0],x[1],x[2],x[3]][mi],\
                        np.dot(U[xpmi[0],xpmi[1],xpmi[2],xpmi[3]][ni],\
                        np.dot(hc(U[xpni[0],xpni[1],xpni[2],xpni[3]][mi]),hc(U[x[0],x[1],x[2],x[3]][ni])))))
        incmi[mi]=0
    return np.real(Wilsonaxa)/(3*6)

#The following function computes the Wilson line on an ax2a rectangle starting in point x.
#It takes as input the set of the configurations of the link variables U, the position vector on the lattice x
#and the number of sites per lattice side Nsite, and outputs the value of the Wilson line integral ax2a.
def Wilson_ax2a(U,x,Nsite):
    Wilsonax2a = 0.
    incmi = np.zeros(4, dtype=int)
    incni = np.zeros(4, dtype=int)
    for mi in range(4):
        incmi[mi] = 1
        xpmi = (x+incmi.copy())%Nsite
        for ni in range (3,mi,-1):
            if ni!= mi:
                incni[ni]=1
                xpni = (x+incni.copy())%Nsite
                xpmipni = (xpmi+incni.copy())%Nsite
                xp2ni = (xpni+incni.copy())%Nsite
                incni[ni]=0
                Wilsonax2a+=np.real(np.trace(np.dot(U[x[0],x[1],x[2],x[3]][mi], \
                    np.dot(U[xpmi[0],xpmi[1],xpmi[2],xpmi[3]][ni], \
                    np.dot(U[xpmipni[0],xpmipni[1],xpmipni[2],xpmipni[3]][ni], \
                    np.dot(hc(U[xp2ni[0],xp2ni[1],xp2ni[2],xp2ni[3]][mi]),\
                    np.dot(hc(U[xpni[0],xpni[1],xpni[2],xpni[3]][ni]),hc(U[x[0],x[1],x[2],x[3]][ni]))))))))
        incmi[mi]=0
    return Wilsonax2a/(3*6)

def WilsonLoopMC(U,M,Nsite,Ncf,Ncor, improved=False):
    WilsonLoop_axa = np.zeros(Ncf)
    WilsonLoop_ax2a = np.zeros(Ncf)
    x = np.zeros(4, dtype=int)
    avgWL_axa=0.
    avgWL_ax2a=0.
    avgWLsquared_axa=0.
    avgWLsquared_ax2a=0.
    for _ in range (2*Ncor):
        LinkUpdate(U,M,Nsite,improved=improved) #lattice thermalization
    print('Thermalized')
    if improved:
        file = open("Results_Improved_Action.txt",'w')
    else:
        file = open("Results_Wilson_Action.txt", 'w')
    file.write("Itn.    WL axa              WL ax2a")
    for alpha in range(Ncf):
        for _ in range(Ncor):
            LinkUpdate(U,M,Nsite,improved=improved)
        for x0 in range(Nsite):
            x[0] = x0
            for x1 in range(Nsite):
                x[1] = x1
                for x2 in range (Nsite):
                    x[2] = x2
                    for x3 in range(Nsite):
                        x[3] = x3
                        WilsonLoop_axa[alpha] += Wilson_axa(U,x,Nsite)
                        WilsonLoop_ax2a[alpha] += Wilson_ax2a(U,x,Nsite)

        WilsonLoop_axa[alpha] = WilsonLoop_axa[alpha]/Nsite**4
        WilsonLoop_ax2a[alpha] = WilsonLoop_ax2a[alpha]/Nsite**4
        avgWL_axa += WilsonLoop_axa[alpha]
        avgWLsquared_axa += WilsonLoop_axa[alpha]**2
        avgWL_ax2a +=  WilsonLoop_ax2a[alpha]
        avgWLsquared_ax2a +=  WilsonLoop_ax2a[alpha]**2
        print('Iteration %f of %g done' %(alpha+1, Ncf))
        file.write('\n')
        file.write(str(alpha+1))
        file.write('   ')
        file.write(str(WilsonLoop_axa[alpha]))
        file.write('   ')
        file.write(str(WilsonLoop_ax2a[alpha]))
        file.write('   ')

    avgWL_axa = avgWL_axa/Ncf
    avgWLsquared_axa = avgWLsquared_axa/Ncf
    avgWL_ax2a = avgWL_ax2a/Ncf
    avgWLsquared_ax2a = avgWLsquared_ax2a/Ncf

    delta_axa = np.sqrt((avgWLsquared_axa -avgWL_axa**2)/Ncf)
    delta_ax2a = np.sqrt((avgWLsquared_ax2a -avgWL_ax2a**2)/Ncf)

    file.write('\n')
    file.write('MC average of WL axa:  ')
    file.write(str(avgWL_axa))
    file.write('+-')
    file.write(str(delta_axa))
    file.write('\n')
    file.write('MC average of WL ax2a:  ')
    file.write(str(avgWL_ax2a))
    file.write('+-')
    file.write(str(delta_ax2a))
    file.close()
    return avgWL_axa, delta_axa, avgWL_ax2a, delta_ax2a

im= 1j
Nsite = 8
improved = True
Ncf= 10
Ncor = 50
Nmatrices = 100
epsilon = 0.24
U = np.zeros((Nsite,Nsite,Nsite,Nsite,4,3,3), dtype= complex)
SU3Matrices = SU3Random(Nmatrices,epsilon)

for x in range (Nsite):
    for y in range (Nsite):
        for z in range(Nsite):
            for t in range(Nsite):
                for mi in range(4):
                    U[x,y,z,t][mi] = np.eye(3)

WL_axa, DeltaWL_axa, WL_ax2a, DeltaWL_ax2a = WilsonLoopMC(U,SU3Matrices,Nsite,Ncf,Ncor,improved=improved)
print('MC average of WL axa = ', WL_axa, '+/-' ,DeltaWL_axa)
print('MC average of WL ax2a = ', WL_ax2a, '+/-' ,DeltaWL_ax2a)
