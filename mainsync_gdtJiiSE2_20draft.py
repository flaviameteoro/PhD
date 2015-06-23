# Implementation of a state and parameter estimation with sparse obs using SYNCHRONISATION, with the Lorenz 96 model

import numpy as np
import matplotlib.pyplot as plt
import functions as m
import model as mod
from time import time
from scipy.linalg import toeplitz
from scipy.linalg import sqrtm

plt.close('all')

#observations = 1


#Plots to make
plotTruth = 0
plotTraj = 1
plotTrace = 0


# Parameter settings
J = 10000      # number of steps
Obs = 100     # obs window
N = 20       # number of state variables
dt = 0.01  #1    # deltat                            ## SEE IF THIS TIME STEP GUARANTEES STABILITY AS DT 0.025!!## 
tau= 0.1  #10     # constant time delay (=10dt)
obsgrid = 4  #number of observations at analysis time (1=all observed, 2=every other observed, 3=half state observed, 4=1 variable)
ns = 2  #number of time steps between obs
g = 10 #0.1      # coupling term

# Choose random seed - With the seed reset,same numbers will appear every time 
# Controlling experiment... 
r=1 
np.random.seed(r)                               


# Creating the truth run
xhulp = np.zeros([N,2000])
xtrue = np.zeros([N,J+1])
force = np.zeros(N)

#spin up
F=8.17
######xhulp[:,0] = F          
######pert = 0.05
######pospert = np.ceil(N/2.0)-1
######xhulp[pospert,0] = F+pert
######spinup=1999
######for j in range(spinup):
######    force = np.zeros(N)
######    xhulp[:,j+1] = mod.lorenz96(xhulp[:,j],force,dt)  
######xtrue[:,0] = xhulp[:,spinup]
xtrue[:,0] = np.random.rand(N)
#####print 'xtrue', xtrue
for j in range(J):
    #random = np.random.randn(N)
    #force = np.dot(scov_model,random)
    force = np.zeros(N)                                 
    xtrue[:,j+1] = mod.lorenz96(xtrue[:,j],force,dt)   
print 'truth created'
print xtrue.shape


# Creating the observations
#if (observations == 1):

NM = J/ns
    # Select an observation operator
if obsgrid == 1:
        # Option 1: Observe all
    observed_vars = range(N)
elif obsgrid == 2:
        # Option 2: Observe every other variable
    observed_vars = range(0,N,2)
elif obsgrid == 3:
        # Option 3: Observe left half of variables ("land/sea" configuration)
    observed_vars = range(N/2)
elif obsgrid == 4:
        # Option 4: Observe 1 variable (article configuration)
    observed_vars = range(1)    
MO = len(observed_vars) 

# adding N+1 for including the 21st variable in the future                          
H = np.zeros([MO,N])       
for i in range(MO):
    H[i,observed_vars[i]] = 1.0                        

#  observations   
y = np.zeros([MO,J+1])  
tobs = np.zeros(NM)
for t in range(int(NM)):
    tobs[t] = (t+1)*ns
   
    random = np.zeros(MO)
    ####y[:,tobs[t]] = np.dot(H,xtrue[:,tobs[t]])+random[:]  
y = np.dot(H,xtrue)

print 'observations created'


# Creating the DM-dimensional time-delay vectors
# data vector Y
# Creating DM-dimensional map from physical space to delay embedding space - vector S
DM = 10
Y = np.zeros(DM)            
S = np.zeros(DM)            


# Defining initial x, dF/dx and J    
x = np.zeros([N,J+1])              
# Aplying randomness to x, not to be equal to xtrue 
#####random = np.random.randn(N)
#####x[:,0] = xtrue[:,0] + random
randomini = np.random.rand(N)-0.5
x[:,0] = xtrue[:,0] + randomini
dfdx = np.zeros([N,N]) 

# Creating initial condition for J (Jab = 1 when a=b)
Jac = np.zeros([N,N])                 
for i in range(N):
    Jac[i,i] = 1.

Jac0 = np.copy(Jac)                 
dsdx = np.zeros([DM,N])  
dxds = np.zeros([N,DM]) 

       
# Main loop        
#mcn = 0
summation = np.zeros(N)
run = 1000
count = 1
integ = DM*ns
differ = np.zeros(DM)  
#scount = 0
SE = 0
countse = 1

ntau = tau/dt
integ2 = DM*ntau   
mcn = 0
pinv_tol = 2.2204e-16
max_pinv_rank = N
xtran = np.zeros([N,1]) 

x[:,10] = x[:,0]
for z in range(run):
    
    scount = 0         
    first = z + DM    
    last = int(integ2 + mcn) 
    #print 'last', last

    S[scount] = x[0,z]
    Y[scount] = y[0,z]
    dsdx[scount,:] = Jac0[0,:]  
    scount = scount + 1
    
    Jac = Jac0
    
    x[:,first] = x[:,z]
    #block = first+ntau
    
    for m in range(first,last): 
        block = first+(scount*ntau)           
        if (m+1  == block): 
            #random = np.zeros(N)
            #x[:,m+1] = mod.lorenz96(x[:,m],random,dt) 
    
            ###dfdx = mod.df(x[:,m])
            ###Jac = np.dot(dfdx,Jac)
            #Jac = Jac + dt*(np.dot(dfdx,Jac))   
            print 'Jac for m', m, 'is', Jac
            print 'x at', m, 'is', x[:,m]
            Jacsize = N**2
            Jacv = Jac.reshape(Jacsize) 
            Jacvec = Jacv.reshape(Jacsize,1)
            dxdt = mod.dxdt(x[:,m],Jacvec,N,dt)
            xtran = mod.rk4(dxdt,dt)
            x[:,m+1] = xtran[0:N]
            Jact = xtran[N:]
            Jac = Jact.reshape(N,N)
            print 'Jac for m', m+1, 'is', Jac
            print 'x at', m+1, 'is', x[:,m+1]
            #dfdx = mod.df(x[:,m])
            #dfdx = mod.rk4_J2(x[:,m],dfdx,dt)
            

            #dfdx = mod.df(x[:,m])
            #Jac = Jac + dt*(np.dot(dfdx,Jac))
            #Jacsize = N**2
            #Jacvec = Jac.reshape(Jacsize) 
            #Jac = mod.rk4_J(x[:,m],Jacvec,dt)
            #print 'Jac', Jac
            ####Jac = dt*(np.dot(dfdx,Jac)) 
            #Jac = np.dot(dfdx,Jac)  
                                  
            ####Jacsize = N**2
            ####Jacvec = Jac.reshape(Jacsize)
            ####random = np.zeros(Jacsize)
            ####Jacvecnew = mod.lorenz96(Jacvec,random,dt)
            ####Jac = Jacvecnew.reshape(N,N)

            S[scount] = x[0,m+1]
            dsdx[scount,:] = Jac[0,:] 
            scount = scount + 1
            #mcn = mcn + 1      

        else:
            #random = np.zeros(N)
            #x[:,m+1] = mod.lorenz96(x[:,m],random,dt)  
    
            ###dfdx = mod.df(x[:,m])
            ###Jac = np.dot(dfdx,Jac)
            #Jac = Jac + dt*(np.dot(dfdx,Jac))   
            print 'Jac for m', m, 'is', Jac
            print 'x at', m, 'is', x[:,m]
            Jacsize = N**2
            Jacv = Jac.reshape(Jacsize) 
            Jacvec = Jacv.reshape(Jacsize,1)
            dxdt = mod.dxdt(x[:,m],Jacvec,N,dt)
            xtran = mod.rk4(dxdt,dt)
            x[:,m+1] = xtran[0:N]
            Jact = xtran[N:]
            Jac = Jact.reshape(N,N)
            print 'Jac for m', m+1, 'is', Jac
            print 'x at', m+1, 'is', x[:,m+1]
            #dfdx = mod.df(x[:,m])
            #dfdx = mod.rk4_J2(x[:,m],dfdx,dt)
            

            #dfdx = mod.df(x[:,m])
            #Jac = Jac + dt*(np.dot(dfdx,Jac))
            #Jacsize = N**2
            #Jacvec = Jac.reshape(Jacsize) 
            #Jac = mod.rk4_J(x[:,m],Jacvec,dt)
            #print 'Jac', Jac
            ####Jac = dt*(np.dot(dfdx,Jac)) 
            #Jac = np.dot(dfdx,Jac)           
            
            ####Jacsize = N**2
            ####Jacvec = Jac.reshape(Jacsize)
            ####random = np.zeros(Jacsize)
            ####Jacvecnew = mod.lorenz96(Jacvec,random,dt)
            ####Jac = Jacvecnew.reshape(N,N)
    
    #for d in range(DM):     
    for d in range(2,DM): 
        td = z+d*ntau
        Y[d] = y[:,td]                 
        #Y[d] = y[:,td]                 
        #S[d] = x[0,td]                
    
    print 'dsdx', dsdx            
    #dxds = np.linalg.pinv(dsdx)
    U, G, V = mod.svd(dsdx)
    #print 'U', U
    #print 'U size', U.shape
    #print 'G', G
    #print 'G size', G.shape
    #print 'V', V
    #print 'V size', V.shape
    #G = np.diag(G)
    #print 'G', G
    for k in range(len(G)):
        mask = np.ones(len(G))        
        if G[k] >= pinv_tol:
        #for G[k] >= pinv_tol:        
        #if float(G[k]) < pinv_tol:
            mask[k] = 1
        else:
            mask[k] = 0
    #print 'mask', mask
    r = min(max_pinv_rank,sum(mask)) 
    Ginv = G[:r]**(-1) 
    #print 'Ginv', Ginv
    Ginv = np.diag(Ginv)
    #print 'Ginv', Ginv
    #print 'Ginv size', Ginv.shape
    #print 'V[:,:r] size', V[:,:r].shape
    #print 'U[:,:r] size', U[:,:r].shape
    dxds = np.dot(V[:,:r],Ginv)   
    dxds = np.dot(dxds,(np.transpose(U[:,:r])))  
    #print 'dxds', dxds
    #scount = 0
    #mcn = mcn + 1        
    #count = mcn + 1
    print 'dxds', dxds    
    # Full dynamics
    dif = Y - S
    print 'dif is', dif
    summation = np.dot(dxds,dif)  
    
    coup = g*summation
    print 'coup is', coup    
    #Jac = Jac0
    
    random = np.zeros(N)
    
    #x[:,z+1] = x[:,z] + dt*(x[:,z] + coup)    
    xtran2 = x[:,z] + coup
    x[:,z+1] = mod.lorenz96(xtran2,random,dt) 
    print 'xnew at', z+1, 'is', x[:,z+1]
    ###x[:,z+1] = mod.lorenz96(x[:,z],random,dt)     
    ###x[:,z+1] = x[:,z+1] + coup
    #########xtran[:,z] = mod.l95(x[:,z],dt) + coup 
    #########x[:,z+1] = mod.rk4(xtran[:,z],dt) 

    for d in range(z+1):
        ####dd = np.zeros([N,z+2]) 
        ####dd[:,d] = xtrue[:,d] - x[:,d]
        dd = np.zeros([1,z+1])   
        dd[:,d] = xtrue[0,d] - x[0,d]
    SE = np.sqrt(np.mean(np.square(dd)))            
    print 'SE for', z+1, 'is', SE
    
    plt.plot(d,SE,'b*') 
    plt.yscale('log')
    plt.hold(True)
    
    
    mcn = mcn + 1      
    
plt.draw()
 #Figures-----------------------------------------------------------------------------------------------------------------------------
#Truth Figures
if (plotTruth == 1):
    plt.figure().suptitle('Truth')
    plotArray = np.zeros([J+1,N])
    for n in range(N):
        plotArray[:,n]=xtrue[n,:]
    C = plt.contourf(np.arange(N),np.linspace(0,J,J+1),plotArray,10)
    plt.hold(True)
    plt.contour(np.arange(N),np.linspace(0,J,J+1),plotArray,10,colors='k',linestyles='solid')
    plt.xlabel('variable number')
    plt.ylabel('time')
    plt.title('Hovmoller diagram')
    plt.colorbar(C)
    plt.show()

#Plot samples, truth, observations for SIR
if (plotTraj == 1):
    plt.figure(figsize=(12, 10)).suptitle('Full sync - J reinitialized')
    for i in range(N/3):
        plt.subplot(np.ceil(N/8.0),2,i+1)
        if i == 0:  
            plt.plot(y[0,:],'r.',label='obs')   
            plt.hold(True)      
      
        
        plt.plot(x[i,:],'g',label='X')
        plt.hold(True)
        plt.plot(xtrue[i,:],'b-',linewidth=2.0,label='truth')
        plt.hold(True)
        
        plt.ylabel('x['+str(i)+']')
        plt.xlabel('time steps')
    plt.show()



