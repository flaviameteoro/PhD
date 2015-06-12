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
xhulp[:,0] = F         
pert = 0.05
pospert = np.ceil(N/2.0)-1
xhulp[pospert,0] = F+pert
spinup=1999
for j in range(spinup):
    force = np.zeros(N)
    xhulp[:,j+1] = mod.lorenz96(xhulp[:,j],force,dt)   
xtrue[:,0] = xhulp[:,spinup]

for j in range(J):
   
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
    y[:,tobs[t]] = np.dot(H,xtrue[:,tobs[t]])+random[:]  

print 'observations created'


# Creating the DM-dimensional time-delay vectors
# data vector Y
# Creating DM-dimensional map from physical space to delay embedding space - vector S
DM = 5
Y = np.zeros(DM)            
S = np.zeros(DM)            


# Defining initial x, dF/dx and J    
x = np.zeros([N,J+1])              
# Aplying randomness to x, not to be equal to xtrue 
random = np.random.randn(N)
x[:,0] = xtrue[:,0] + random
dfdx = np.zeros([N,N]) 
dflast = np.zeros([N,N]) 

# Creating initial condition for J (Jab = 1 when a=b)
Jac = np.zeros([N,N])                 
for i in range(N):
    Jac[i,i] = 1.

Jac0 = np.copy(Jac)                 
dsdx = np.zeros([DM,N])  
dxds = np.zeros([N,DM]) 

       
# Main loop        
mcn = 0
summation = np.zeros(N)
run = 200
count = 1
integ = DM*ns   
differ = np.zeros(DM)  
scount = 0
SE = 0
countse = 1

##for t in range(10, run):
for t in range(run):
    if (t == tobs[mcn]): 
    
        ldelay = integ + mcn*ns
    
        for m in range(t,ldelay+1):            
            if m == t:
                random = np.zeros(N)
                x[:,m+1] = mod.lorenz96(x[:,m],random,dt) 
    
                dsdx[scount,:] = Jac[0,:] 
                
                dfdx = mod.df(x[:,m])
                Jac = Jac + dt*(np.dot(dfdx,Jac))
                
                #Jacvec = mod.rk4_J(x[:,m],dt) 
                #Jac = Jacvec.reshape(N,N)
                #Jaclast = Jac 
                #dflast = dfdx               
                #Jac = Jaclast + dt*(np.dot(dflast,Jaclast))
                #dfdx = mod.df(x[:,m+1])
                #Jac = Jaclast + 0.5*dt*(np.dot(dfdx,Jac)-(np.dot(dflast,Jaclast)))                 
                #Jac = Jac + 0.5*dt*(3*(np.dot(dfdx,Jac)-(np.dot(dflast,Jaclast))))            
                
                scount = scount + 1
            else:
                random = np.zeros(N)
    
                x[:,m+1] = mod.lorenz96(x[:,m],random,dt)  
    
                dfdx = mod.df(x[:,m])
                Jac = Jac + dt*(np.dot(dfdx,Jac))
                
                #Jacvec = mod.rk4_J(x[:,m],dt) 
                #Jac = Jacvec.reshape(N,N)
                #Jaclast = Jac 
                #dflast = dfdx               
                #Jac = Jaclast + dt*(np.dot(dflast,Jaclast))
                #dfdx = mod.df(x[:,m+1])
                #Jac = Jaclast + 0.5*dt*(np.dot(dfdx,Jac)-(np.dot(dflast,Jaclast)))                 
                #Jac = Jac + 0.5*dt*(3*(np.dot(dfdx,Jac)-(np.dot(dflast,Jaclast))))                
                
                
                newcount = (count+1)*ns
                if (m+1 == newcount):          
                    dsdx[scount,:] = Jac[0,:] 
    
                    scount = scount + 1
                    count = count + 1
    
        for d in range(DM): 
            td = t+d*ns
    
            Y[d] = y[:,td]                 
            S[d] = x[0,td]                
                
        dxds = np.linalg.pinv(dsdx)
        dsdx = np.zeros([DM,N]) 
        
    
        scount = 0
        mcn = mcn + 1        
        count = mcn + 1
        
        # Full dynamics
        dif = Y - S
    
        summation = np.dot(dxds,dif)  
    
        coup = g*summation
        
        Jac = Jac0
    
        random = np.zeros(N)
    
        x[:,t+1] = x[:,t] + dt*(x[:,t] + coup)    
    
        for d in range(t+1):
            dd = np.zeros([N,t+1]) 
            dd[:,d] = xtrue[:,d] - x[:,d]
           
        SE = np.sqrt(np.mean(np.square(dd)))            
        print 'SE is', SE
    
        plt.plot(d,SE,'b*') 
        plt.yscale('log')
        plt.hold(True)
    
    else:
        random = np.zeros(N)
        x[:,t+1] = mod.lorenz96(x[:,t],random,dt)  
    
        dfdx = mod.df(x[:,t])
        Jac = Jac + dt*(np.dot(dfdx,Jac))
                
        #Jacvec = mod.rk4_J(x[:,m],dt) 
        #Jac = Jacvec.reshape(N,N)
        
        #Jactn = Jac 
        #Jac = Jac + dt*(np.dot(dfdx,Jac))
        #dflast = np.copy(dfdx)    

        #Jaclast = Jac

 
    
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
            plt.plot(y[0,:],'r.',label='obs')   ## create y with no zeros to plot correctly ###
            plt.hold(True)      
      
        
        plt.plot(x[i,:],'g',label='X')
        plt.hold(True)
        plt.plot(xtrue[i,:],'b-',linewidth=2.0,label='truth')
        plt.hold(True)
        
        plt.ylabel('x['+str(i)+']')
        plt.xlabel('time steps')
    plt.show()



