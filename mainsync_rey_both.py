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
ns = 10  #number of time steps between obs
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
######xhulp[:,0] = F          ## CHANGE HERE IF WE WANT TO VARY FORCING PARAMETERS AS TABLE 1????#####
######pert = 0.05
######pospert = np.ceil(N/2.0)-1
######xhulp[pospert,0] = F+pert
######spinup=1999
######for j in range(spinup):
######    force = np.zeros(N)
######    xhulp[:,j+1] = mod.lorenz96(xhulp[:,j],force,dt)   ### this line returns 1 column for the 20 variables at each loop ####
######xtrue[:,0] = xhulp[:,spinup]
xtrue[:,0] = np.random.rand(N)
print 'xtrue', xtrue
for j in range(J):
    #random = np.random.randn(N)
    #force = np.dot(scov_model,random)
    force = np.zeros(N)                                 
    xtrue[:,j+1] = mod.lorenz96(xtrue[:,j],force,dt)   
print 'truth created'
######print xtrue.shape
# Adding 21st state variable
######forc = np.zeros([1,J+1])
######print forc.shape
######forc[:,:] = F
######print forc 
######xtrue = np.append(xtrue,forc, axis=0)
#####print 'New xtrue', xtrue.shape
#print xtrue[0,1]


# Creating the observations
#if (observations == 1):
#####NM = J*tau #number of measurement times   # (Increasing DM is equivalent to increasing the number of measurements!) 
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
print 'H', H
#  observations   
y = np.zeros([MO,J+1])   # verify: J+1???? why not J here? it's definition, not a loop HOWEVER IT CONFLICTS WITH xtrue SHAPE#           
tobs = np.zeros(NM)
for z in range(int(NM)):
    tobs[z] = (z+1)*ns
########random=np.random.randn(MO)
random = np.zeros(MO)
#########y[:,tobs[t]] = np.dot(H,xtrue[:,tobs[t]])+random[:]
y = np.dot(H,xtrue)
print 'y is', y
print 'xtrue is', xtrue
print 'observations created'


# Creating the DM-dimensional time-delay vectors
# data vector Y
# Creating DM-dimensional map from physical space to delay embedding space - vector S
DM = 10
Y = np.zeros(DM)            
S = np.zeros(DM)            


# Defining initial x, dF/dx and J    
x = np.zeros([N,J+1])               ## verify: J+1???? why not J here? it's definition, not a loop# 
# Aplying randomness to x, not to be equal to xtrue ***** PJ - 19/03 *****
randomini = np.random.rand(N)-0.5
x[:,0] = xtrue[:,0] + randomini
dfdx = np.zeros([N,N]) 

# Creating initial condition for J (Jab = 1 when a=b)
Jac = np.zeros([N,N])                 
for i in range(N):
    Jac[i,i] = 1.
#print 'J is', Jac
Jac0 = np.copy(Jac)                 
dsdx = np.zeros([DM,N])  
dxds = np.zeros([N,DM]) 

       
# Main loop        
mcn = 0
summation = np.zeros(N)
run = 20
count = 1
integ = DM*ns   
differ = np.zeros(DM)  
scount = 0
SE = 0
countse = 1
jacsize = (N)**2

##for t in range(10, run):
for t in range(run):
    #if (t == tobs[mcn]):          
    ldelay = integ + mcn*ns   
   
    dsdx[0,:] = H
    Jac = Jac0   
    #print 'Jac is', Jac

    for m in range(t,ldelay+1):            
            #####if m == t:
            #####    random = np.zeros(N+1)
            #####    x[:,m+1] = mod.lorenz96(x[:,m],random,dt) 
               
            #####    dsdx[scount,:] = Jac[0,:] 
            #####    dfdx = mod.df(x[:,m])
            #####    Jac = Jac + dt*(np.dot(dfdx,Jac))
               
            #####    scount = scount + 1
            #####else:
        random = np.zeros(N)
                #random = np.random.randn(N+1)
        x[:,m+1] = mod.lorenz96(x[:,m],random,dt)  
        #print 'x size', x[:,m].shape      
        
        dfdx = mod.df(x[:,m])

        Jacvec = Jac.reshape(jacsize)
        #print 'Jacvec size', Jacvec.shape
        randJ = np.zeros(jacsize)
        Jacstep = mod.lorenz96(Jacvec,randJ,dt)  
        Jac = Jacstep.reshape(N,N)
        #print 'Jac is', Jac
                ######Jac = Jac + dt*(np.dot(dfdx,Jac))  
                #####Jac = Jac*(np.exp(dt*dfdx))   
                #Jaclast = Jac
        newcount = (count+1)*ns
        if (m+1 == newcount):          
            dsdx[scount,:] = Jac[0,:] 
                   
            scount = scount + 1
            count = count + 1
       
    for d in range(DM): 
        td = t+d*ns
            #td = d*ns              
        Y[d] = y[:,td]                 
        S[d] = x[0,td]                
                
    dxds = np.linalg.pinv(dsdx)
    dsdx = np.zeros([DM,N]) 
        #check = np.allclose(dsdx, np.dot(dsdx, np.dot(dxds, dsdx)))
        #print 'check', check
        #print 'dxds', dxds
        
        #count = 0
    scount = 0
    mcn = mcn + 1        
    count = mcn + 1
        
        # Full dynamics
    dif = Y - S
        ###################dif = Y[0] - S[0]
        ####print 'Y - S', dif
        #print 'dif is', dif
    summation = np.dot(dxds,dif)  
        ###################summation = np.dot(dxds[:,0],dif)  
        ###print 'sum', summation               
    coup = g*summation
        ###############coup = g*dt*summation
        ################x[:,t] = x[:,t] + coup    
        ################x[:,t] = x[:,t] + dt*(x[:,t] + coup)  
        ################x[:,t] = dt*(x[:,t] + coup)     
    x[:,t] = x[:,t] + coup    
   
    random = np.zeros(N)
    x[:,t+1] = mod.lorenz96(x[:,t],random,dt) 
      
    if (t == tobs[mcn]): 
        dd = np.zeros([N,J+1]) 
        for d in range(t+1):
            dd[:,d] = xtrue[:,d] - x[:,d]
                    
        SE = np.sqrt(np.mean(np.square(dd)))            
        print 'SE is', SE

        plt.plot(t,SE,'b*') 
        plt.yscale('log')
        plt.hold(True)
        
        # With new x (after coupling)
        #Jac = Jac0
        #print 'Jac', Jac
    ####random = np.zeros(N+1)
        ################x[:,t+1] = mod.lorenz96(x[:,t],random,dt) 
    ####x[:,t+1] = x[:,t] + dt*(x[:,t] + coup)    
        ###print 'x', 'at', t+1, x[:,t+1]
        ################dfdx = mod.df(x[:,t])
        ################Jac = Jac + dt*(np.dot(dfdx,Jac))
        #####Jac = Jac*(np.exp(dt*dfdx))   
    
    ####if (t > 99):   
    ####    for z in range(DM): 
                #td = t+z*ns
    ####        step = (z+countse)*ns              
    ####        Y[z] = y[:,step]                 
    ####        S[z] = x[0,step]          
    ####    countse = countse + 1

        ####for n in range(DM):                       
        ####    differ[n] = (Y[n]-S[n])**2        
        ####SE = np.sqrt((1./DM)*(np.sum(differ))) 
        ####print 'SE is', SE     

        ####plt.plot(t,SE,'b*')
        ####plt.hold(True)

    
    ####else:
    ####    random = np.zeros(N+1)
        #####x[:,t+1] = mod.lorenz96(x[:,t],random,dt)  
        ###print 'x', 'at', t+1, x[:,t+1]
        #print 'x is', x[:,t+1]
        #####dfdx = mod.df(x[:,t])
        #####Jac = Jac + dt*(np.dot(dfdx,Jac))
        ######Jac = Jac*(np.exp(dt*dfdx))   
        ####print 'Jacs are', Jac
        #####Jactn = Jac

 
    
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



