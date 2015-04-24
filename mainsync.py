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
dt = 0.01    # deltat                            ## SEE IF THIS TIME STEP GUARANTEES STABILITY AS DT 0.025!!## 
tau=0.1      # constant time delay (=10dt)
obsgrid = 4  #number of observations at analysis time (1=all observed, 2=every other observed, 3=half state observed, 4=1 variable)
ns = 10       #number of time steps between obs
g = 10      # coupling term

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
xhulp[:,0] = F          ## CHANGE HERE IF WE WANT TO VARY FORCING PARAMETERS AS TABLE 1????####
pert = 0.05
pospert = np.ceil(N/2.0)-1
xhulp[pospert,0] = F+pert
spinup=1999
for j in range(spinup):
    force = np.zeros(N)
    xhulp[:,j+1] = mod.lorenz96(xhulp[:,j],force,dt)   ### this line returns 1 column for the 20 variables at each loop ####
xtrue[:,0] = xhulp[:,spinup]

for j in range(J):
    #random = np.random.randn(N)
    #force = np.dot(scov_model,random)
    force = np.zeros(N)                                 
    xtrue[:,j+1] = mod.lorenz96(xtrue[:,j],force,dt)   
print 'truth created'
print xtrue.shape
# Adding 21st state variable
forc = np.zeros([1,J+1])
print forc.shape
forc[:,:] = F
print forc 
xtrue = np.append(xtrue,forc, axis=0)
print 'New xtrue', xtrue.shape
#print xtrue[0,1]


# Creating the observations
#if (observations == 1):
NM = J*tau #number of measurement times   # (Increasing DM is equivalent to increasing the number of measurements!) 
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
H = np.zeros([MO,N+1])       
for i in range(MO):
    H[i,observed_vars[i]] = 1.0                        

#  observations   
y = np.zeros([MO,J+1])   # verify: J+1???? why not J here? it's definition, not a loop HOWEVER IT CONFLICTS WITH xtrue SHAPE#                                                   
tobs = np.zeros(NM)
for t in range(int(NM)):
    tobs[t] = (t+1)*ns
    random=np.random.randn(MO)
    y[:,tobs[t]] = np.dot(H,xtrue[:,tobs[t]])+random[:]  
#print 'y[0,0.2] is', y[0,0.2]
#print 'y[0,20] is', y[0,20]
#print y[0,10000]
#print y.shape
print 'observations created'


# Creating the DM-dimensional time-delay vectors
# data vector Y
# Creating DM-dimensional map from physical space to delay embedding space - vector S
DM = 10
Y = np.zeros(DM)            
S = np.zeros(DM)            


# Defining initial x, dF/dx and J    
x = np.zeros([N+1,J+1])               ## verify: J+1???? why not J here? it's definition, not a loop# 
# Aplying randomness to x, not to be equal to xtrue ***** PJ - 19/03 *****
random = np.random.randn(N+1)
x[:,0] = xtrue[:,0] + random
dfdx = np.zeros([N+1,N+1]) 

# Creating initial condition for J (Jab = 1 when a=b)
Jac = np.zeros([N+1,N+1])                 
for i in range(N+1):
    Jac[i,i] = 1.
#print 'J is', Jac
#Jac0 = np.copy(Jac)                 #** ALT AT 26/03 - ADDING THIS VARIABLE TO ALL Jac CALCULATIONS**
dsdx = np.zeros([DM,N+1])  
dxds = np.zeros([N+1,DM]) 



## Running from 0 to 10
##for w in range(10):
##    random = np.zeros(N+1)
##    x[:,w+1] = mod.lorenz96(x[:,w],random,dt)  
    
##    dfdx = mod.df(x[:,w])         

##    Jac = Jac + dt*(np.dot(dfdx,Jac))   

##print 'x10 is', x[:,10]
        
# Main loop        
mcn = 0
summation = np.zeros(N+1)
run = 3000
count = 1
integ = DM*ns   
diff = np.zeros(DM)  
scount = 0
print 'Initial of initials dsdx', dsdx
##for t in range(10, run):
for t in range(run):
    if (t == tobs[mcn]): 
        ### Calculating dS/dx (from d to d+(DM-1)tau) advancing at each measurement time
        ##ldelay = integ + t
        print 'Initial x10 is', x[:,10]
        print 'Initial dsdx', dsdx
        ldelay = integ + mcn*ns
        for m in range(t,ldelay+1):
            if m == t:
                dsdx[scount,:] = Jac[0,:] 
                print 'Second dsdx', dsdx
                scount = scount + 1
            else:
                random = np.zeros(N+1)
                #random = np.random.randn(N+1)
                x[:,m+1] = mod.lorenz96(x[:,m],random,dt)  
                #print 'x is', x[:,m+1]
                dfdx = mod.df(x[:,m])
                Jac = Jac + dt*(np.dot(dfdx,Jac))   ## SEE IF Jac SHOULD BE FROM TIME 9 TO 10 ##
                newcount = (count+1)*ns
                if (m+1 == newcount):          
                    dsdx[scount,:] = Jac[0,:] 
                    #print 'dsdx', dsdx
                    scount = scount + 1
                    count = count + 1
        print 'dsdx', dsdx
        for d in range(DM): 
            td = t+d*ns
            #td = d*ns              
            Y[d] = y[:,td]                 
            S[d] = x[0,td]                
                
        dxds = np.linalg.pinv(dsdx)
        #check = np.allclose(dsdx, np.dot(dsdx, np.dot(dxds, dsdx)))
        #print 'check', check
        print 'dxds', dxds
        #count = 0
        scount = 0
        mcn = mcn + 1        
        count = mcn + 1
        # Full dynamics
        dif = Y - S
        #print 'dif is', dif
        summation = np.dot(dxds,dif)                 
        coup = g*summation
        #print 'coup is', coup
        x[:,t] = x[:,t] + coup
        print 'x10 is', x[:,10]
        random = np.zeros(N+1)
        x[:,t+1] = mod.lorenz96(x[:,t],random,dt) 
    
        for n in range(DM):                       
            diff[n] = (Y[n]-S[n])**2        
        SE = np.sqrt((1./DM)*np.sum(diff)) 
        print 'SE is', SE     

        plt.plot(t,SE,'b*')
        plt.hold(True)

        #print 'x11 is', x[:,11]
    else:
        random = np.zeros(N+1)
        x[:,t+1] = mod.lorenz96(x[:,t],random,dt)  
        #print 'x is', x[:,t+1]
        dfdx = mod.df(x[:,t])
        Jac = Jac + dt*(np.dot(dfdx,Jac))


# Main loop!
                   
#mcn = 0
##summation = 0
#run = 1000

#Jac = Jac9
#for z in range(10,run):   
    #print z 
#    random = np.zeros(N+1)
#    x[:,z+1] = mod.lorenz96(x[:,z],random,dt)   

#    dfdx = mod.df(x[:,z])  

#    Jac = Jac + dt*(np.dot(dfdx,Jac))                                  

#    if (z == tobs[mcn]): 
#        ldelay = integ + z
#        fdelay = ldelay - DM  
#        for m in range(fdelay+1,ldelay+1):
#            #random = np.random.randn(N+1)
#            # Taking off randomness from the model (deterministic in the article) ***** PJ - 19/03 **
#            random = np.zeros(N+1)            
#            x[:,m+1] = mod.lorenz96(x[:,m],random,dt)  
#            #print 'x at', z, 'is', x[:,m+1]
            
#            # Calculating the Jacobian from t+DM to (t+DM)+10 
#            dfdx = mod.df(x[:,m])  
#            #print 'dfdx', dfdx
#            Jac = Jac + dt*(np.dot(dfdx,Jac))
            #print 'Jac', Jac
        
        # Updating dS/dx
#        for d in range(DM-1):
#            dsdx[d,:] = dsdx[d+1,:]
#        dsdx[DM-1,:] = Jac[0,:] 
#        #print 'Updated dsdx', dsdx        

        # Calculating dx/dS
        #dxds = np.linalg.pinv(dsdx,rcond=1e-16)
#        dxds = np.linalg.pinv(dsdx)

        # Calculating Y and S from new t to t+DM
#        for d in range(DM): 
#            td = z+(d+1)*ns
            #td = d*ns              
            #i = d-1
#            Y[d] = y[:,td]                 
#            S[d] = x[0,td]
        #print 'Y is', Y
        #print 'S is', S
        #print y[0,10]
        #print x[0,10]   
        
        # Calculating the coupling term
#        summation = np.zeros(N+1)
#        dif = Y - S
        #print dif
#        summation = np.dot(dxds,dif)
        
#        coup = g*summation
        
#        x[:,z+DM] = x[:,z+DM] + coup
        #print 'X(8) pos is',  x[10,z]    

        # Monitoring the synchronisation error
        #S[0] = x[0,z]
#        for t in range(DM):                       
#            diff[t] = (Y[t]-S[t])**2        
            #print 'Diff', diff
#        SE = np.sqrt((1./DM)*np.sum(diff)) 
        #print 'Improved SE at', z, 'is', SE
        
#        mcn = mcn+1 
        
#        plt.plot(z,SE,'b*')
#        plt.hold(True)
   ## else: 
        #random = np.random.randn(N+1)
        # Taking off randomness from the model (deterministic in the article) ***** PJ - 19/03 **
   ##     random = np.zeros(N+1)
   ##     x[:,z+1] = mod.lorenz96(x[:,z],random,dt)      
   ##     dfdx = mod.df(x[:,z+1])  
   ##     Jac = Jac + dt*(np.dot(dfdx,Jac))
    
        
    
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

# X until the time before 1st measurement
#Plot samples, truth, observations for SIR
if (plotTraj == 1):
    plt.figure(figsize=(12, 10)).suptitle('Trajectories')
    for i in range(N/3):
        plt.subplot(np.ceil(N/8.0),2,i+1)
        #nn=round(np.random.rand(1)*N)
        plt.plot(x[i,:],'g',label='X')
#plt.plot(np.linspace(0,J,J+1),Usir[nn,i,:],'b',label='Posterior samples')
#        #samples
        plt.hold(True)
#    samples = min(int(Nsir),1000)
#        for b in range(samples):
#            nn=round(np.random.rand(1)*(Nsir-1))
#            plt.plot(np.linspace(0,J,J+1),Usir[nn,i,:],'b')
         #truth
        plt.plot(xtrue[i,:],'b-',linewidth=2.0,label='truth')
        plt.hold(True)
        #observations
        #if i in observed_vars:
        #    plt.errorbar(tobs,y,fmt='x',mec='g',ecolor='g',linewidth=2.0,label='observations')
        #for t in range(NM):
        #    tobs[t] = (t+1)*ns
        plt.plot(y[0,:],'r.',label='obs')    ### create y with only obs times(no zeros) to plot correctly ###
    plt.xlabel('time steps')
    plt.ylabel('x['+str(i)+']')
#        plt.xlim([0,J])
#        if (i == 0):
#            legend = plt.legend(loc='upper left',markerscale=0.4,numpoints=1)
#            for label in legend.get_texts():
#                label.set_fontsize('small')
#            for label in legend.get_lines():
#                label.set_linewidth(0.5)
#    plt.hold(False)
#    plt.subplots_adjust(wspace=0.7,hspace=0.3)
#    plt.savefig('sir_traj.eps',format='eps',bbox_inches='tight')
    plt.show()


