import numpy as np
#import matplotlib.pyplot as plt
#import functions as m
import model as mod


J = 10000      # number of steps
Obs = 100     # obs window
N = 20       # number of state variables
dt = 0.01    # deltat                            ## SEE IF THIS TIME STEP GUARANTEES STABILITY AS DT 0.025!!## 
tau=0.1      # constant time delay (=10dt)
obsgrid = 4  #number of observations at analysis time (1=all observed, 2=every other observed, 3=half state observed, 4=1 variable)
ns = 10       #number of time steps between obs


# Creating the truth run
xhulp = np.zeros([N,2000])
xtrue = np.zeros([N,J+1])
#xtrueII = np.zeros([N,J+1])
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
    xhulp[:,j+1] = mod.lorenz96(xhulp[:,j],force,dt)   ### this line returns 1 column for the 20 variables at each loop ####
    #print 'xhulp', xhulp
xtrue[:,0] = xhulp[:,spinup]
#xtrueII[:,0] = xtrue[:,0]
for j in range(J):
    #random = np.random.randn(N)
    #force = np.dot(scov_model,random)
    #xtrueII[:,j+1] = force
    force = np.zeros(N)                                 
    xtrue[:,j+1] = mod.lorenz96(xtrue[:,j],force,dt)   
    #print 'xtrue', xtrue
print 'truth created'
print xtrue.shape
forc = np.zeros([1,J+1])
print forc.shape
forc[:,:] = F
print forc 
xtrue = np.append(xtrue,forc, axis=0)
print 'New xtrue', xtrue.shape
#print xtrue[0,1]

x = np.zeros([N+1,J+1])                         
x[:,0] = xtrue[:,0]

for t in range(10):
    random = np.random.randn(N+1)
    x[0,t+1] = mod.lorenz96(x[0,t],random,dt)    

print x
