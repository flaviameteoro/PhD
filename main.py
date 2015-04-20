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
plotTruth = 1
plotTraj = 0
plotTrace = 0
plotEff = 0 #effective sample size

## Parameter settings
J = 100      # observation window
N = 20       # number of state variables
dt = 0.01    # deltat                                   ###### SEE IF THIS TIME STEP GUARANTEES STABILITY AS DT 0.025!! #############
#nudgeFac=3 #nudge factor b for the equivalent-weights particle filter
#obsgrid = 3 #number of observations at analysis time (1=all observed, 2=every other observed, 3=half state observed)
#ns = 10 #number of time steps between obs
#dx=1./N

# Choose random seed - With the seed reset,same numbers will appear every time 
r=1 
np.random.seed(r)

# Creating the truth run
xhulp = np.zeros([N,2000])
xtrue = np.zeros([N,J+1])
#xtrueII = np.zeros([N,J+1])
force = np.zeros(N)

#spin up
F=8.17
xhulp[:,0] = F                                          ##### CHANGE HERE IF WE WANT TO VARY FORCING PARAMETERS AS TABLE 1????#######
pert = 0.05
pospert = np.ceil(N/2.0)-1
xhulp[pospert,0] = F+pert
spinup=1999
for j in range(spinup):
    force = np.zeros(N)
    xhulp[:,j+1] = mod.lorenz96(xhulp[:,j],force,dt)    ####### this line returns 1 column for the 20 variables at each loop ########
    #print 'xhulp', xhulp
xtrue[:,0] = xhulp[:,spinup]
#xtrueII[:,0] = xtrue[:,0]
for j in range(J):
    #random = np.random.randn(N)
    #force = np.dot(scov_model,random)
    #xtrueII[:,j+1] = force
    force = np.zeros(N)                                 
    xtrue[:,j+1] = mod.lorenz96(xtrue[:,j],force,dt)    ####### THE WHOLE TRUE MATRIX IS CREATED ####################################
    #print 'xtrue', xtrue
print 'truth created'
print xtrue.shape

# Data y - Only the first state variable is selected to serve as the data 
y = xtrue[0,:]
print y

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



