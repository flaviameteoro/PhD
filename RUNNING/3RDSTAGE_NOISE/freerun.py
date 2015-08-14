#from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
#import functions as m
import model as mod


N = 1000
Obs = 100
dt = 0.01    #original value=0.01
fc=2000



D = 20 
F=8.17

M = 10
tau= 0.1
nTau = tau/dt
print 'D=', D, 'variables and M=', M ,'time-delays'


#r=18 #for x[:,0] = xtrue[:,0], both for 20 variables
r=37 #for original code (for M = 10)
#r=44  #for RK4 and 0.0005 uniform noise (for M = 10)
#r=39   #for RK4 and 0.0005 uniform noise (for M = 12)

np.random.seed(r)  



observed_vars = range(1)    
L = len(observed_vars) 
h = np.zeros([L,D])       
for i in range(L):
    h[i,observed_vars[i]] = 1.0   


xtrue = np.zeros([D,N+1])
xtrue[:,0] = np.random.rand(D)  #Changed to randn! It runned for both 10 and 20 variables
print 'xtrue[:,0]', xtrue[:,0]
dx0 = np.random.rand(D)-0.5     #Changed to randn! It runned for both 10 and 20 variables
##dx0 = np.random.rand(D)



x = np.zeros([D,N+1])   
z = np.zeros([D,N+1])      
w = np.zeros([D,N+1])      
k = np.zeros([D,N+1])      
x[:,0] = xtrue[:,0] #+ dx0
z[:,0] = x[:,0]
w[:,0] = x[:,0]
k[:,0] = x[:,0]

print 'x[:,0]', x[:,0]

nTD = N + (M-1)*nTau
#t = np.zeros([1,nTD])
datat = np.dot(dt,list(xrange(N+1)))


for j in range(N):      # try nTD
    force = np.zeros(D)  
    #force = np.random.rand(D)-0.5  # For only rand for 10 or 20 variables it overflows at time 2!!!!                              
    xtrue[:,j+1] = mod.lorenz96(xtrue[:,j],force,dt)  
#xtrue[:,1] = xtrue[:,0] # try to sort python problem for 0 beginning index 
#x[:,1] = x[:,0]         # try to sort python problem for 0 beginning index 
print 'truth created'



y = np.zeros([L,N]) 
#### No noise for y (ok for seed=37)
y = np.dot(h,xtrue[:,:N]) 
#print 'y', y.shape

#### Good noise values for y (for seed=37)
#y = np.dot(h,xtrue[:,:N]) + np.random.uniform(0,1.2680e-04,N)-6.34e-05
#y = np.dot(h,xtrue[:,:N]) + np.random.normal(0,6.34e-05,N)                 # gaussian distribution
#y = np.dot(h,xtrue[:,:N]) + np.random.uniform(0,1.2680e-04,N)-9.34e-05  #(out of zero mean! so tiny it's almost zero)
#y = np.dot(h,xtrue[:,:N]) + np.random.uniform(0,1.8680e-04,N)-9.34e-05

#### Noise that runs perfect until time step 1500 (for seed=37) and until 4000 (for seed=44) and runs totally ok for dt=0.005!!!!
y = np.dot(h,xtrue[:,:N]) + np.random.uniform(0,0.001,N)-0.0005
#y = np.dot(h,xtrue[:,:N]) + np.random.normal(0,0.0005,N)                   # gaussian distribution

#### Bad noise values for y (for seed=37)
#y = np.dot(h,xtrue[:,:N]) + np.random.rand(N)-0.5
#y = np.dot(h,xtrue[:,:N]) + np.random.uniform(0,0.04,N)-0.02
#y = np.dot(h,xtrue[:,:N]) + np.random.uniform(0,0.01,N)-0.005
#y = np.dot(h,xtrue[:,:N]) + np.random.uniform(0,0.0022,N)-0.0011
#y = np.dot(h,xtrue[:,:N]) + np.random.uniform(0,0.002,N)-0.001

#print 'y', y
#print 'xtrue', xtrue


run = 1000
#fcrun = run + 2000

random = np.zeros(D)
random2 = np.random.uniform(0,1.2680e-04,D)-6.34e-05

random3 = np.random.uniform(0,0.001,D)-0.0005
random4 = np.random.normal(0,0.0001,D)  

random5 = np.random.uniform(0,0.002,D)-0.001
random6 = np.random.normal(0,0.001,D)  

random7 = np.random.uniform(0,0.02,D)-0.01
random8 = np.random.normal(0,0.01,D)  

random9 = np.random.uniform(0,0.2,D)-0.1
random10 = np.random.normal(0,0.1,D)  


for n in range(run):
    x[:,n+1] = mod.lorenz96(x[:,n],random4,dt) 
    z[:,n+1] = mod.lorenz96(z[:,n],random6,dt) 
    w[:,n+1] = mod.lorenz96(w[:,n],random8,dt) 
    #k[:,n+1] = mod.lorenz96(k[:,n],random8,dt) 

#for d in range(run+1,fc):
#    x[:,d+1] = mod.lorenz96(x[:,d],random,dt) 

##############################Plotting#########################################

plt.figure(figsize=(12, 10)).suptitle('Lorenz free runs - Noise influence - Gaussian distributions')
for i in range(D/3):
    plt.subplot(np.ceil(D/8.0),2,i+1)
    plt.plot(x[i,:],'g',label='X')
    plt.hold(True)
    plt.plot(z[i,:],'m',label='Z')
    plt.hold(True)
    plt.plot(w[i,:],'y',label='W')
    plt.hold(True)
    #plt.plot(k[i,:],'c',label='k')
    #plt.hold(True)
    plt.plot(xtrue[i,:],'b-',linewidth=2.0,label='truth')
    plt.hold(True)

    plt.legend(['e-04', 'e-03', 'e-02', 'truth'], ncol=4, prop={'size':10}, loc='upper left')
    
plt.show()
    

