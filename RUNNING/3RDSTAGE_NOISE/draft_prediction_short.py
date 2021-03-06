#from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
#import functions as m
import model as mod


N = 2300
dt = 0.01    #original value=0.01
fc=4000



D = 20 
F=8.17

M = 10
tau= 0.1
nTau = tau/dt
print 'D=', D, 'variables and M=', M ,'time-delays'


r=18 #for x[:,0] = xtrue[:,0], both for 20 variables
#r=37 #for original code (for M = 10)
#r=44  #for RK4 and 0.0005 uniform noise (for M = 10)
#r=39   #for RK4 and 0.0005 uniform noise (for M = 12)

np.random.seed(r)  



observed_vars = range(1)    
L = len(observed_vars) 
h = np.zeros([L,D])       
for i in range(L):
    h[i,observed_vars[i]] = 1.0   



K = 4.e1*np.diag(np.ones([D]))      # also testing: 2.e1, 5.e1, 1.e2
Ks = 1.e0*np.diag(np.ones([L*M]))  

pinv_tol =  (np.finfo(float).eps)#*max((M,D))#apparently same results as only 2.2204e-16
max_pinv_rank = 7



xtrue = np.zeros([D,fc+1])
xtrue[:,0] = np.random.rand(D)  #Changed to randn! It runned for both 10 and 20 variables
print 'xtrue[:,0]', xtrue[:,0]
dx0 = np.random.rand(D)-0.5     #Changed to randn! It runned for both 10 and 20 variables
##dx0 = np.random.rand(D)



x = np.zeros([D,fc+1])      
x[:,0] = xtrue[:,0] + dx0
#x[:,0] = xtrue[:,0]
print 'x[:,0]', x[:,0]

nTD = N + (M-1)*nTau
#t = np.zeros([1,nTD])
datat = np.dot(dt,list(xrange(N+1)))


for j in range(fc):      # try nTD
    force = np.zeros(D)  
    #force = np.random.rand(D)-0.5  # For only rand for 10 or 20 variables it overflows at time 2!!!!                              
    xtrue[:,j+1] = mod.lorenz96(xtrue[:,j],force,dt)  
xtrue[:,1] = xtrue[:,0] # try to sort python problem for 0 beginning index 
x[:,1] = x[:,0]         # try to sort python problem for 0 beginning index 
print 'truth created'



y = np.zeros([L,N]) 
#### No noise for y (ok for seed=37)
#y = np.dot(h,xtrue[:,:N]) 
#print 'y', y.shape

#### Good noise values for y (for seed=37)
#y = np.dot(h,xtrue[:,:N]) + np.random.uniform(0,1.2680e-04,N)-6.34e-05
#y = np.dot(h,xtrue[:,:N]) + np.random.normal(0,6.34e-05,N)                 # gaussian distribution
#y = np.dot(h,xtrue[:,:N]) + np.random.uniform(0,1.2680e-04,N)-9.34e-05  #(out of zero mean! so tiny it's almost zero)
#y = np.dot(h,xtrue[:,:N]) + np.random.uniform(0,1.8680e-04,N)-9.34e-05

#### Noise that runs perfect until time step 1500 (for seed=37) and until 4000 (for seed=44) and runs totally ok for dt=0.005!!!!
#y = np.dot(h,xtrue[:,:N]) + np.random.uniform(0,0.001,N)-0.0005
#y = np.dot(h,xtrue[:,:N]) + np.random.normal(0,0.0005,N)                   # gaussian distribution

#### Bad noise values for y (for seed=37)
#y = np.dot(h,xtrue[:,:N]) + np.random.rand(N)-0.5
y = np.dot(h,xtrue[:,:N]) + np.random.uniform(0,0.02,N)-0.01
#y = np.dot(h,xtrue[:,:N]) + np.random.uniform(0,0.01,N)-0.005
#y = np.dot(h,xtrue[:,:N]) + np.random.uniform(0,0.0022,N)-0.0011
#y = np.dot(h,xtrue[:,:N]) + np.random.uniform(0,0.002,N)-0.001

#print 'y', y
#print 'xtrue', xtrue



Y = np.zeros([1,M])            
S = np.zeros([1,M]) 
dsdx = np.zeros([M,D])         
dxds = np.zeros([D,M])  

xx = np.zeros([D,1])      
xtran = np.zeros([D,1]) 



Jac = np.zeros([D,D])                 
for i in range(D):
    Jac[i,i] = 1.

Jac0 = np.copy(Jac)   



run = 2200
#fcrun = run + 2000

for n in range(1,run+1):
    t = (n-1)*dt

    S[:,0] = np.dot(h,x[:,n])   # this 0 term should be (0:L) in case of more obs
    Y[:,0] = y[:,n]             # this 0 term should be (0:L) in case of more obs
    dsdx[0:L,:] = h
    #print 'dsdx', dsdx
    xx = x[:,n]
    ###xx = xx.reshape(D,1)
    Jac = Jac0
    #print 'Jac', Jac
    #print 'Initial xx is', xx

    for m in range(2,M+1):
        for i in range(1,int(nTau)+1):
            tt = t + dt*(i-1+(m-1)*nTau)
            
            #Jac calculation with Runge-Kutta4 scheme
            Jacsize = D**2
            Jacv = Jac.reshape(Jacsize)       # creates an array (Jacsize,)
            Jacvec = Jacv.reshape(Jacsize,1)  # creates an array (Jacsize,1)
            Jac = mod.rk4_J3(Jacvec,D,xx,dt)
            
                                 
            #Jac calculation with Euler scheme
            ###dfdx = mod.df(xx)
            ###Jac = Jac + dt*(np.dot(dfdx,Jac))
            #Jac = np.dot(dfdx,Jac)
            #Jac = dt*(np.dot(dfdx,Jac))           
            #print 'dfdx', dfdx
            #print 'Dfdx min:', np.min(dfdx),'max:', np.max(dfdx)

            random = np.zeros(D)
            #random = np.random.rand(D)-0.5
            xx = mod.lorenz96(xx,random,dt) 
            
            #print 'n=', n, 'xx at m', m, 'is', xx
            #print 'xx shape is', xx.shape
        
        #U, G, V = mod.svd(Jac)
        #print 'G', G.shape
        #print 'ln(G)', np.log(G)
        
        idxs = L*(m-1) #+ (L)    # attention to this (L)term, which should be (1:L) in case of more obs
        #print 'idxs', idxs        
        S[:,idxs] = np.dot(h,xx)
        #print 'S at m', m, 'is', S.shape
        #idy = n+(m-1)*nTau
        Y[:,idxs] = y[:,n+(m-1)*nTau]   # attention to y(0,...), which should increase in case of more obs
        dsdx[idxs,:] = np.dot(h,Jac)
    #print 'dsdx', dsdx

    ####dxds = np.linalg.pinv(dsdx,rcond=pinv_tol)    
    #dxds = dxds.round(decimals=4)     # Applied this as it was appearing in matlab code (1st row 1 0 0 0...)
    
    U, G, V = mod.svd(dsdx)
    #print 'U', U.shape
    #print 'G', G                       # All positive values, for good or bad runs. 
    #print 'V', V.shape
    #print 'ln(G)', np.log(G)
    mask = np.ones(len(G)) 
    for k in range(len(G)):
        #mask = np.ones(len(G))        
        if G[k] >= pinv_tol:
            mask[k] = 1
        else:
            mask[k] = 0
        #print 'mask', mask
    r = min(max_pinv_rank,sum(mask)) 
    #print 'r is', r
    g = G[:r]**(-1) 
    #print 'g', g  
    Ginv = np.zeros((M, D))
    Ginv[:r, :r] = np.diag(g)
    #print 'Ginv', Ginv 
    ###Ginv = np.diag(Ginv)
    #print 'Ginv2', Ginv    
    dxds1 = np.dot((np.transpose(V[:,:])),(np.transpose(Ginv)))   
    #print 'dxds1', dxds1.shape
    ########dxds = np.dot(dxds1,(np.transpose(U[:,:r])))  
    dxds = np.dot(dxds1,(np.transpose(U[:,:])))  
    #print 'dxds', dxds 
    #print 'Y', Y
    #print 'S', S
    
    dx1 = np.dot(K,dxds)
    dx2 = np.dot(Ks,np.transpose((Y-S)))
    dx = np.dot(dx1,dx2)
    dx = dx.reshape(D)
    #print 'dx', dx
    #print 'x[:,n]shape', x[:,n].shape
    
    random = np.zeros(D)
    #xtran2 = x[:,n] + dx                                           #n3m4
    #x[:,n+1] = mod.lorenz96(xtran2,random,dt) 
   
    ##xtran2 = x[:,n] + dt*dx                                       #4)n2500m4 (peaks) #8)n1m7             
    ##x[:,n+1] = mod.lorenz96(xtran2,random,dt) 

    ###x[:,n+1] = x[:,n] + dt*(x[:,n] + dx)                         #n193m4

    ####x[:,n+1] = x[:,n] + dt*dx                                   #n787m4

    #####x[:,n+1] = dt*(x[:,n] + dx)                                ##4)n2500m4 (damping) #8)n1m7
    
    ######x[:,n+1] = mod.lorenz96(x[:,n],random,dt) + dx              #n4m2

    #######x[:,n+1] = mod.lorenz96(x[:,n],random,dt) + dt*dx          #4)n2500m4 (peaks) #8)n1m7            

    ########xtran2 = mod.l95(x[:,n],dt) + dx                          #n4m3
    ########x[:,n+1] = mod.lorenz96(xtran2,random,dt)

    ########xtran2 = mod.l95(x[:,n],dt) + dt*dx                        ##4)n2500m4 (damping) #8)n1m7       
    ########x[:,n+1] = mod.lorenz96(xtran2,random,dt)

    ##########xtran2 = mod.l95(x[:,n],dt) + dx                          #n4m4
    ##########x[:,n+1] = mod.rk4(xtran2,dt)

    ###########xtran2 = mod.l95(x[:,n],dt) + dt*dx                       ##4)n2500m4 (damping) #8)n1m7    
    ###########x[:,n+1] = mod.rk4(xtran2,dt)
    
    x[:,n+1] = mod.rk4_end(x[:,n],dx,dt) #+ dx0                               #4)n4m2 #4)n2500m4 (peaks) for dt*dx in rk4

    #############ddx = dx*dt                                             #4)n2500m4 (peaks) #8)n1m7 
    #############x[:,n+1] = mod.rk4_end(x[:,n],ddx,dt) 

    ##############x[:,n+1] = dt*(mod.lorenz96(x[:,n],random,dt) + dx)     ##4)n2500m4 (damping) #8)n1m7 

    ###############xtran2 = dt*(x[:,n] + dx)                                             ##4)n2500m4 (damping)  
    ###############x[:,n+1] = mod.lorenz96(xtran2,random,dt) 

        
    #print 'x[:,n+1] at', n+1, 'is', x[:,n+1]

    ##if np.mod(n+1,10) == 1:
        ##SE = np.zeros([D,n+1])
        ##for d in range(n+1):
            ##SE[:,d] = xtrue[:,d] - x[:,d]
        #SE = xtrue(:,1:n+1) - x(:,1:n+1)
        ##SE = np.sqrt(np.mean(np.square(SE)))
        ##print 'SE at', n, 'is', SE       
        ##plt.plot(n+1,SE,'b*') 
        ##plt.yscale('log')
        ##plt.hold(True)
    dd = np.zeros([D,1])
    dd[:,0] = xtrue[:,n+1] - x[:,n+1]
    SE = np.sqrt(np.mean(np.square(dd)))            
    print 'SE for', n, 'is', SE
    plt.plot(n+1,SE,'b*') 
    plt.yscale('log')
    plt.hold(True)

plt.show()

random = np.zeros(D)
for d in range(run+1,fc):
    x[:,d+1] = mod.lorenz96(x[:,d],random,dt) 

##############################Plotting#########################################

plt.figure(figsize=(12, 10)).suptitle('Full sync - J reinitialized')
for i in range(D/3):
    plt.subplot(np.ceil(D/8.0),2,i+1)
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
    

