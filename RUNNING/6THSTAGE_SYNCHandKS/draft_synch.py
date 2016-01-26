#from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
#import functions as m
import model as mod

#################### Initial settings ################################
N = 10000
Obs = 100
dt = 0.01    #original value=0.01

D = 20 
F=8.17

M = 10
tau= 0.1
nTau = tau/dt
print 'D=', D, 'variables and M=', M ,'time-delays'

###################### Seeding for 20 variables#######################

r=18 #for x[:,0] = xtrue[:,0]
#r=37 #for original code 
#r=44  #for RK4 and 0.0005 uniform noise (for M = 10)
#r=39   #for RK4 and 0.0005 uniform noise (for M = 12)

np.random.seed(r)  


#################### Constructing h (obs operator) ##################
observed_vars = range(1)    
L = len(observed_vars) 
h = np.zeros([L,D])       
for i in range(L):
    h[i,observed_vars[i]] = 1.0   


################### Setting coupling matrices ########################
K = 1.e1*np.diag(np.ones([D]))      # also testing: 2.e1, 5.e1, 1.e2
Ks = 1.e0*np.diag(np.ones([L*M]))  


######### Setting tolerance and maximum for rank calculations ########
pinv_tol =  (np.finfo(float).eps)#*max((M,D))#apparently same results as only 2.2204e-16
max_pinv_rank = M


################### Creating truth ###################################
xtrue = np.zeros([D,N+1])
#***xtrue[:,0] = np.random.rand(D)  #Changed to randn! It runned for both 10 and 20 variables
#print 'xtrue[:,0]', xtrue[:,0]

####### Start by spinning model in ###########
xtest = np.zeros([D,1001]) 
xtest[:,0]=np.random.rand(D)
for j in range(1000):
    force = np.zeros(D)
    xtest[:,j+1]=mod.lorenz96(xtest[:,j],force,dt)
         
xtrue[:,0] = xtest[:,1000]
print 'xtrue[:,0]', xtrue[:,0]

###### Plot xtrue to understand initial conditions influences ####
#plt.figure(1).suptitle('xtrue for seed='+str(r)+'')
#plt.plot(xtrue[:,0],'g-')
#plt.show()

dx0 = np.random.rand(D)-0.5     #Changed to randn! It runned for both 10 and 20 variables
##dx0 = np.random.rand(D)

x = np.zeros([D,N+1])      
x[:,0] = xtrue[:,0] + dx0
#x[:,0] = xtrue[:,0]
print 'x[:,0]', x[:,0]


nTD = N + (M-1)*nTau
#t = np.zeros([1,nTD])
datat = np.dot(dt,list(xrange(N+1)))
for j in range(N):      # try nTD
    force = np.zeros(D)  
    #force = np.random.rand(D)-0.5  # For only rand for 10 or 20 variables it overflows at time 2!!!!                              
    xtrue[:,j+1] = mod.lorenz96(xtrue[:,j],force,dt)  
xtrue[:,1] = xtrue[:,0] # try to sort python problem for 0 beginning index 
x[:,1] = x[:,0]         # try to sort python problem for 0 beginning index 
print 'truth created'


################### Creating the obs ##################################
y = np.zeros([L,N+1]) 
###### No noise for y (ok for seeds = 37,18 and 44 with SPINUP)
#y = np.dot(h,xtrue) 

###### Good noise values for y (for seed=37)
#y = np.dot(h,xtrue) + np.random.uniform(0,1.2680e-04,N+1)-6.34e-05
#y = np.dot(h,xtrue) + np.random.uniform(0,1.2680e-04,N+1)-9.34e-05  #(out of zero mean!)
#y = np.dot(h,xtrue) + np.random.uniform(0,1.8680e-04,N+1)-9.34e-05

###### Noise that ran perfect until time step 1500 (for seed=37), BUT after SPINUP-1000 it only runs until 650...
###### Ran totally ok for seed=44, BUT after SPINUP-1000 it runs until 8000...
###### Runs totally ok for seed=18, ALSO with SPINUP-1000!!!!!!!!!!!!!!!!!!!
y = np.dot(h,xtrue) + np.random.uniform(0,0.001,N+1)-0.0005
#y = np.dot(h,xtrue) + np.random.normal(0,0.0005,N+1)     

###### Bad noise values for y (for seed=37)
#y = np.dot(h,xtrue) + np.random.rand(N+1)-0.5
#y = np.dot(h,xtrue) + np.random.uniform(0,0.2,N+1)-0.1
#y = np.dot(h,xtrue) + np.random.uniform(0,0.02,N+1)-0.01
#y = np.dot(h,xtrue) + np.random.uniform(0,0.01,N+1)-0.005
###(for seed=18)
#y = np.dot(h,xtrue) + np.random.normal(0,0.01,N+1)  

###### Noise that runs perfect until time step 1800 and 2300 (for seed=18, K=40, max_rank=7) 
#y = np.dot(h,xtrue) + np.random.uniform(0,0.002,N+1)-0.001
#y = np.dot(h,xtrue) + np.random.uniform(0,0.02,N+1)-0.01
#y = np.dot(h,xtrue) + np.random.normal(0,0.001,N+1)  


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

run = 9900

oo = np.zeros([1,run+1])      #for observability calculation
svmaxvec = np.zeros([1,run+1]) 
svmaxvec2 = np.zeros([1,run+1]) 

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
            ##dxdt = mod.dxdt(xx,Jacvec,D,dt)
            ##xtran = mod.rk4(dxdt,dt)
            ##xx = xtran[0:D]
            ##print 'n=', n, 'xx at m', m, 'is', xx
            ##Jact = xtran[D:]
            ##Jac = Jact.reshape(D,D)
            
            ############dfdx = mod.df(xx)
            #print 'dfdx', dfdx
            #print 'Dfdx min:', np.min(dfdx),'max:', np.max(dfdx)
            
            random = np.zeros(D)
            #random = np.random.rand(D)-0.5
            xx = mod.lorenz96(xx,random,dt) 
            
            #Jac calculation with Euler scheme
            ############Jac = Jac + dt*(np.dot(dfdx,Jac))
            #Jac = np.dot(dfdx,Jac)
            #Jac = dt*(np.dot(dfdx,Jac))           
            #print 'xx at m', m, 'is', xx
            #dfdx = mod.df(xx)
            #print 'dfdx', dfdx
           
            #Jacsize = D**2
            #Jacv = Jac.reshape(Jacsize) 
            #Jacvec = Jacv.reshape(Jacsize,1)
            #Jac = mod.rk4_J3(Jacvec,D,dt)
            #f = dxdt[0:D]
            #########xx = mod.rk4(f,dt)
            #print 'Jac', Jac
            #print 'n=', n, 'xx at m', m, 'is', xx
            #print 'xx shape is', xx.shape

        idxs = L*(m-1) #+ (L)    # attention to this (L)term, which should be (1:L) in case of more obs
        #print 'idxs', idxs        
        S[:,idxs] = np.dot(h,xx)
        #print 'S at m', m, 'is', S.shape
        #idy = n+(m-1)*nTau
        Y[:,idxs] = y[:,n+(m-1)*nTau]   # attention to y(0,...), which should increase in case of more obs
        dsdx[idxs,:] = np.dot(h,Jac)
    #print 'dsdx', dsdx

    ########dxds = np.linalg.pinv(dsdx,rcond=pinv_tol)    
    #dxds = dxds.round(decimals=4)     # Applied this as it was appearing in matlab code (1st row 1 0 0 0...)
    
    U, G, V = mod.svd(dsdx)
    #print 'U', U.shape
    #print 'G', G                       # All positive values, for good or bad runs. 
    #print 'V', V.shape
    #print 'ln(G)', np.log(G)

    svmin = np.min(G)
    #print 'Smallest sing value:', svmin              #no influence until now...(around e-03)
    svmax = np.max(G) 
    #print 'Largest sing value:', svmax   
    svmaxvec[:,n] = svmax
    svmaxvec2[:,n] = G[1]
    difmax = abs(svmaxvec[:,n] - svmaxvec[:,n-1])
    difmax2 = abs(svmaxvec2[:,n] - svmaxvec2[:,n-1])
    ###difmax = svmaxvec[:,n] - svmaxvec[:,n-1]
    #print 'Difmax=', difmax


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
    
    difsv = svmax - svmin    
    ratioobs = svmin/svmax
    condnumber = svmax/svmin
    #print 'Condition number is', condnumber
    oo[:,n] = (ratioobs)**2
    obin = np.sum(oo)   
    #print 'observability', observ                   #no influence until now...(between e-05 and e-04)


    dx1 = np.dot(K,dxds)
    #print 'dx1', dx1
    dx2 = np.dot(Ks,np.transpose((Y-S)))
    dx = np.dot(dx1,dx2)
    dx = dx.reshape(D)
    #print 'dx', dx
    #print 'x[:,n]shape', x[:,n].shape
    
    ddd = Y-S
    #print 'ddd', ddd

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
    
    ##plt.plot(n+1,svmin,'c<') 
    ##plt.hold(True)

    ##plt.plot(n+1,svmax,'r>') 
    ##plt.hold(True)

    #plt.plot(n+1,ratioobs,'yo') 
    #plt.hold(True)
  
    ##plt.plot(n+1,condnumber,'m.') 
    ##plt.hold(True)

    #plt.plot(n+1,obin,'m<') 
    #plt.hold(True)

    #plt.plot(n+1,difsv,'mo') 
    #plt.hold(True)

    #plt.plot(n+1,difmax,'yo') 
    #plt.hold(True)

    #plt.plot(n+1,difmax2,'mo') 
    #plt.hold(True)


#obin_gama = (1./float(n))*np.sum(oo)               #see article Parlitz, Schumann-Bischoff and Luther, 2015
#print 'Observability Index is', obin_gama

plt.show()

plt.figure(figsize=(12, 10)).suptitle('Variables for D='+str(D)+', M='+str(M)+', r='+str(r)+', K='+str(K[0,0])+', max_pinv_rank= '+str(max_pinv_rank)+'')
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
    

