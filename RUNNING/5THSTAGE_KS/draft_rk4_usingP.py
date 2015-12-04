#from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
#import functions as m
import model as mod
#from draft_ksmy1to10newFplots_main import P
import Pcalc as pcal


############### Define variables #############################
N = 10000
Obs = 100
dt = 0.01    #original value=0.01

D = 20 
F=8.17

M = 8
tau= 0.1
nTau = tau/dt
print 'D=', D, 'variables and M=', M ,'time-delays'

#for 20 variables:

#r=18 #for x[:,0] = xtrue[:,0]
r=37 #for original code 
#r=44  #for RK4 and 0.0005 uniform noise (for M = 10)
#r=39   #for RK4 and 0.0005 uniform noise (for M = 12)

np.random.seed(r)  

##################### Initialise arrays ###########################
observed_vars = range(1)    
L = len(observed_vars) 
h = np.zeros([L,D])       
for i in range(L):
    h[i,observed_vars[i]] = 1.0   

K = 1.e1*np.diag(np.ones([D]))      
Ks = 1.e0*np.diag(np.ones([L*M]))  

pinv_tol =  (np.finfo(float).eps)
max_pinv_rank = M

################### Construct xtrue and y ########################
xtrue = np.zeros([D,N+1])
xtrue[:,0] = np.random.rand(D)  
print 'xtrue[:,0]', xtrue[:,0]
dx0 = np.random.rand(D)-0.5     


x = np.zeros([D,N+1])      
x[:,0] = xtrue[:,0] + dx0
print 'x[:,0]', x[:,0]

nTD = N + (M-1)*nTau

datat = np.dot(dt,list(xrange(N+1)))
for j in range(N):      # try nTD
    force = np.zeros(D)  
    
    xtrue[:,j+1] = mod.lorenz96(xtrue[:,j],force,dt)  
xtrue[:,1] = xtrue[:,0] # try to sort python problem for 0 beginning index 
x[:,1] = x[:,0]         # try to sort python problem for 0 beginning index 
print 'truth created'

y = np.zeros([L,N+1]) 

### No noise for y (ok for seed=37)
y = np.dot(h,xtrue) 

######################### Initialising other arrays ######################
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

################# Main loop ##################################
run = 500

P = pcal.Pmatrix(x[:,1],Jac0)
#print 'Initial P is', P

#indi = str(M-1)+str(M-1)
#Jac0 = P[indi]

#Jac0 = P['99']
#Jac0 = P['1919']
Jac0 = P['1919']

for n in range(1,run+1):
    #t = (n-1)*dt

    S[:,0] = np.dot(h,x[:,n])   
    Y[:,0] = y[:,n]             
    dsdx[0:L,:] = h
    
    xx = x[:,n]
    
    Jac = Jac0
    

    for m in range(2,M+1):
        for i in range(1,int(nTau)+1):
            #tt = t + dt*(i-1+(m-1)*nTau)
            
            #Jac calculation with Runge-Kutta4 scheme
            Jacsize = D**2
            Jacv = Jac.reshape(Jacsize)       
            Jacvec = Jacv.reshape(Jacsize,1)  
            Jac = mod.rk4_J3(Jacvec,D,xx,dt)
            
            
            random = np.zeros(D)
            xx = mod.lorenz96(xx,random,dt) 
            
            
        idxs = L*(m-1) 
               
        S[:,idxs] = np.dot(h,xx)
        Y[:,idxs] = y[:,n+(m-1)*nTau]   
        
        dsdx[idxs,:] = np.dot(h,Jac)
    
    ################ Calculate Pseudo-Inverse ###########################
    U, G, V = mod.svd(dsdx)
    
    mask = np.ones(len(G)) 
    for k in range(len(G)):
              
        if G[k] >= pinv_tol:
            mask[k] = 1
        else:
            mask[k] = 0
        
    r = min(max_pinv_rank,sum(mask)) 
    
    g = G[:r]**(-1) 
    
    Ginv = np.zeros((M, D))
    Ginv[:r, :r] = np.diag(g)
        
    dxds1 = np.dot((np.transpose(V[:,:])),(np.transpose(Ginv)))   
    
    dxds = np.dot(dxds1,(np.transpose(U[:,:])))  
        
    ########################## Calculate coupling term ###############   
    dx1 = np.dot(K,dxds)
    #print 'dx1', dx1
    dx2 = np.dot(Ks,np.transpose((Y-S)))
    dx = np.dot(dx1,dx2)
    dx = dx.reshape(D)
    
    ################## Evolve in time #################################    
    random = np.zeros(D)
    x[:,n+1] = mod.rk4_end(x[:,n],dx,dt) 

    #############Calculate new P matrix at each 10 time steps #########
    #if np.mod(n,10) == 0:
    #    P = pcal.Pmatrix(x[:,n+1],Jac0)
    #    print 'At n', n, 'new P is', P
    
    #####################Define Jac0 for next step #####################
    #if n < 10:    
    #    iii = str(n)+str(n)  
    #    print 'At n', n, 'iii is', iii
    #    Jac0 = P[iii]
        
    #if n >= 10:
    #    modu = int(np.ceil(n/10))
    #    nn = n-(modu*10)
    #    iii = str(nn)+str(nn)    
    #    print 'At n', n, 'iii is', iii
    #    Jac0 = P[iii] 
    
    ##################################PLots ############################
    dd = np.zeros([D,1])
    dd[:,0] = xtrue[:,n+1] - x[:,n+1]
    SE = np.sqrt(np.mean(np.square(dd)))            
    print 'SE for', n, 'is', SE
    plt.plot(n+1,SE,'b*') 
    plt.yscale('log')
    plt.hold(True)
    
plt.show()

#cmap=plt.get_cmap('RdBu')
#plt.figure(3).suptitle('P Matrix')
#for b in range(M):
#    for a in range(M):
#        c = M*b
#        plt.subplot(M,M,c+a+1)
#        idx = str(b)+str(a)
#        plt.imshow(P[idx],cmap=cmap)
#        #plt.colorbar()
#        plt.hold(True)
#        plt.xlabel('P['+str(idx)+']')
#        plt.hold(True)
#    plt.hold(True)
#plt.savefig('P.png')

cmap=plt.get_cmap('RdBu')
plt.figure(4).suptitle('P Matrix')
#    if n == 20:
        #for i in range(2):
        #plt.subplot(1,3,1)
plt.imshow(P['1919'],cmap=cmap)
plt.xlabel('P[1919]')
plt.colorbar()   
plt.savefig('P.png')
    
plt.figure(figsize=(12, 10)).suptitle('Synchronisation')
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
    

