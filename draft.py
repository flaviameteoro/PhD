import numpy as np
import matplotlib.pyplot as plt
import functions as m
import model as mod

N = 10000
Obs = 100
dt = 0.01

D = 20 
F=8.17

M = 10
tau= 0.1
nTau = tau/dt

observed_vars = range(1)    
L = len(observed_vars) 
h = np.zeros([L,D])       
for i in range(L):
    h[i,observed_vars[i]] = 1.0   

K = 1.e1*np.diag(np.ones([D]))  
Ks = 1.e0*np.diag(np.ones([L*M]))  

pinv_tol = 2.2204e-16
max_pinv_rank = D

xtrue = np.zeros([D,N+1])
xtrue[:,0] = np.random.rand(D)
dx0 = np.random.rand(D)-0.5

x = np.zeros([D,N+1])      
x[:,0] = xtrue[:,0] + dx0

nTD = N + (M-1)*nTau
#t = np.zeros([1,nTD])
datat = np.dot(dt,list(xrange(N+1)))
for j in range(N):      # try nTD
    force = np.zeros(D)                                 
    xtrue[:,j+1] = mod.lorenz96(xtrue[:,j],force,dt)  
xtrue[:,1] = xtrue[:,0] # try to sort python problem for 0 beginning index 
x[:,1] = x[:,0]         # try to sort python problem for 0 beginning index 
print 'truth created'

y = np.zeros([L,N+1])  
y = np.dot(h,xtrue)

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

run = 1000

for n in range(1,run+1):
    t = (n-1)*dt

    S[:,0] = np.dot(h,x[:,0])# attention to this 0 term, which should be (0:L) in case of more obs
    Y[:,0] = y[:,n]          # attention to this 0 term, which should be (0:L) in case of more obs
    dsdx[0:L,:] = h
    print 'dsdx', dsdx
    xx = x[:,n-1]
    Jac = Jac0
    print 'Jac', Jac
    
    for m in range(2,M+1):
        for i in range(1,int(nTau)+1):
            tt = t + dt*(i-1+(m-1)*nTau)
            #xx = my_ode(ff,tt,xx);
            Jacsize = D**2
            Jacv = Jac.reshape(Jacsize) 
            Jacvec = Jacv.reshape(Jacsize,1)
            dxdt = mod.dxdt(xx,Jacvec,D,dt)
            xtran = mod.rk4(dxdt,dt)
            xx = xtran[0:D]
            print 'xx', xx
            Jact = xtran[D:]
            Jac = Jact.reshape(D,D)
            ###random = np.zeros(N)
            ###xx = mod.lorenz96(xx,random,dt) 
            ###dfdx = mod.df(xx)
            ###Jac = Jac + dt*(np.dot(dfdx,Jac)) 
        idxs = L*(m-1) #+ (L)    # attention to this (L)term, which should be (1:L) in case of more obs
        S[:,idxs] = np.dot(h,xx)
        #idy = n+(m-1)*nTau
        Y[:,idxs] = y[:,n+(m-1)*nTau]# attention to y(0,...), which should increase in case of more obs
        dsdx[idxs,:] = np.dot(h,Jac)

    U, G, V = mod.svd(dsdx)
    for k in range(len(G)):
        mask = np.ones(len(G))        
        if G[k] >= pinv_tol:
            mask[k] = 1
        else:
            mask[k] = 0
    r = min(max_pinv_rank,sum(mask)) 
    Ginv = G[:r]**(-1) 
    Ginv = np.diag(Ginv)
    dxds = np.dot(V[:,:r],Ginv)   
    dxds = np.dot(dxds,(np.transpose(U[:,:r])))  
    
    dx1 = np.dot(K,dxds)
    dx2 = np.dot(Ks,np.transpose((Y-S)))
    dx = np.dot(dx1,dx2)

    random = np.zeros(N)
    xtran2 = x[:,n] + dx
    x[:,n+1] = mod.lorenz96(xtran2,random,dt) 

    if np.mod(n+1,10) == 1:
        SE = np.zeros([D,n+1])
        for d in range(n+1):
            SE[:,d] = xtrue[:,d] - x[:,d]
        #SE = xtrue(:,1:n+1) - x(:,1:n+1)
        SE = np.sqrt(np.mean(np.square(SE)))
                
        plt.plot(d,SE,'b*') 
        plt.yscale('log')
        plt.hold(True)

plt.draw()
    

