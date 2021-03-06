#from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
#import functions as m
import model as mod


N = 10000
Obs = 100
dt = 0.01    #original value=0.01

D = 20 
F=8.17

M = 10
tau= 0.1
nTau = tau/dt
print 'D=', D, 'variables and M=', M ,'time-delays'

#for 20 variables:

r=18 #for x[:,0] = xtrue[:,0]
#r=37 #for original code 
#r=44  #for RK4 and 0.0005 uniform noise (for M = 10)
#r=39   #for RK4 and 0.0005 uniform noise (for M = 12)

np.random.seed(r)  

observed_vars = range(1)    
L = len(observed_vars) 
h = np.zeros([L,D])       
for i in range(L):
    h[i,observed_vars[i]] = 1.0   

K = 40.e0*np.diag(np.ones([D]))      # also testing: 2.e1, 5.e1, 1.e2
Ks = 1.e0*np.diag(np.ones([L*M]))  

pinv_tol =  (np.finfo(float).eps)#*max((M,D))#apparently same results as only 2.2204e-16
max_pinv_rank = M

xtrue = np.zeros([D,N+1])
xtrue[:,0] = np.random.rand(D)  #Changed to randn! It runned for both 10 and 20 variables
print 'xtrue[:,0]', xtrue[:,0]
dx0 = np.random.rand(D)-0.5     #Changed to randn! It runned for both 10 and 20 variables
##dx0 = np.random.rand(D)
#print 'dx0', dx0

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

y = np.zeros([L,N+1]) 
### No noise for y (ok for seed=37)
#y = np.dot(h,xtrue) 

### Good noise values for y (for seed=37)
### (noises are centralized in zero)
#y = np.dot(h,xtrue) + np.random.uniform(0,1.2680e-04,N+1)-6.34e-05
#y = np.dot(h,xtrue) + np.random.uniform(0,1.2680e-04,N+1)-9.34e-05  #(out of zero mean!)
#y = np.dot(h,xtrue) + np.random.uniform(0,1.8680e-04,N+1)-9.34e-05

### Noise that runs perfect until time step 1500 (for seed=37) 
# and runs totally ok for seed=44!!!!
#y = np.dot(h,xtrue) + np.random.uniform(0,0.001,N+1)-0.0005
#y = np.dot(h,xtrue) + np.random.normal(0,0.0005,N+1)     

### Bad noise values for y (for seed=37)
#y = np.dot(h,xtrue) + np.random.rand(N+1)-0.5
#y = np.dot(h,xtrue) + np.random.uniform(0,0.2,N+1)-0.1
#y = np.dot(h,xtrue) + np.random.uniform(0,0.02,N+1)-0.01
#y = np.dot(h,xtrue) + np.random.uniform(0,0.01,N+1)-0.005
###(for seed=18)
#y = np.dot(h,xtrue) + np.random.normal(0,0.01,N+1)  

### Noise that runs perfect until time step 200 (for seed=37) 
#y = np.dot(h,xtrue) + np.random.normal(0,0.1,N+1)

### Noise that runs perfect until time step 1800 and 2300 (for seed=18, K=40, max_rank=7) 
#y = np.dot(h,xtrue) + np.random.uniform(0,0.002,N+1)-0.001
y = np.dot(h,xtrue) + np.random.uniform(0,0.02,N+1)-0.01
#y = np.dot(h,xtrue) + np.random.normal(0,0.001,N+1)  


#print 'y', y[0,5]
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

run = 2000

oo = np.zeros([1,run+1])      #for observability calculation
cond = np.zeros([1,run+1])  
difcond = 0

svmaxvec = np.zeros([1,run+1]) 
svmaxvec2 = np.zeros([1,run+1]) 

dlyaini = x[:,1] - xtrue[:,1]
#print 'dlyaini', dlyaini

for n in range(1,run+1):
    #####if n == 2300:
        #####K = 45.e0*np.diag(np.ones([D])) 
        #####max_pinv_rank = 6
        ##print 'K', K[0,0]
    ##if n == 2500:
        ##K = 55.e0*np.diag(np.ones([D])) 
        ##print 'K', K[0,0]
    #####if n == 2550:
        #K = 40.e0*np.diag(np.ones([D])) 
        #####max_pinv_rank = 7
    #####if n == 2700:
        #K = 45.e0*np.diag(np.ones([D])) 
        #####max_pinv_rank = 6
        ##print 'K', K[0,0]
      

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
    #if n >= 2000:
    #print 'G', G.shape                       # All positive values, for good or bad runs. 
    #print 'G2', G[1]
    #print 'V', V.shape
    #print 'ln(G)', np.log(G)

    
    svmin = np.min(G)
    #print 'Smallest sing value:', svmin              #no influence until now...(around e-03)
    svmin2 = G[M-2]
    #print '2nd smallest sing value:', svmin2
    svmax = np.max(G) 
    #print 'Largest sing value:', svmax   
    svmaxvec[:,n] = svmax
    svmaxvec2[:,n] = G[1]
    difmax = abs(svmaxvec[:,n] - svmaxvec[:,n-1])
    difmax2 = abs(svmaxvec2[:,n] - svmaxvec2[:,n-1])
    ###difmax = svmaxvec[:,n] - svmaxvec[:,n-1]
    #print 'Difmax=', difmax
    #print 'Difmax2=', difmax2

    ##### Frobenius #######
    fro = np.sqrt(sum(G**2))
    #print 'fro', fro
    #######################if n >= 2:
        ##elif difmax >= 0.5 <1:
        ########################if difmax >= 0.5:
            ########################if svmaxvec[:,n] > svmaxvec[:,n-1]:
                #########################svmaxvec[:,n] = svmaxvec[:,n-1]+0.5
            ########################else:
                #########################svmaxvec[:,n] = svmaxvec[:,n-1]-0.5

        #G[0] = svmaxvec[:,n-1]
        ################################G[0] = svmaxvec[:,n]
    #print 'New largest sing value:', svmaxvec[:,n]  

    ######## Condition number and observability calculation #######
    difsv = svmax - svmin    
    ratioobs = svmin/svmax
    condnumber = svmax/svmin
    cond[:,n] = condnumber
    
    if np.mod(n,100) == 0:
        difcond = cond[:,n] 
    else:
        difcond = 0 
    
    oo[:,n] = (ratioobs)**2
    obin = np.sum(oo)       



    ############### Testing rank max #########################

    #test_rank = True
    linf_norm2 = 51
    #######linf_norm = 11
    max_pinv_rank = M

    #def start():
    while linf_norm2 >= 50:
    #######while linf_norm >= 10:  
  
        #k=0
        #if test_rank = False:
        #    max_pinv_rank = max_pinv_rank + 1 
        
        print 'max_pinv_rank', max_pinv_rank        
        
        
        mask = np.ones(len(G)) 
        for k in range(len(G)):
            if G[k] >= pinv_tol:
                mask[k] = 1
            else:
                mask[k] = 0
    
        rr = min(max_pinv_rank,sum(mask)) 

        g = G[:rr]**(-1) 
    
        Ginv = np.zeros((M, D))
        Ginv[:rr, :rr] = np.diag(g)
    
        dxds1 = np.dot((np.transpose(V[:,:])),(np.transpose(Ginv)))   
    
        dxds = np.dot(dxds1,(np.transpose(U[:,:])))  
    
        

        dx1 = np.dot(K,dxds)

        dx2 = np.dot(Ks,np.transpose((Y-S)))
        dx = np.dot(dx1,dx2)
        dx = dx.reshape(D)

    
        ###### Calculating the L-2 and L-inf norms (trying to choose an ideal rank) ########
        deltax = np.dot(dxds,np.transpose(Y-S))
    
        for i in range(D):
            deltax2 = deltax[i]**2
        deltax_norm = np.sqrt(np.sum(deltax2))
    

        for i in range(D):
            dx2 = dx[i]**2
        dx_norm = np.sqrt(np.sum(dx2))
    
        random = np.zeros(D)
        x_unpert = mod.lorenz96(x[:,n],random,dt) 
    
        for i in range(D):
            x_unpert2 = x_unpert[i]**2
        xunpert_norm = np.sqrt(np.sum(x_unpert2))
    

        l_inf = np.zeros(D)
        for i in range(D):
            l_inf[i] = abs(deltax[i]/x_unpert[i])
        linf_norm = max(l_inf)
        

        l_inf2 = np.zeros(D)
        for i in range(D):
            #l_inf2[i] = abs(float(dx[i])/float(x_unpert[i]))
            l_inf2[i] = abs(dx[i]/x_unpert[i])
        linf_norm2 = max(l_inf2)
        #linf_norm2_prev = linf_norm2

        print 'linf_norm2', linf_norm2
        ######print 'linf_norm', linf_norm

        #if linf_norm2 <= 10:
        #    test_rank = False
        max_pinv_rank = max_pinv_rank - 1     
        
        #if max_pinv_rank > M+1:
        #    break

        if max_pinv_rank == 7:
            break

    #print 'Final max_pinv_rank', max_pinv_rank  
    #while test_rank:
    #    start() 

    ############################################################################################
    max_pinv_rank  = max_pinv_rank+1
    ##if max_pinv_rank  == 0:
        ##max_pinv_rank  = 1
    print 'Ideal max_pinv_rank', max_pinv_rank  

    mask = np.ones(len(G)) 
    for k in range(len(G)):
        if G[k] >= pinv_tol:
            mask[k] = 1
        else:
            mask[k] = 0
    
    rr = min(max_pinv_rank,sum(mask)) 

    g = G[:rr]**(-1) 
    
    Ginv = np.zeros((M, D))
    Ginv[:rr, :rr] = np.diag(g)
    
    dxds1 = np.dot((np.transpose(V[:,:])),(np.transpose(Ginv)))   
    
    dxds = np.dot(dxds1,(np.transpose(U[:,:])))  
    
        
    dx1 = np.dot(K,dxds)

    dx2 = np.dot(Ks,np.transpose((Y-S)))
    dx = np.dot(dx1,dx2)
    dx = dx.reshape(D)



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

    dlya = x[:,n+1] - xtrue[:,n+1]
    #print 'dlya', dlya
    dl = abs(dlya/dlyaini)   
    #print 'dl', dl
    lya = (1/float(n))*(np.log(dl))
    
    lyaposit = 0
    for i in range(D):
        if lya[i] > 0:
            lyaposit = lyaposit + 1

    print 'Lyapositive', lyaposit
################# Adjusting synchronisation ######################

    #if lyaposit <= 10:
    #    K = 40.e0*np.diag(np.ones([D])) 
    #    max_pinv_rank = 7

    #if 5 < lyaposit <= 10:
    #    K = 45.e0*np.diag(np.ones([D])) 
    #    max_pinv_rank = 6

    #if 10 < lyaposit <= 15:
    #    K = 45.e0*np.diag(np.ones([D])) 
    #    max_pinv_rank = 6        

    #if lyaposit > 15:
        #K = 50.e0*np.diag(np.ones([D])) 
    #    max_pinv_rank = 7


    #print 'Lyapunov exponent', lya
    #print 'Max lyapunov exponent', max(lya)  #it is not the max value that matters, it is the amount of negative lyapunov exponents!!
    
    #plt.figure(figsize=(12, 10)).suptitle('Lyapunov Exponents')
    #if [any(item) in lya >0 
    plt.figure(1).suptitle('Lyapunov Exponents for D=20, M=10, r='+str(r)+', K='+str(K[0,0])+', max_pinv_rank= '+str(max_pinv_rank)+'')
    plt.axhline(y=0, xmin=0, xmax=run, linewidth=1, color = 'm')
    ##for i in range(D):   
    ###for i in range(D):                   # to plot all variables!!
        ###plt.subplot(D/4,4,i+1)           # to plot all variables!!
    for i in range(D/3):
        plt.subplot(np.ceil(D/8.0),2,i+1)  
        if lya[i] >0:
            plt.plot(n+1,lya[i],'y.',linewidth=2.0,label='truth')
            plt.yscale('log')
            plt.ylabel('x['+str(i)+']')
            plt.hold(True)
            
        
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
    ##print '*************************************'
    print 'SE for', n, 'is', SE
    print '*************************************'

      

    #plt.figure(figsize=(12, 10)).suptitle('Synchronisation Error')
    plt.figure(2).suptitle('Synchronisation Error for D=20, M=10, r='+str(r)+', K='+str(K[0,0])+', max_pinv_rank= '+str(max_pinv_rank)+'')
    plt.plot(n+1,SE,'b*') 
    plt.yscale('log')
    plt.hold(True)
    
    #plt.plot(n+1,svmin,'c<') 
    #plt.hold(True)

    #plt.plot(n+1,svmax,'r>') 
    #plt.hold(True)

    #plt.plot(n+1,ratioobs,'yo') 
    #plt.hold(True)
  
    plt.plot(n+1,condnumber,'y*') 
    plt.yscale('log')
    plt.hold(True)

    #plt.plot(n+1,difcond,'b.') 
    #plt.yscale('log')
    #plt.hold(True)

    #plt.plot(n+1,obin,'m<') 
    #plt.hold(True)

    #plt.plot(n+1,difmax,'yo') 
    #plt.hold(True)

    #plt.plot(n+1,difmax2,'mo') 
    #plt.hold(True)

    #plt.plot(n+1,fro,'y*') 
    #plt.hold(True)
    
    #plt.plot(n+1,fro_rank,'g*') 
    #plt.hold(True)

    #plt.plot(n+1,G[7],'y.') 
    #plt.hold(True)
    
    #plt.plot(n+1,G[8],'g.') 
    #plt.hold(True)

    #plt.plot(n+1,lyaposit,'y.') 
    #plt.hold(True)

    #plot(n+1,linf_norm2,'m*') 
    #plt.yscale('log')
    #plt.hold(True)
        
    plt.plot(n+1,max_pinv_rank,'mo') 
    plt.hold(True)

obin_gama = (1./float(n))*np.sum(oo)               #see article Parlitz, Schumann-Bischoff and Luther, 2015
#print 'Observability Index is', obin_gama

plt.show()

plt.figure(figsize=(12, 10)).suptitle('Variables for D=20, M=10, r='+str(r)+', K='+str(K[0,0])+', max_pinv_rank= '+str(max_pinv_rank)+'')
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
    

