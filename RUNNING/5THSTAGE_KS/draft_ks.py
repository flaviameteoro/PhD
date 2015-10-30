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

##################### Seeding for 20 variables#######################
#r=18 #for x[:,0] = xtrue[:,0]
r=37 #for original code 
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
K1 = 11.e0*np.diag(np.ones([D]))

######### Setting tolerance and maximum for rank calculations ########
pinv_tol =  (np.finfo(float).eps)#*max((M,D))#apparently same results as only 2.2204e-16
max_pinv_rank = M


################### Creating truth ###################################
xtrue = np.zeros([D,N+1])
xtrue[:,0] = np.random.rand(D)  #Changed to randn! It runned for both 10 and 20 variables
print 'xtrue[:,0]', xtrue[:,0]
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

### No noise for y (ok for seed=37)
y = np.dot(h,xtrue) 

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

### Noise that runs perfect using HHT + R until time step 200 (for seed=37) and 500 (for seed=18, K=40, max_rank=7) 
#y = np.dot(h,xtrue) + np.random.normal(0,0.1,N+1)  #which gives the variance of 0.01  

#R = np.zeros([M, M])
#for i in range(M):
#    R[i,i] = 0.01

### Noise that runs perfect until time step 1800 and 2300 (for seed=18, K=40, max_rank=7) 
#y = np.dot(h,xtrue) + np.random.uniform(0,0.002,N+1)-0.001

### Noise that runs perfect using HHT + R:
### until time step 2600 (for seed=18, K=10, max_rank=D)
### until time step 800  (for seed=18, K=40, max_rank=D)
### until time step 2000 (for seed=18, K=40, max_rank=9) 
#y = np.dot(h,xtrue) + np.random.uniform(0,0.02,N+1)-0.01   #which gives the variance of 0.0001 

R = np.zeros([M, M])
for i in range(M):
    R[i,i] = 0.0001

#y = np.dot(h,xtrue) + np.random.normal(0,0.001,N+1)  


#################### Initialising matrices #############################
Y = np.zeros([1,M])            
S = np.zeros([1,M]) 
dsdx = np.zeros([M,D])         
dxds = np.zeros([D,M]) 

P1HT = np.zeros([D,M]) 
HPHT_KS = np.zeros([M,M]) 

gama = np.zeros([M,M])   
gamainv = np.zeros([M,M])   
#P = (0.01)*np.diag(np.ones([D]))  
P = np.zeros([D, D])

xx = np.zeros([D,1])      
xtran = np.zeros([D,1]) 

Jac = np.zeros([D,D])    
JacB = np.zeros([D,D])   
diag = np.zeros([D,D])              

for i in range(D):
    Jac[i,i] = 1.

Jac0 = np.copy(Jac)   

run = 3000

oo = np.zeros([1,run+1])      #for observability calculation
svmaxvec = np.zeros([1,run+1]) 
svmaxvec2 = np.zeros([1,run+1]) 

dlyaini = x[:,1] - xtrue[:,1]
#print 'dlyaini', dlyaini



################### Main loop ##########################################
for n in range(1,run+1):
    t = (n-1)*dt

    S[:,0] = np.dot(h,x[:,n])   # this 0 term should be (0:L) in case of more obs
    Y[:,0] = y[:,n]             # this 0 term should be (0:L) in case of more obs
    dsdx[0:L,:] = h
    P1HT[:,0:L] = np.transpose(h)
    HPHT_KS[0,0] = h[:,0]
    #F1 = np.dot(np.transpose(Jac0),Jac0)
    #HPHT_KS[1,0] = F1[0,0]

    xx = x[:,n]
    ###xx = xx.reshape(D,1)
    Jac = Jac0
    P = {}
    P['00'] = Jac0
    
    #idxs = 1

    for s in range(1,M):
        idxs = s
        for m in range(1,M):
            #idxs = L*(m-1)        # attention to this (L)term, should be (1:L) if more obs
            ii = idxs - m
            iid = idxs + 1

            id1 = str(ii)+str(iid-1)
            id2 = str(ii)+str(iid-2)
        
            id3 = str(iid-1)+str(ii)
            id4 = str(iid-2)+str(ii)

            id5 = str(iid-1)+str(iid-1)
            id6 = str(iid-2)+str(iid-2)
        
            if ii >= 0:
                Jac2 = P[id2] 
                Jac3 = P[id4]
                Jac4 = P[id6]
                Jac4_old = Jac4

                for i in range(1,int(nTau)+1):
                    ##########Jac calculation with Runge-Kutta4 scheme##############

                    ######################## Jac for P1HT ##########################
                    #Jacsize = D**2
                    #Jacv = Jac.reshape(Jacsize)       # creates an array (Jacsize,)
                    #Jacvec = Jacv.reshape(Jacsize,1)  # creates an array (Jacsize,1)
            
                    #Jac = mod.rk4_J3(Jacvec,D,xx,dt)  
                    #Jac = JacF
            
                    ############### Jacs for HPHT_KS and P1HT ######################
                    Jacsize = D**2

                    Jacv2 = Jac2.reshape(Jacsize)       
                    Jacvec2 = Jacv2.reshape(Jacsize,1)  
            
                    Jac2 = mod.rk4_J3(Jacvec2,D,xx,dt)  


                    Jacv3 = Jac3.reshape(Jacsize)       
                    Jacvec3 = Jacv3.reshape(Jacsize,1)  
            
                    Jac3 = mod.rk4_J3(Jacvec3,D,xx,dt) 

                
                    Jacv4 = Jac4.reshape(Jacsize)       
                    Jacvec4 = Jacv4.reshape(Jacsize,1)  
                
                    Jac4 = mod.rk4_J4(Jacvec4,D,xx,dt)   ### it runs another module to multiply dkdx with Jold!

                    ########## Unperturbed inside-loop Lorenz runs##################            
                    random = np.zeros(D)
                    #random = np.random.rand(D)-0.5
                    xx = mod.lorenz96(xx,random,dt) 

                ### At the end of each cycle (10 step mini-loop) we get Jac1i constructed ###
                #print 'Jac1i', Jac
        
                P[id1] = Jac2    
                P[id3] = Jac3   
                P[id5] = Jac4  
                ###print 'P', P

                ################# Constructing HPHT_KS matrix #################
                ###(uses only the first elements of the resulting matrices)####
                F1 = P[id1]            
                F1T = np.transpose(F1)
                HPHT_KS[ii,idxs] = F1T[0,0]

                F2 = P[id3] 
                HPHT_KS[idxs,ii] = F2[0,0]

                F3 = P[id5]            
                F3T = np.transpose(F3)
                HPHT_KS[idxs,idxs] = F3T[0,0]
        
                ###print 'HPHT_KS', HPHT_KS

                #################### Constructing P1HT matrix #################
                #test = np.dot(h,Jac)
                #print 'test', test.shape
                #print 'test', test
                #JacT = np.transpose(Jac)
                #col = np.dot(JacT,np.transpose(h))
                col = F1[:,0]
                #print 'test2', test2.shape
                #print 'test2', test2
                ###col = col.reshape(D)
                #print 'test2', test2 
                P1HT[:,idxs] = col
                print 'P1HT', P1HT 


            ########### Constructing S and Y vectors #######################   
            S[:,idxs] = np.dot(h,xx)
        
            Y[:,idxs] = y[:,n+(m-1)*nTau]   # attention to y(0,...), which should increase in case of more obs

            #dsdx[idxs,:] = np.dot(h,Jac)
        
            #idxs = L*(m-1)


   
    #### Calculating dxds as a pseudoinverse using python function######
    dxds = np.linalg.pinv(dsdx,rcond=pinv_tol)    
    #dxds = dxds.round(decimals=4)     # Applied this as it was appearing in matlab code (1st row 1 0 0 0...)
    
    
    ##### Calculating the supposed equivalent for KS structure##########


    #(Use this inverse instead of SVD, as gama is a diagonal!!!!!!!)
    ###dsdxt = np.transpose(dsdx)
    ###for i in range(M):
        ###gama[i,i] = np.dot(dsdx[i,:],dsdxt[:,i])
    #####gama = np.dot(dsdx,dsdxt)
    #print 'Gama1', gama
    ###for i in range(M):
        ###gamainv[i,i] = gama[i,i]**(-1) 
    #print 'Gamainv', gamainv
        
    #####HHT = np.dot(dsdx,(np.transpose(dsdx)))
    ####HHT = np.dot(dsdx,(np.transpose(dsdx)))+R
    
    #####dxdst = np.linalg.pinv(np.transpose(dsdx),rcond=pinv_tol)    
    #####P = np.dot(dxds,np.dot(R,dxdst))
    #P = 10*(np.dot(dxds,np.dot(R,dxdst)))
    ####P1 = (np.dot(dxds,np.dot(R,dxdst)))
    ####P2 = np.diag(P1)
    ####P = np.zeros([D, D])
    ####P = np.diag(P2)
    #print 'P1', P1
    #print 'P', P
    ##P = (0.0001)*P
    
    #####newinv = (11.*R)
    #print 'R', R
    #print 'newinv', newinv

    #####for i in range(len(R)):
        #####newinv[i,i]=(newinv[i,i])**(-1)  
    #print 'newinv', newinv  

    ####HKHT = np.dot(dsdx,np.dot(K,(np.transpose(dsdx))))
    ####HPHT = np.dot(dsdx,np.dot(P,(np.transpose(dsdx))))

    ####HKHT = np.dot(dsdx,np.dot(K,(np.transpose(dsdx))))+R
    ####HPHT = np.dot(dsdx,np.dot(P,(np.transpose(dsdx))))+R

    #### Calculating the inverse of HPHT (KF structure) through SVD ####
    ####U, G, V = mod.svd(HHT)     # considering P=1
    ####U, G, V = mod.svd(HKHT)
    ####U, G, V = mod.svd(HPHT)
    U, G, V = mod.svd(gama)

    ################### Last 3 singular values #########################
    svmin = np.min(G)
    
    svmin2 = G[M-2]
    
    #svmin3 = G[M-3]


    ################## Modifying G to use the max_pinv_rank ############
    mask = np.ones(len(G)) 
    for k in range(len(G)):
        if G[k] >= pinv_tol:
            mask[k] = 1
        else:
            mask[k] = 0
        
    rr = min(max_pinv_rank,sum(mask)) 

    g = G[:rr]**(-1) 
    
    Ginv = np.zeros((M, M))
    Ginv[:rr, :rr] = np.diag(g)
        

    ################# Calculating the inverse ###########################
    HHTinv1 = np.dot((np.transpose(V[:,:])),(np.transpose(Ginv)))   
    
    HHTinv = np.dot(HHTinv1,(np.transpose(U[:,:])))  
    
    #print 'HHTinv', HHTinv
    
    ################# Calculating the KF equivalent of dxds #############
    ####dxds = np.dot(np.transpose(dsdx),HHTinv)
    ####dxds = np.dot(np.transpose(dsdx),gamainv)
    ####dxds = np.dot(Jac,np.dot(np.transpose(dsdx),HHTinv))
    ####dxds = np.dot(K,np.dot(np.transpose(dsdx),HHTinv))    
    ####dxds = np.dot(P,np.dot(np.transpose(dsdx),HHTinv))    
    ####dxds = np.dot(P,np.dot(np.transpose(dsdx),newinv))

    ################# Multiplying it with (y - hx) ######################
    dx1 = np.dot(K,dxds)                        # considering an extra K 
    dx2 = np.dot(Ks,np.transpose((Y-S)))
    dx = np.dot(dx1,dx2)
    
    ##dx = np.dot(dxds, np.transpose((Y-S)))  # (without extra k - does not work...) 

    dx = dx.reshape(D)  
    #print 'dx', dx
    #print 'x[:,n]shape', x[:,n].shape
    

    ################ Running the coupled dynamics ######################
    random = np.zeros(D)
    x[:,n+1] = mod.rk4_end(x[:,n],dx,dt) #+ dx0           


    ################ Calculating Lyapunov exponent #####################
    dlya = x[:,n+1] - xtrue[:,n+1]
    #print 'dlya', dlya
    dl = abs(dlya/dlyaini)   
    #print 'dl', dl
    lya = (1/float(n))*(np.log(dl))
    ##print 'Lyapunov exponent', lya
    

    ################ Plotting Lyapunov exponent #####################
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
            
        
    ################ Calculating SE #################################
    dd = np.zeros([D,1])
    dd[:,0] = xtrue[:,n+1] - x[:,n+1]
    SE = np.sqrt(np.mean(np.square(dd)))            
    ##print '*************************************'
    print 'SE for', n, 'is', SE
    ##print '*************************************'


    ################## Plotting SE and other variables ##############
    #plt.figure(figsize=(12, 10)).suptitle('Synchronisation Error')
    plt.figure(2).suptitle('Synchronisation Error for D=20, M=10, r='+str(r)+', K='+str(K[0,0])+', max_pinv_rank= '+str(max_pinv_rank)+'')
    plt.plot(n+1,SE,'b*') 
    plt.yscale('log')
    plt.hold(True)
    
    plt.plot(n+1,svmin,'yo') 
    plt.hold(True)

    plt.plot(n+1,svmin2,'go') 
    plt.hold(True)

    #plt.plot(n+1,svmin3,'mo') 
    #plt.hold(True)

    #plt.plot(n+1,svmax,'r>') 
    #plt.hold(True)

    #plt.plot(n+1,ratioobs,'yo') 
    #plt.hold(True)
  
    #plt.plot(n+1,condnumber,'m.') 
    #plt.hold(True)

    #plt.plot(n+1,obin,'m<') 
    #plt.hold(True)

    #plt.plot(n+1,difmax,'yo') 
    #plt.hold(True)

    #plt.plot(n+1,difmax2,'mo') 
    #plt.hold(True)



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
    

