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
max_pinv_rank = M-2


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

### No noise for y 
y = np.dot(h,xtrue) 

#y = np.dot(h,xtrue) + np.random.uniform(0,0.02,N+1)-0.01   #which gives the variance of 0.0001 
#R = np.zeros([M, M])
#for i in range(M):
#    R[i,i] = 0.0001


#################### Initialising matrices #############################
Y = np.zeros([1,M])            
S = np.zeros([1,M]) 
diff = np.zeros([1,M])  

P1HT = np.zeros([D,M]) 
HPHT_KS = np.zeros([M,M]) 
HPHT = np.zeros([1,M]) 

xx = np.zeros([D,1])      
xxx = np.zeros([D,nTau]) 

Jac = np.zeros([D,D])    

for i in range(D):
    Jac[i,i] = 1.

Jac0 = np.copy(Jac)   

run = 300


################### Main loop ##########################################
for n in range(1,run+1):
    #t = (n-1)*dt

    S[:,0] = np.dot(h,x[:,n])   # this 0 term should be (0:L) in case of more obs
    Y[:,0] = y[:,n]             # this 0 term should be (0:L) in case of more obs
    #print 'S', S   
    #print 'Y', Y

    P1HT[:,0:L] = np.transpose(h)
    HPHT_KS[0,0] = h[:,0]
    #F1 = np.dot(np.transpose(Jac0),Jac0)
    #HPHT_KS[1,0] = F1[0,0]

    xx = x[:,n]
    #xxx[:,0] = xx 
    ###xx = xx.reshape(D,1)
    Jac = Jac0
    P = {}
    P['00'] = Jac0
    #########xxlast = xx

    #idxs = 1
    
    for t in range(M):
    #for t in range(1,M):

        for s in range(1,M):
            idxs = s
            #####xxx[:,0] = xxlast
            ###########xxx[:,0] = xx

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
                    #Jac4_old = Jac4

                    #########################
                    Jac2 = np.transpose(Jac2)
                    #########################

                    #xx = xxx[:,0] 
                    #for i in range(1,int(nTau)+1):
                    ##########Jac calculation with Runge-Kutta4 scheme##############
            
                    ################# Jacs to construct P #######################
                    ##############xx = xxx[:,s-1]
                    #print 'xx at', i, 'th cycle is', xx, 'for m =', m

                    # Calculating all elements in the upper part of the diagonal#
                    Jacsize = D**2

                    Jacv2 = Jac2.reshape(Jacsize)       
                    Jacvec2 = Jacv2.reshape(Jacsize,1)  
            
                    Jac2 = mod.rk4_J3(Jacvec2,D,xx,dt)  

                    # Calculating all elements in the lower part of the diagonal#
                    Jacv3 = Jac3.reshape(Jacsize)       
                    Jacvec3 = Jacv3.reshape(Jacsize,1)  
          
                    Jac3 = mod.rk4_J3(Jacvec3,D,xx,dt) 

                    # Calculating all elements in the diagonal#  
                    if m == 1:
                        ########if i == 1:
                        ########    Jac4 = Jac3                     
                        #########   Jacv4 = Jac4.reshape(Jacsize)       
                        ########Jacvec4 = Jacv4.reshape(Jacsize,1) 
                    
                        ########Jac4 = mod.rk4_J3(Jacvec4,D,xx,dt) 

                        #########################
                        Jac4 = np.transpose(Jac2)
                        #########################

                        #Jacv4 = Jac3.reshape(Jacsize)       
                        Jacv4 = Jac4.reshape(Jacsize) 

                        Jacvec4 = Jacv4.reshape(Jacsize,1)  
                
                        Jac4 = mod.rk4_J3(Jacvec4,D,xx,dt)
                        #Jac4 = mod.rk4_J4(Jacvec4,D,xx,dt)   ### it runs another module to multiply dkdx with Jold!
                        #Jac4 = mod.rk4_J5(Jacvec4,D,xx,dt)
                        
                    ###Jac4 = mod.rk4_J3(Jacvec4,D,xx,dt)
                    ###Jacv4 = Jac4.reshape(Jacsize)       
                    ###Jacvec4 = Jacv4.reshape(Jacsize,1) 
                    ###Jac4 = mod.rk4_J3(Jacvec4,D,xx,dt)
                    ########## Unperturbed inside-loop Lorenz runs################   
                    ##########if m == 1:         
                        ########random = np.zeros(D)
                        ########xx = mod.lorenz96(xx,random,dt) 
                    #if i < nTau:
                        ##############xxx[:,s] = xx
                    #if i == 10:
                    #    xxlast = xx
                        #print 'xxlast at m=',m,'is', xxlast
                    #xxlast = xx
                    ### At the end of each cycle (10 step mini-loop) we get Jac1i constructed ###
                    #print 'Jac1i', Jac
        
                    P[id1] = Jac2    
                    P[id3] = Jac3
                    #####P[id1] = np.transpose(Jac3)   
                    P[id5] = Jac4  
                    #print 'P', P

                    ################# Constructing HPHT_KS matrix #################
                    ###(uses only the first elements of the resulting matrices)####
                    F1 = P[id1]            
                    F1T = np.transpose(F1)
                    HPHT_KS[ii,idxs] = F1T[0,0]
                    #####HPHT_KS[ii,idxs] = F1[0,0]                
    
                    F2 = P[id3] 
                    HPHT_KS[idxs,ii] = F2[0,0]
                
                    if m == 1:
                        F3 = P[id5]            
                        #######################################            
                        ##########F3T = np.transpose(F3)
                        ##########HPHT_KS[idxs,idxs] = F3T[0,0]
                        #######################################
                        HPHT_KS[idxs,idxs] = F3[0,0]
        
                    #print 'HPHT_KS', HPHT_KS

                    #################### Constructing P1HT matrix #################
                    col = F1[:,0]
                    #col = F2[:,0]
                    P1HT[:,idxs] = col
                    #print 'P1HT', P1HT

                if m == (M-1):         
                    random = np.zeros(D)
                    xx = mod.lorenz96(xx,random,dt) 
        #################### Constructing P1HT matrix #################
        ###col1 = P['09']
        ###col = col1[:,0]
        
        ###P1HT[:,t] = col
      
        ###HPHT[:,t] = HPHT_KS[0,9]
            
        ###for i in range(1,int(nTau)+1):
            ###random = np.zeros(D)
            ###xx = mod.lorenz96(xx,random,dt) 
        
        random = np.zeros(D)
        xx = mod.lorenz96(xx,random,dt) 

        ########### Constructing S and Y vectors #######################   
        S[:,t] = np.dot(h,xx)
        #print 'xx for S at m=', m, 'is', xxlast
        Y[:,t] = y[:,n+(t*nTau)]   
        ####Y[:,t] = y[:,n+t]         
        #Y[:,idxs] = y[:,s*nTau] 

        #print 'S', S   
        #print 'Y', Y
    
        #print 'P1HT', P1HT[0,:]
        #print 'HPHT_KS', HPHT_KS[0,:]
        #HPHT_KS = HPHT_KS + R
        #print 'HPHT_KS After', HPHT_KS

    ########## Calculating the equivalent for KS structure##############
    #### Calculating the inverse of HPHT through SVD ####
    U, G, V = mod.svd(HPHT_KS)          # considering R=0
    print 'G', G
    #### Modifying G to use the max_pinv_rank ###########
    mask = np.ones(len(G)) 
    for k in range(len(G)):
        if G[k] >= pinv_tol:
            mask[k] = 1
        else:
            mask[k] = 0
        
    rr = min(max_pinv_rank,sum(mask)) 

    g = G[:rr]**(-1) 
    #print 'g', g
    Ginv = np.zeros((M, M))
    Ginv[:rr, :rr] = np.diag(g)
    #print 'Ginv', Ginv   

    ############## Calculating the inverse ##############
    HPHTinv1 = np.dot((np.transpose(V[:,:])),(np.transpose(Ginv)))   
    
    HPHTinv = np.dot(HPHTinv1,(np.transpose(U[:,:])))  
    
    #print 'HPHTinv', HPHTinv

    #HPHTi = np.linalg.pinv(HPHT_KS)
    #print 'HPHTi', HPHTi
    
    ###HPHTi = np.linalg.pinv(HPHT)

    ##### Multiplying the whole term with (y - hx) ######
    #dx1 = np.dot(P1HT,HPHTi)
    dx1 = np.dot(P1HT,HPHTinv)    
    #print 'dx1', dx1 

    ddd = Y[:,0] - S[:,0]
    #print 'diff', diff
    for i in range(M):
       diff[:,i] = ddd  
    #print diff

    dys = Y - S
    #print 'Y-S', dys

    dx = np.dot(dx1,np.transpose((Y-S)))
    #dx = np.dot(dx1,np.transpose(Y))
    #dx = np.dot(dx1,np.transpose(diff))
    #dx = np.dot(dx1,np.transpose(ddd))
    #dx = np.dot(dx1,(Y-S))

    #dx2 = np.dot(dx1,np.transpose((Y-S)))
    #dx2 = np.dot(dx1,np.transpose(diff))
    #dx2 = np.dot(dx1,np.transpose(Y))
    #dx = np.dot(K,dx2)
        
    dx = dx.reshape(D)  
    #print 'x[:,n]', x[:,n]
    #print 'dx', dx
    ############ Running the coupled dynamics ###########
    random = np.zeros(D)
    x[:,n+1] = mod.rk4_end(x[:,n],dx,dt)        
    #x[:,n+1] = x[:,n] + dx

    #print 'xtrue[:,n+1]', xtrue[:,n+1]
    #print 'x[:,n+1]', x[:,n+1]

    ########################## Calculating SE ############################
    dd = np.zeros([D,1])
    dd[:,0] = xtrue[:,n+1] - x[:,n+1]
    SE = np.sqrt(np.mean(np.square(dd)))            
    ##print '*************************************'
    print 'SE for', n, 'is', SE
    ##print '*************************************'


    ######### Plotting SE and other variables ###########
    #plt.figure(figsize=(12, 10)).suptitle('Synchronisation Error')
    plt.figure(2).suptitle('Synchronisation Error for D=20, M=10, r='+str(r)+', K='+str(K[0,0])+', max_pinv_rank= '+str(max_pinv_rank)+'')
    plt.plot(n+1,SE,'b*') 
    plt.yscale('log')
    plt.hold(True)
    
    
plt.show()

######################## Plotting variables ###############################
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
    

