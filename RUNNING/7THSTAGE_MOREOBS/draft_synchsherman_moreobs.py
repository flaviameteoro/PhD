#from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
#import functions as m
import model as mod
import time
start_time = time.clock()
#################### Initial settings ################################
N = 13000
Obs = 100
dt = 0.01    #original value=0.01
fc = 12500

D = 20 
F=8.17

M = 6
tau= 0.1
nTau = tau/dt
print 'D=', D, 'variables and M=', M ,'time-delays'

###################### Seeding for 20 variables#######################

#r=18 #for x[:,0] = xtrue[:,0]
r=37 #for original code 
#r=44  #for RK4 and 0.0005 uniform noise (for M = 10)
#r=39   #for RK4 and 0.0005 uniform noise (for M = 12)

np.random.seed(r)  


#################### Constructing h (obs operator) ##################
#observed_vars = range(1)    
observed_vars = range(5)    ######MORE OBS#########
L = len(observed_vars) 
h = np.zeros([L,D])       
for i in range(L):
#    h[i,observed_vars[i]] = 1.0  ######MORE OBS######### 
#    h[i,observed_vars[i]*2] = 1.0 ######MORE OBS######### 
    h[i,observed_vars[i]*(D/L)] = 1.0 ######MORE OBS######### for observing vars equally sparsed
print 'h', h



################### Setting coupling matrices ########################
K = 1.e1*np.diag(np.ones([D]))      # also testing: 2.e1, 5.e1, 1.e2
#print 'K', K

###Trying tridiagonal K### (tested with no difference from original)
#K = np.zeros([D,D])    
#it,jt = np.indices(K.shape)
#K[it==jt] = 1.e1
#K[it==jt-1] = 1.e-1
#K[it==jt+1] = 1.e-1
#print 'New K', K


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
###### No noise for y (with SPINUP, ok for seeds = 37,18,44 - rank M=10 and ok for r= 39 - rank 8!!)
#y = np.dot(h,xtrue) 
#sigma2 = 0

###### Good noise values for y (for seed=37)
#y = np.dot(h,xtrue) + np.random.uniform(0,1.2680e-04,N+1)-6.34e-05
#y = np.dot(h,xtrue) + np.random.uniform(0,1.2680e-04,N+1)-9.34e-05  #(out of zero mean!)
#y = np.dot(h,xtrue) + np.random.uniform(0,1.8680e-04,N+1)-9.34e-05

###### Noise that runs totally ok for seeds=37 and 44 (rank=9) (after SPINUP-1000) (until 1500 for seed=37 before)  
###### Runs totally ok for seed=18 (rank=9), with SPINUP-1000!!!!!!!!!!!!!!!!!!!
###### Runs until 8500 for seed=39 (rank=8)
#y = np.dot(h,xtrue) + np.random.uniform(0,0.001,N+1)-0.0005
###Std deviation: 1e-4###
#y = np.dot(h,xtrue) + np.random.normal(0,0.0005,N+1) 
#sigma2 = 0.00000025    

###### Bad noise values for y (for seed=37)
#y = np.dot(h,xtrue) + np.random.rand(N+1)-0.5
#y = np.dot(h,xtrue) + np.random.uniform(0,0.2,N+1)-0.1
#y = np.dot(h,xtrue) + np.random.uniform(0,0.02,N+1)-0.01
#y = np.dot(h,xtrue) + np.random.uniform(0,0.01,N+1)-0.005
###(for seed=18)
###Std deviation: 1e-2###
#y = np.dot(h,xtrue) + np.random.normal(0,0.01,N+1)  
#sigma2 = 0.0001

###Std deviation: 1e-1###
y = np.dot(h,xtrue) + np.random.normal(0,0.1,N+1)  
sigma2 = 0.01

###### Noise that runs perfect until time step 1800 and 2300 (for seed=18, K=40, max_rank=7) 
#y = np.dot(h,xtrue) + np.random.uniform(0,0.002,N+1)-0.001
#y = np.dot(h,xtrue) + np.random.uniform(0,0.02,N+1)-0.01
###Std deviation: 1e-3###
#y = np.dot(h,xtrue) + np.random.normal(0,0.001,N+1)  
#sigma2 = 1.e-6

#print 'y', y
#print 'xtrue', xtrue

Y = np.zeros([1,M*L])    #####MORE OBS#######            
S = np.zeros([1,M*L])    #####MORE OBS#######
dsdx = np.zeros([M*L,D]) #####MORE OBS#######        
dxds = np.zeros([D,M*L]) #####MORE OBS#######

xx = np.zeros([D,1])      
xtran = np.zeros([D,1]) 

Jac = np.zeros([D,D])                 
for i in range(D):
    Jac[i,i] = 1.

Jac0 = np.copy(Jac)   

run = 735

oo = np.zeros([1,run+1])      
svmaxvec = np.zeros([1,run+1]) 
svmaxvec2 = np.zeros([1,run+1]) 

for n in range(1,run+1):
    t = (n-1)*dt

    #S[:,0] = np.dot(h,x[:,n])   
    #Y[:,0] = y[:,n]             
    dsdx[0:L,:] = h          #####MORE OBS#######
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

            if m == 2:
                if i == 1:
                    xf = xx
                    
                    S[:,0:L] = np.dot(h,xx)   #####MORE OBS#######
                    Y[:,0:L] = y[0:L,n+1]     #####MORE OBS#######

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
        S[:,idxs:idxs+L] = np.dot(h,xx)        #####MORE OBS#####
        #print 'S at m', m, 'is', S.shape
        #idy = n+(m-1)*nTau
        Y[:,idxs:idxs+L] = y[0:L,n+(m-1)*nTau] #####MORE OBS##### 
        dsdx[idxs:idxs+L,:] = np.dot(h,Jac)    #####MORE OBS#####
    #print 'dsdx', dsdx

    ########dxds = np.linalg.pinv(dsdx,rcond=pinv_tol)    
    #dxds = dxds.round(decimals=4)     # Applied this as it was appearing in matlab code (1st row 1 0 0 0...)
    

    ### Constructing Sherman-Morrison-Woodbury format of calculating the Kalman gain to compare with synch ###
    Pinverse = np.linalg.pinv(Jac0)
    #print 'Pinverse', Pinverse
    sigmaP = np.dot(sigma2,Pinverse)

    SS = np.dot(np.transpose(dsdx),dsdx)

    sigmaPSS = sigmaP + SS 
    #print 'sigmaPSS shape', sigmaPSS.shape 

    ############ Calculating the inverse using SVD decomposition ################
    #U, G, V = mod.svd(dsdx)
    U, G, V = mod.svd(sigmaPSS)
    #print 'U', U
    #print 'G', G                       # All positive values, for good or bad runs. 
    #print 'V', V
    #print 'ln(G)', np.log(G)

    ##### Calculating singular values, condition number, observability and ratios ######
    svmin = np.min(G)
    #svminrank = svmin
    #print 'Smallest sing value:', svmin              #no influence until now...(around e-03)
    svmax = np.max(G) 
    #print 'Largest sing value:', svmax   
    svmaxvec[:,n] = svmax
    svmaxvec2[:,n] = G[1]
    difmax = abs(svmaxvec[:,n] - svmaxvec[:,n-1])
    difmax2 = abs(svmaxvec2[:,n] - svmaxvec2[:,n-1])
    ###difmax = svmaxvec[:,n] - svmaxvec[:,n-1]
    #print 'Difmax=', difmax
    
    difsv = svmax - svmin    
    ratioobs = svmin/svmax
    condnumber = svmax/svmin
    #print 'Condition number is', condnumber
    oo[:,n] = (ratioobs)**2
    obin = np.sum(oo)   
    #print 'observability', observ                   #no influence until now...(between e-05 and e-04)

    ##### Dynamic rank calculation #######
    ###if condnumber > 1.e6:
        #if max_pinv_rank == M:
        ###max_pinv_rank = 7
        #else:
        #    max_pinv_rank = max_pinv_rank - 1

        #svminrank = G[max_pinv_rank-1]        
        ###print 'max_pinv_rank', max_pinv_rank

    ###elif condnumber > 1.e7:
        ###max_pinv_rank = 6

        ###svminrank = G[max_pinv_rank-1] 

        ###print 'max_pinv_rank', max_pinv_rank

    #else:
        #max_pinv_rank = M

        #print 'max_pinv_rank', max_pinv_rank

    #print 'max_pinv_rank', max_pinv_rank

    svminrank = G[max_pinv_rank-1]

    ################ SVD ##################  
    mask = np.ones(len(G)) 
    for k in range(len(G)):
        #mask = np.ones(len(G))        
        if G[k] >= pinv_tol:
            mask[k] = 1
        else:
            mask[k] = 0
        #print 'mask', mask
    rr = min(max_pinv_rank,sum(mask)) 
    #print 'r is', r
    g = G[:rr]**(-1) 
    #print 'g', g  
    Ginv = np.zeros((D, D))
    Ginv[:rr, :rr] = np.diag(g)
    #print 'Ginv', Ginv 
    ###Ginv = np.diag(Ginv)
    #print 'Ginv2', Ginv    
    dxds1 = np.dot((np.transpose(V[:,:])),(np.transpose(Ginv)))   
    #print 'dxds1', dxds1.shape
    ########dxds = np.dot(dxds1,(np.transpose(U[:,:r])))  
    ##dxds = np.dot(dxds1,(np.transpose(U[:,:])))
    sigmaPSSinv = np.dot(dxds1,(np.transpose(U[:,:])))    
    #print 'dxds', dxds 
    #print 'Y', Y
    #print 'S', S
    

    #dx1 = np.dot(K,dxds)
    dx1 = np.dot(sigmaPSSinv,np.transpose(dsdx))
    #print 'dx1', dx1
    dx2 = np.dot(Ks,np.transpose((Y-S)))
    dx = np.dot(dx1,dx2)
    dx = dx.reshape(D)
    #print 'dx', dx
    #print 'x[:,n]shape', x[:,n].shape
    
    ddd = Y-S
    #print 'ddd', ddd

    random = np.zeros(D)
        
    #x[:,n+1] = mod.rk4_end(x[:,n],dx,dt) #+ dx0                               #4)n4m2 #4)n2500m4 (peaks) for dt*dx in rk4

    x[:,n+1] = xf + dx

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
    
    #plt.plot(n+1,svmin,'c<') 
    #plt.yscale('log')
    #plt.hold(True)

    #plt.plot(n+1,svminrank,'yo') 
    #plt.yscale('log')
    #plt.hold(True)

    #plt.plot(n+1,svmax,'r>') 
    #plt.yscale('log')
    #plt.hold(True)

    #plt.plot(n+1,ratioobs,'yo') 
    #plt.hold(True)
  
    #plt.plot(n+1,condnumber,'m.') 
    #plt.yscale('log')
    #plt.hold(True)

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

print time.clock() - start_time, "seconds"
################################ Prediction ############################################
random = np.zeros(D)
time_pred1 = run
time_pred2 = run
time_pred3 = run

threshold1 = np.sqrt(sigma2)*3
threshold2 = np.sqrt(sigma2)*4
threshold3 = np.sqrt(sigma2)*5

for w in range(run+1,fc):
    x[:,w+1] = mod.lorenz96(x[:,w],random,dt) 

    dddd = np.zeros([D,1])
    dddd[:,0] = xtrue[:,w+1] - x[:,w+1]
    SEpred = np.sqrt(np.mean(np.square(dddd))) 
    
    if SEpred <= threshold3:
        time_pred3 = w+1
        #SE_predlim3 = SEpred
    pred_range3 = time_pred3 - (run+1)
    
    if SEpred <= threshold2:
        time_pred2 = w+1
        #SE_predlim2 = SEpred
    pred_range2 = time_pred2 - (run+1)
    
    if SEpred <= threshold1:
        time_pred1 = w+1
        #SE_predlim1 = SEpred
    pred_range1 = time_pred1 - (run+1)
              
    #print 'SE prediction for', w+1, 'is', SEpred
    print 'Prediction Range for threshold 1 is', pred_range1
    print 'Prediction Range for threshold 2 is', pred_range2
    print 'Prediction Range for threshold 3 is', pred_range3

    plt.plot(w+1,SEpred,'mo') 
    plt.plot ([run,fc],[threshold1,threshold1], 'g-', lw=2)  
    plt.plot ([run,fc],[threshold2,threshold2], 'y-', lw=2)
    plt.plot ([run,fc],[threshold3,threshold3], 'r-', lw=2)  
    plt.legend(['RMSE','Threshold1 ='+str(threshold1)+'', 'Threshold2 = '+str(threshold2)+'', 'Threshold3 = '+str(threshold3)+''], loc='upper left')
    plt.yscale('log')
    plt.xlim((run,fc))
    plt.title('Prediction Ranges')
      
    plt.hold(True)

    if w == fc-1:
        ##plt.annotate(time_pred, xy=(time_pred,SE_predlim,'bx')
        #plt.text(fc-50,SEpred, pred_range, fontsize=12)
        plt.text(fc-50,threshold1, pred_range1, bbox=dict(facecolor='green', alpha=0.5))
        plt.text(fc-50,threshold2, pred_range2, bbox=dict(facecolor='yellow', alpha=0.5))
        plt.text(fc-50,threshold3, pred_range3, bbox=dict(facecolor='red', alpha=0.5))
        #plt.text(0.8, 0.8, pred_range,horizontalalignment='center', verticalalignment='center')
#plt.annotate(time_pred, xy=(time_pred,SE_predlim,'bx')

plt.show()

######################## Plotting variables ###############################
plt.figure(figsize=(12, 10)).suptitle('Variables for D='+str(D)+', M='+str(M)+', r='+str(r)+', K='+str(K[0,0])+', max_pinv_rank= '+str(max_pinv_rank)+'')#, Prediction Range= '+str(pred_range)+'')
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
    

