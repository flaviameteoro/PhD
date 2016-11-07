#from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
#import functions as m
import model as mod
import time
#start_time = time.clock()
start_time2 = time.time()
#print 'time', start_time2

#################### Initial settings ################################
N = 13000
Obs = 100
dt = 0.01    #original value=0.01
fc = 12999

D = 20     # Note that max_pinv is now 2L (for all Ds)!
F=8.17

M = 5
tau= 0.1
nTau = tau/dt
print 'D=', D, 'variables and M=', M ,'time-delays'


###################### Seeding for 20 variables########################
#r = 5
#r=18 #for x[:,0] = xtrue[:,0]
r=37 #for original code 
#r=44  #for RK4 and 0.0005 uniform noise (for M = 10)
#r=39   #for RK4 and 0.0005 uniform noise (for M = 12)

np.random.seed(r)  


#################### Constructing h (obs operator) ##################
observed_vars = range(5)    ######MORE OBS#########
L = len(observed_vars) 
h = np.zeros([L,D])       
for i in range(L):          
#    h[i,observed_vars[i]] = 1.0  ######MORE OBS######### for observing the first vars in sequence 
#    h[i,observed_vars[i]*2] = 1.0 ######MORE OBS######### for observing vars sparsely
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
max_pinv_rank = 2*L


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

###### Good noise values for y (for seed=37)
#y = np.dot(h,xtrue) + np.random.uniform(0,1.2680e-04,N+1)-6.34e-05
#y = np.dot(h,xtrue) + np.random.uniform(0,1.2680e-04,N+1)-9.34e-05  #(out of zero mean!)
#y = np.dot(h,xtrue) + np.random.uniform(0,1.8680e-04,N+1)-9.34e-05

###### Noise that runs totally ok for seeds=37 and 44 (rank=9) (after SPINUP-1000) (until 1500 for seed=37 before)  
###### Runs totally ok for seed=18, with SPINUP-1000!!!!!!!!!!!!!!!!!!!
###### Runs until 8500 for seed=39 (rank=8)
#y = np.dot(h,xtrue) + np.random.uniform(0,0.001,N+1)-0.0005
#y = np.dot(h,xtrue) + np.random.normal(0,0.0005,N+1)     

###### Bad noise values for y (for seed=37)
#y = np.dot(h,xtrue) + np.random.rand(N+1)-0.5
#y = np.dot(h,xtrue) + np.random.uniform(0,0.2,N+1)-0.1     #### OK for 4 obs vars!!! (r = 18)####
y = np.dot(h,xtrue) + np.random.normal(0,0.1,N+1)           #### OK for 4 obs vars!!! (r = 18)####
sigma2 = 0.01
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

SEstore = []                     #### for calculating the mean and variance of the total SEs ####
SEvarstore = []                  #### for calculating the mean and variance of the total SEs ####

run = 9950

oo = np.zeros([1,run+1])      #for observability calculation
svmaxvec = np.zeros([1,run+1]) 
svmaxvec2 = np.zeros([1,run+1]) 

dlyaini = x[:,1] - xtrue[:,1]
#print 'dlyaini', dlyaini


########################### Main loop ######################################
for n in range(1,run+1):
    t = (n-1)*dt

    S[:,0:L] = np.dot(h,x[:,n])   #####MORE OBS####### # this 0 term should be (0:L) in case of more obs
    #print 'S', S
    Y[:,0:L] = y[0:L,n]           #####MORE OBS####### # this 0 term should be (0:L) in case of more obs
    #print 'Y', Y    
    dsdx[0:L,:] = h               #####MORE OBS#######
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

        idxs = L*(m-1)        #attention to this (L)term, which should be (1:L) in case of more obs(??)
        #print 'idxs', idxs        
        S[:,idxs:idxs+L] = np.dot(h,xx)        #####MORE OBS#####
        #print 'S at m', m, 'is', S
        #idy = n+(m-1)*nTau
        Y[:,idxs:idxs+L] = y[0:L,n+(m-1)*nTau] #####MORE OBS#####  
        #print 'Y at m', m, 'is', Y
        dsdx[idxs:idxs+L,:] = np.dot(h,Jac)    #####MORE OBS#####
        #print 'dsdx', dsdx
    #print 'dsdx', dsdx

    ########dxds = np.linalg.pinv(dsdx,rcond=pinv_tol)    
    #dxds = dxds.round(decimals=4)     # Applied this as it was appearing in matlab code (1st row 1 0 0 0...)
    
    U, G, V = mod.svd(dsdx)
    #print 'U', U.shape
    #print 'G', G.shape                       # All positive values, for good or bad runs. 
    #print 'V', V.shape
    #print 'ln(G)', np.log(G)


    ################### Calculating singular values, condition number, observability and ratios ########################
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

    condnumber = svmax/svmin

    difsv = svmax - svmin    
    ratioobs = svmin/svmax
    
    oo[:,n] = (ratioobs)**2
    obin = np.sum(oo)   
    #print 'observability', observ                   #no influence until now...(between e-05 and e-04)


    ################ Proceeding with SVD computation...##############
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
    Ginv = np.zeros((L*M, D))            #####MORE OBS#####
    Ginv[:rr, :rr] = np.diag(g)
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
    

    ####### Calculating the coupling term in coupled dynamics ######
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

    x[:,n+1] = mod.rk4_end(x[:,n],dx,dt) #+ dx0                               #4)n4m2 #4)n2500m4 (peaks) for dt*dx in rk4

   
    ###################Calculating and plotting Lyapunov exponents ########################
    dlya = x[:,n+1] - xtrue[:,n+1]
    #print 'dlya', dlya
    dl = abs(dlya/dlyaini)   
    #print 'dl', dl
    lya = (1/float(n))*(np.log(dl))
    
    #lyaposit = 0
    #for i in range(D):
    #    if lya[i] > 0:
    #        lyaposit = lyaposit + 1
    #print 'Lyapositive', lyaposit

    plt.figure(1).suptitle('Positive Lyapunov Exponents')
    #plt.axhline(y=0, xmin=0, xmax=run, linewidth=1, color = 'm')
    ##for i in range(D):   
    ###for i in range(D):                   # to plot all variables!!
        ###plt.subplot(D/4,4,i+1)           # to plot all variables!!
    #for i in range(D/3):
    for i in range(D/2):                    # for 20 vars
    #for i in range(D/10):                   # for 100 vars
    #for i in range(D/100):                   # for 1000 vars
    #for i in range(D/500):                   # for 5000 vars
       #plt.subplot(np.ceil(D/8.0),2,i+1)  
        plt.subplot(5,2,i+1)
        if lya[i] >0:
            plt.plot(n+1,lya[i],'y.',label='truth')
            plt.yscale('log')
            plt.ylabel('x['+str(i)+']',fontsize=8)
            plt.hold(True)

    plt.figure(2).suptitle('Positive Lyapunov Exponents')
    #plt.axhline(y=0, xmin=0, xmax=run, linewidth=1, color = 'm')
    for k in range(D/2,D):                 # for 20 vars
    #for k in range(D/10,D/5):              # for 100 vars
    #for k in range(D/100,D/50):             # for 1000 vars
    #for k in range(D/500,D/250):             # for 5000 vars
        i2 = k - 10
        plt.subplot(5,2,i2+1)
        if lya[k] >0:
            plt.plot(n+1,lya[k],'y.',label='truth')
            plt.yscale('log')
            plt.ylabel('x['+str(k)+']',fontsize=8)
            plt.hold(True)


###################### Calculating and plotting SE values ###############
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
    
    fig = plt.figure(3)
    ax = fig.add_subplot(1, 1, 1) # create an axes object in the figure
    plt.plot(n+1,SE,'b*') 
    plt.yscale('log')

    plt.title('RMSE')
    plt.xlabel('Time units')        
    plt.ylabel('RMSE(t)')
    for tick in ax.xaxis.get_ticklabels():
        tick.set_fontsize('large')
        #tick.set_fontname('Times New Roman')
        #tick.set_color('blue')
        #tick.set_weight('bold')
    for tick in ax.yaxis.get_ticklabels():
        tick.set_fontsize('large')
        #tick.set_fontname('Times New Roman')
        #tick.set_color('blue')
        #tick.set_weight('bold')
    plt.hold(True)
    
    ##plt.plot(n+1,svmin,'c<') 
    ##plt.hold(True)

    ##plt.plot(n+1,svmax,'r>') 
    ##plt.hold(True)

    #plt.plot(n+1,ratioobs,'yo') 
    #plt.hold(True)
  
    #plt.plot(n+1,condnumber,'m.') 
    #plt.hold(True)

    #plt.plot(n+1,obin,'m<') 
    #plt.hold(True)

    #plt.plot(n+1,difsv,'mo') 
    #plt.hold(True)

    #plt.plot(n+1,difmax,'yo') 
    #plt.hold(True)

    #plt.plot(n+1,difmax2,'mo') 
    #plt.hold(True)

    ####### Storing SEs after the 1st minimum to take the mean and variance ############
    if n > 250:
        SEstore.append(SE)
   
        SEmean = np.mean(SEstore)
            
        SEvareach = (SE-SEmean)**2
        SEvarstore.append(SEvareach)

        SEvar = np.mean((SE-SEmean)**2)
      
        #if n > 251:
        plt.figure(5)
        plt.plot(n+1,SEmean,'bo')
        #plt.yscale('log')
        plt.hold(True)
            
        plt.figure(6)
        plt.plot(n+1,SEvar,'rx')
        #plt.yscale('log')
        plt.hold(True)
        #print 'SEvar', SEvar

SEmeantot = np.mean(SEstore)
SEvartot = np.mean(SEvarstore)

print 'SEmeantot=', SEmeantot
print 'SEvartot=', SEvartot

##print 'time', time.clock()
##end_time = time.clock() - start_time
#print (time.clock() - start_time), "seconds"
##print end_time, "seconds"

end_time2 = time.time() - start_time2
print 'time', end_time2

#obin_gama = (1./float(n))*np.sum(oo)               #see article Parlitz, Schumann-Bischoff and Luther, 2015
#print 'Observability Index is', obin_gama

plt.show()

#print time.clock() - start_time, "seconds"

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

    ###plt.plot(w+1,SEpred,'mo') 
    ###plt.plot ([run,fc],[threshold1,threshold1], 'g-', lw=2)  
    ###plt.plot ([run,fc],[threshold2,threshold2], 'y-', lw=2)
    ###plt.plot ([run,fc],[threshold3,threshold3], 'r-', lw=2)  
    ###plt.legend(['RMSE','Threshold1 ='+str(threshold1)+'', 'Threshold2 = '+str(threshold2)+'', 'Threshold3 = '+str(threshold3)+''], loc='upper left')
    ###plt.yscale('log')
    ###plt.xlim((run,fc))
    ###plt.title('Prediction Ranges')
      
    ###plt.hold(True)

    ###if w == fc-1:
        ##plt.annotate(time_pred, xy=(time_pred,SE_predlim,'bx')
        #plt.text(fc-50,SEpred, pred_range, fontsize=12)
        ###plt.text(fc-50,threshold1, pred_range1, bbox=dict(facecolor='green', alpha=0.5))
        ###plt.text(fc-50,threshold2, pred_range2, bbox=dict(facecolor='yellow', alpha=0.5))
        ###plt.text(fc-50,threshold3, pred_range3, bbox=dict(facecolor='red', alpha=0.5))
        #plt.text(0.8, 0.8, pred_range,horizontalalignment='center', verticalalignment='center')
#plt.annotate(time_pred, xy=(time_pred,SE_predlim,'bx')

###plt.show()

######################## Plotting variables ###############################
plt.figure(figsize=(12, 10)).suptitle('Variables for D='+str(D)+', M='+str(M)+', r='+str(r)+', K='+str(K[0,0])+', max_pinv_rank= '+str(max_pinv_rank)+'')
#for i in range(D/3):
for i in range(D/2):  # for 20 vars
#for i in range(D/4):   # for 40 vars
#for i in range(D/10):   # for 100 vars       old:range(D/20):   
#for i in range(D/100):   # for 1000 vars
#for i in range(D/500):     # for 5000 vars
#    plt.subplot(np.ceil(D/8.0),2,i+1)
    plt.subplot(5,2,i+1)
    #plt.subplot(5,5,i+1)
    #if i == 0:  
        #plt.plot(y[0,:],'r.',label='obs')   ## create y with no zeros to plot correctly ###
        #plt.hold(True)      
    ################## Plotting the observations ###########################
    i_obs = (D/L)
    if np.mod(i,i_obs) == 0:   
        i_y = i/i_obs
        plt.plot(y[i_y,:],'r.',label='obs')   ## create y with no zeros to plot correctly ###
        plt.hold(True)

    ##################### Plotting the variables ###########################
    plt.plot(x[i,:],'g',label='X')
    plt.hold(True)
    plt.plot(xtrue[i,:],'b-',linewidth=2.0,label='truth')
    plt.hold(True)
        
    plt.ylabel('x['+str(i)+']')
    plt.xlabel('time steps')

plt.figure(figsize=(12, 10)).suptitle('Variables for D='+str(D)+', M='+str(M)+', r='+str(r)+', K='+str(K[0,0])+', max_pinv_rank= '+str(max_pinv_rank)+'')
for k in range(D/2,D):  # for 20 vars
#for k in range(D/4,D/2):  # for 40 vars
#for k in range(D/10,D/5):  # for 100 vars    old: range(D/20,D/10): 
#for k in range(D/100,D/50):   # for 1000 vars
#for k in range(D/500,D/250):     # for 5000 vars
#for i in range(D/3):
#for i in range(D):
    #plt.subplot(np.ceil(D/8.0),2,i+1)
    i2 = k - 10
    plt.subplot(5,2,i2+1)
    #plt.subplot(5,5,i2+1)
    #plt.subplot(5,4,i+1)
    ################## Plotting the observations ###########################
    i_obs = (D/L)
    if np.mod(k,i_obs) == 0:   
        i_y = k/i_obs
        plt.plot(y[i_y,:],'r.',label='obs')   ## create y with no zeros to plot correctly ###
        plt.hold(True)

    ##################### Plotting the variables ###########################    
    plt.plot(x[k,:],'g',label='X')
    plt.hold(True)
    plt.plot(xtrue[k,:],'b-',linewidth=2.0,label='truth')
    plt.hold(True)
        
    plt.ylabel('x['+str(k)+']')
    plt.xlabel('time steps')
plt.show()
    

