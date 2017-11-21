#from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
#import functions as m
import model as mod
import time
start_time = time.clock()

#################### Initial settings ################################
N = 13000
Obs = 100
dt = 0.01    
fc = 12500

D = 20 
F=8.17

M = 5
tau= 0.1
nTau = tau/dt
print 'D=', D, 'variables and M=', M ,'time-delays'

Nens = 50    # ensemble size 

############# To plot different time-delays in the same graph ##################
#Nens_list = [10,15,20,50]
#ns_list = [10,15,20]
Nens_list = [10,15]
for w in Nens_list:
    Nens = w
    print 'Nens', Nens


    ###################### Seeding for 20 variables########################
    #r = 5
    #r=18 #for x[:,0] = xtrue[:,0]
    r=37 #for original code 
    #r=44  #for RK4 and 0.0005 uniform noise (for M = 10)
    #r=39   #for RK4 and 0.0005 uniform noise (for M = 12)

    np.random.seed(r)  


    #################### Constructing h (obs operator) ##################
    observed_vars = range(5)    
    L = len(observed_vars) 
    h = np.zeros([L,D])       
    for i in range(L):          
    #    h[i,observed_vars[i]] = 1.0      ## for observing the first vars in sequence 
    #    h[i,observed_vars[i]*2] = 1.0    ## for observing vars sparsely
        h[i,observed_vars[i]*(D/L)] = 1.0 ## for observing vars equally sparsed
    print 'h', h


    ################### Setting coupling matrices ########################
    K = 1.e1*np.diag(np.ones([D]))      # also testing: 2.e1, 5.e1, 1.e2
    #print 'K', K

    Ks = 1.e0*np.diag(np.ones([L*M]))  


    ######### Setting tolerance and maximum for rank calculations ########
    pinv_tol =  (np.finfo(float).eps)
    max_pinv_rank = M


    ################### Creating truth ###################################
    xtrue = np.zeros([D,N+1])
    #***xtrue[:,0] = np.random.rand(D)  
    #print 'xtrue[:,0]', xtrue[:,0]

    ####### Start by spinning model in ###########
    xtest = np.zeros([D,1001]) 
    xtest[:,0]=np.random.rand(D)
    #print '1st rand', xtest[:,0]

    for j in range(1000):
        force = np.zeros(D)
        xtest[:,j+1]=mod.lorenz96(xtest[:,j],force,dt)
             
    xtrue[:,0] = xtest[:,1000]
    print 'xtrue[:,0]', xtrue[:,0]

    ## Plot xtrue to understand initial conditions influences ##
    #plt.figure(1).suptitle('xtrue for seed='+str(r)+'')
    #plt.plot(xtrue[:,0],'g-')
    #plt.show()

    dx0 = np.random.rand(D)-0.5  
    #print 'dx0', dx0   
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
        #force = np.random.rand(D)-0.5  
        xtrue[:,j+1] = mod.lorenz96(xtrue[:,j],force,dt)  
    xtrue[:,1] = xtrue[:,0] 
    x[:,1] = x[:,0]         
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


    ############ Defining vectors and matrices ####################
    Ks = 1.e0*np.diag(np.ones([L*M]))  
    
    Y = np.zeros([1,M*L])          
    S = np.zeros([1,M*L])    
    dsdx = np.zeros([M*L,D])         
    dxds = np.zeros([D,M*L]) 

    xx = np.zeros([D,1])      
    xtran = np.zeros([D,1]) 

    Jac = np.zeros([D,D])                 
    for i in range(D):
        Jac[i,i] = 1.

    Jac0 = np.copy(Jac)   

    SEstore = []                     #### for calculating the mean and variance of the total SEs ####
    SEvarstore = []                  #### for calculating the mean and variance of the total SEs ####

    run = 3

    oo = np.zeros([1,run+1])         #for observability calculation
    svmaxvec = np.zeros([1,run+1]) 
    svmaxvec2 = np.zeros([1,run+1]) 

    E = np.zeros([D])                ### ensemble mean ###
    A = np.zeros([D,Nens])           ### for the covariance matrix ###
    B = np.zeros([D,Nens])           ### for the covariance matrix ###
    C = np.zeros([D,D])              ### covariance matrix ###   


    ########################### Main loop ##########################################
    for n in range(1,run+1):
        
        ################### Creating ensemble ##########################
        Ens = np.zeros([D,Nens])
        for i in range(len(Ens[0,:])):
            #random = np.random.randn(N)

            ##force = np.random.rand(D) 
            ##force = np.random.rand(D)-0.5    ## IT SYNCHRONISES WITH THESE UNIF PERTURBATIONS!
            force = 0.1*np.random.randn(D) + 0 ## IT SYNCHRONISES WITH THESE NORMAL PERTURBATIONS!
            #print 'force', force
            #force = dx0

            #initialEns[:,i] = xtrue[:,0] + force
            Ens[:,i] = x[:,n] + force        # ensemble created from our initial x
            #print 'Ensemble member', i, 'is', Ens[:,i]
        #print 'initial ensemble created'
        #print 'Ens for time', n, 'is', Ens

        ############ Calculate the ensemble mean to construct S ########    
        for a in range(D):
            E[a] = np.mean(Ens[a,:])
        #print 'The ensemble mean is', E
        #print 'E shape', E.shape 
        #print 'x is', x[:,n]
        #print 'x shape', x[:,n].shape
       
        ############ For the covariance (time n) ##########  
        for b in range(D):
            for c in range(Nens):
                A[b,c] = Ens[b,c] - E[b]  
        #print 'A is', A.shape
        #Acovar = (np.dot(A,np.transpose(A)))/(Nens-1)    
        #print 'A covar is', Acovar 

        ########## Constructing S and Y vectors ########################
        S[:,0:L] = np.dot(h,E)        
        #print 'S', S
        Y[:,0:L] = y[0:L,n]           
        #print 'Y', Y    
        dsdx[0:L,:] = h               
        #print 'dsdx', dsdx
      
        ###########Jac = Jac0
        #print 'Jac', Jac


        for m in range(2,M+1):
            ####### Propagate the ensemble members forward "tau" times #########
            for i in range(1,int(nTau)+1):
                for j in range(len(Ens[0,:])):
                    random = np.zeros(D)
                    #random = np.random.rand(D)-0.5
                    Ens[:,j] = mod.lorenz96(Ens[:,j],random,dt) 
                #print 'Ensemble for time', i+1, 'is', Ens

            ######## Calculate the mean and covariance of the ensemble#################
            ############ Mean ################    
            for a in range(D):
                E[a] = np.mean(Ens[a,:])
            #print 'The ensemble mean is', E 
            
            ############ Covariance ########## 
            for d in range(D):
                for e in range(Nens):
                    B[d,e] = Ens[d,e] - E[d]      
            #print 'B is', B.shape


            #C = (np.dot(B,np.transpose(A)))/(Nens-1)   ## 1st try

            ##A_inv = np.linalg.pinv(A)                   ## 3rd try
            ##C = (np.dot(B,A_inv))/(Nens-1)  
            
            ###C = (np.dot(B,A_inv))                       ## 5th try  

            C1 = (np.dot(B,np.transpose(A)))         ## 2nd try
            C2 = (np.dot(A,np.transpose(A)))
            #print '1st C2', C2

            # Assuming covariances are zero (only variances considered), due to inversion ###
            ####*for f in range(D):
            ####*    for g in range(D):
            ####*        if f != g:
            ####*            C2[f,g] = 0  
            #print 'C2', C2.shape
            
            C2_inv = np.linalg.pinv(C2)
            ####C = (np.dot(C1,C2_inv))/(Nens-1)
            
            C = (np.dot(C1,C2_inv))                 ## 4th try
            #print 'C is', C

            Jac = C    
            
            ########## Constructing S and Y vectors ########################   
            idxs = L*(m-1)        
            #print 'idxs', idxs        
            S[:,idxs:idxs+L] = np.dot(h,E)        
            #print 'S at m', m, 'is', S
            #idy = n+(m-1)*nTau
            Y[:,idxs:idxs+L] = y[0:L,n+(m-1)*nTau] 
            #print 'Y at m', m, 'is', Y
            dsdx[idxs:idxs+L,:] = np.dot(h,Jac)    
            #print 'dsdx', dsdx
        #print 'dsdx', dsdx

        ##### Calculating the pseudoinverse (using SVD decomposition) #####
        #dxds = np.linalg.pinv(dsdx,rcond=pinv_tol)    
        
        U, G, V = mod.svd(dsdx)
        #print 'U', U.shape
        #print 'G', G.shape                       
        #print 'V', V.shape
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
        print 'Condition number is', condnumber
        oo[:,n] = (ratioobs)**2
        obin = np.sum(oo)   
        #print 'observability', observ                   #no influence until now...(between e-05 and e-04)

        ##### Proceeding with SVD computation...#######
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
        Ginv = np.zeros((L*M, D))            
        Ginv[:rr, :rr] = np.diag(g)
        #print 'Ginv', Ginv 
        ###Ginv = np.diag(Ginv)
        
        dxds1 = np.dot((np.transpose(V[:,:])),(np.transpose(Ginv)))   
        #print 'dxds1', dxds1.shape
        ########dxds = np.dot(dxds1,(np.transpose(U[:,:r])))  
        dxds = np.dot(dxds1,(np.transpose(U[:,:])))  
        #print 'dxds', dxds 
        #print 'Y', Y
        #print 'S', S
        
        #### Calculating the coupling term in coupled dynamics ######
        dx1 = np.dot(K,dxds)
        #print 'dx1', dx1
        dx2 = np.dot(Ks,np.transpose((Y-S)))
        dx = np.dot(dx1,dx2)
        dx = dx.reshape(D)
        #print 'dx', dx
        #print 'x[:,n]shape', x[:,n].shape
        
        ddd = Y-S
        #print 'ddd', ddd

        #### Evolving variables with time #######
        random = np.zeros(D)
        x[:,n+1] = mod.rk4_end(x[:,n],dx,dt) 
                               
        #x[:,n+1] = mod.lorenz96(x[:,n],random,dt) + dx              

        #x[:,n+1] = mod.lorenz96(x[:,n],random,dt) + dt*dx                     

        #print 'x[:,n+1] at', n+1, 'is', x[:,n+1]

        ####### Calculating the synchronisation error (RMSE) #######
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
        
        ####### Plotting the synchronisation error (RMSE) #######
        fig = plt.figure(1)
        ax = fig.add_subplot(1, 1, 1) # create an axes object in the figure
        
        color = ['r', 'b', 'g', 'm']

        if Nens == Nens_list[0]:
            cc = color[0]
 
            plt.plot(n+1,condnumber,''+str(cc)+'+',label='Nens='+str(Nens)+'') 
            l1, = plt.plot(n+1,condnumber,''+str(cc)+'+',label='Nens='+str(Nens)+'') 
            #plt.plot(n+1,condnumber,'m.') 
            plt.yscale('log')


        elif Nens == Nens_list[1]:
            cc = color[1]
 
            plt.plot(n+1,condnumber,''+str(cc)+'o',label='Nens='+str(Nens)+'') 
            l2, = plt.plot(n+1,condnumber,''+str(cc)+'o',label='Nens='+str(Nens)+'') 
            #plt.plot(n+1,condnumber,'y.') 
            plt.yscale('log')


        elif Nens == Nens_list[2]:
            cc = color[2]

            plt.plot(n+1,condnumber,''+str(cc)+'x',label='Nens='+str(Nens)+'') 
            l3, = plt.plot(n+1,condnumber,''+str(cc)+'x',label='Nens='+str(Nens)+'') 
            #plt.plot(n+1,condnumber,'m.') 
            plt.yscale('log')


        else:
            cc = color[3]

            plt.plot(n+1,condnumber,''+str(cc)+'.',label='Nens='+str(Nens)+'') 
            l4, = plt.plot(n+1,condnumber,''+str(cc)+'.',label='Nens='+str(Nens)+'') 
            #plt.plot(n+1,condnumber,'m.') 
            plt.yscale('log')

        plt.title('Condition number')
        plt.xlabel('Time units')        
        plt.ylabel('cond number')

        for tick in ax.xaxis.get_ticklabels():
            tick.set_fontsize('large')
            #tick.set_fontname('Times New Roman')
            #tick.set_color('blue')
            ##tick.set_weight('bold')
        for tick in ax.yaxis.get_ticklabels():
            tick.set_fontsize('large')
            #tick.set_fontname('Times New Roman')
            #tick.set_color('blue')
            ##tick.set_weight('bold')
        plt.hold(True)
        
        ####### Storing SEs after the 1st minimum to take the mean and variance ############
        #if n > 250:
        #    SEstore.append(SE)
       
        #    SEmean = np.mean(SEstore)
                
        #    SEvareach = (SE-SEmean)**2
        #    SEvarstore.append(SEvareach)

        #    SEvar = np.mean((SE-SEmean)**2)
          
        #    #if n > 251:
        #    plt.figure(5)
        #    plt.plot(n+1,SEmean,'bo')
        #    #plt.yscale('log')
        #    plt.hold(True)
                
        #    plt.figure(6)
        #    plt.plot(n+1,SEvar,'rx')
        #    #plt.yscale('log')
        #    plt.hold(True)
        #    #print 'SEvar', SEvar

    #SEmeantot = np.mean(SEstore)
    #SEvartot = np.mean(SEvarstore)

    #print 'SEmeantot=', SEmeantot
    #print 'SEvartot=', SEvartot

#plt.legend()
#plt.legend([label1,label2,label3],loc='best')

#plt.legend((l1, l2, l3, l4),(l1.get_label(),l2.get_label(),l3.get_label(),l4.get_label()), loc='best')
#plt.legend((l1, l2, l3),(l1.get_label(),l2.get_label(),l3.get_label()), loc='best')
plt.legend((l1, l2),(l1.get_label(),l2.get_label()), loc='best')
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
#for i in range(D/2):  # for 20 vars
for i in range(D/4):   # for 40 vars
#    plt.subplot(np.ceil(D/8.0),2,i+1)
    plt.subplot(5,2,i+1)
    if i == 0:  
        plt.plot(y[0,:],'r.',label='obs')   ## create y with no zeros to plot correctly ###
        plt.hold(True)      
           
    plt.plot(x[i,:],'g',label='X')
    plt.hold(True)
    plt.plot(xtrue[i,:],'b-',linewidth=2.0,label='truth')
    plt.hold(True)
        
    plt.ylabel('x['+str(i)+']')
    plt.xlabel('time steps')

plt.figure(figsize=(12, 10)).suptitle('Variables for D='+str(D)+', M='+str(M)+', r='+str(r)+', K='+str(K[0,0])+', max_pinv_rank= '+str(max_pinv_rank)+'')
#for k in range(D/2,D):  # for 20 vars
for k in range(D/4,D/2):  # for 40 vars
#for i in range(D/3):
#for i in range(D):
    #plt.subplot(np.ceil(D/8.0),2,i+1)
    i2 = k - 10
    plt.subplot(5,2,i2+1)
    #plt.subplot(5,4,i+1)
    
    plt.plot(x[k,:],'g',label='X')
    plt.hold(True)
    plt.plot(xtrue[k,:],'b-',linewidth=2.0,label='truth')
    plt.hold(True)
        
    plt.ylabel('x['+str(k)+']')
    plt.xlabel('time steps')
plt.show()
    

