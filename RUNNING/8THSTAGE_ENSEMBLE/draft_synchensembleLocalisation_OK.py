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
dt = 0.01    
fc = 12999

D = 20 
F=8.17

M = 5
tau= 0.1
nTau = tau/dt
print 'D=', D, 'variables and M=', M ,'time-delays'

Nens = 5  # ensemble size 


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

##h4 = h[:,0:Nens]                      ## Probably for all cases, included after the 4th formulation!!
#print 'h4', h4

##h4 = np.zeros([L,Nens])       
##for i in range(L):          
#    h[i,observed_vars[i]] = 1.0      ## for observing the first vars in sequence 
#    h[i,observed_vars[i]*2] = 1.0    ## for observing vars sparsely
##    h4[i,observed_vars[i]*(D/L)] = 1.0 ## for observing vars equally sparsed

if Nens < D:
    h4 = h[:,0:Nens]                      ## Probably for all cases, included after the 4th formulation!!
else:
    h4 = np.zeros([L,Nens])       
    for i in range(L):          
    #    h[i,observed_vars[i]] = 1.0      ## for observing the first vars in sequence 
    #    h[i,observed_vars[i]*2] = 1.0    ## for observing vars sparsely
        h4[i,observed_vars[i]*(D/L)] = 1.0 ## for observing vars equally sparsed

################### Setting coupling matrices ########################
K = 1.e1*np.diag(np.ones([D]))      # also testing: 2.e1, 5.e1, 1.e2
print 'K', K.shape

K_new = 1.e2*np.diag(np.ones([Nens]))      #Increased to 100 to prevent dx to be divided by a factor of 10!!!! (what made Localisation work) # For the 5th formulation 


Ks = 1.e0*np.diag(np.ones([L*M]))  


######### Setting tolerance and maximum for rank calculations ########
pinv_tol =  (np.finfo(float).eps)
max_pinv_rank = Nens


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
Y = np.zeros([1,M*L])          
S = np.zeros([1,M*L])    

##dsdx = np.zeros([M*L,D])          # for ALL WORKING formulations, except 4th
##dxds = np.zeros([D,M*L])          # for ALL WORKING formulations, except 4th
dsdx = np.zeros([M*L,Nens])         # for the 4th formulation!!
dxds = np.zeros([Nens,M*L])         # for the 4th formulation!!
dxds_new = np.zeros([D,M*L])        # for the 4th formulation!!

xx = np.zeros([D,1])      
xtran = np.zeros([D,1]) 

Jac = np.zeros([D,D])                 
for i in range(D):
    Jac[i,i] = 1.

Jac0 = np.copy(Jac)   

SEstore = []                     #### for calculating the mean and variance of the total SEs ####
SEvarstore = []                  #### for calculating the mean and variance of the total SEs ####

run = 1000

oo = np.zeros([1,run+1])         #for observability calculation
svmaxvec = np.zeros([1,run+1]) 
svmaxvec2 = np.zeros([1,run+1]) 

EA = np.zeros([D])                ### ensemble mean ###
EB = np.zeros([D])                ### ensemble mean ###
A = np.zeros([D,Nens])           ### for the covariance matrix ###
B = np.zeros([D,Nens])           ### for the covariance matrix ###
C = np.zeros([D,D])              ### covariance matrix ###   

loc = 1000.
dloc = np.zeros(M*L)
dYS = np.zeros(M*L)
newdYS = np.zeros([1,M*L])

EnsBdic = {}
Edic = {}
#EnsBdic = set()
#Edic = {'1','2','3','4'}
#print 'Edic', Edic
########################### Main loop ##########################################

for n in range(1,run+1):
    
    ################### Creating ensemble ##########################
    EnsA = np.zeros([D,Nens])
    EnsB = np.zeros([D,Nens])

    for i in range(len(EnsA[0,:])):
        #random = np.random.randn(N)

        ##force = np.random.rand(D) 
        ##force = np.random.rand(D)-0.5    ## IT SYNCHRONISES WITH THESE UNIF PERTURBATIONS!
        force = 0.1*np.random.randn(D) + 0 ## IT SYNCHRONISES WITH THESE NORMAL PERTURBATIONS!
        #print 'force', force
        #force = dx0

        #initialEns[:,i] = xtrue[:,0] + force
        EnsA[:,i] = x[:,n] + force        # ensemble created from our initial x
        #print 'Ensemble member', i, 'is', Ens[:,i]
    #print 'initial ensemble created'
    #print 'Ens for time', n, 'is', Ens

    ############ Calculate the ensemble mean to construct S ########    
    for a in range(D):
        EA[a] = np.mean(EnsA[a,:])
    #print 'The ensemble mean is', E
    #print 'E shape', E.shape 
    #print 'x is', x[:,n]
    #print 'x shape', x[:,n].shape
   
    ############ For the covariance (time n) ##########  
    for b in range(D):
        for c in range(Nens):
            A[b,c] = EnsA[b,c] - EA[b]  
    #print 'A is', A.shape
    #Acovar = (np.dot(A,np.transpose(A)))/(Nens-1)    
    #print 'A covar is', Acovar 

    ########## Constructing S and Y vectors ########################
    S[:,0:L] = np.dot(h,EA)        
    #print 'S', S
    Y[:,0:L] = y[0:L,n]           
    #print 'Y', Y    
    dsdx[0:L,:] = h4                ## Probably for all cases, included after the 4th formulation!!    
    #print 'dsdx', dsdx
  
    ###########Jac = Jac0
    #print 'Jac', Jac

    EnsB = EnsA
    for m in range(2,M+1):
        if n == 1:
            ####### Propagate the ensemble members forward "tau" times #########
            for i in range(1,int(nTau)+1):
                for j in range(len(EnsB[0,:])):
                    random = np.zeros(D)
                    #random = np.random.rand(D)-0.5
                    EnsB[:,j] = mod.lorenz96(EnsB[:,j],random,dt) 
                #print 'Ensemble for time', i+1, 'is', Ens
                if i == nTau:
                    print 'EnsB is', EnsB
                    globals()['EnsB%s' % m] = EnsB
                    print 'For m', m, 'EnsB is', EnsB%m
            #idic = str(0)+str(m-1)
            #EnsBdic = {EnsB}
            #print EnsBdic
            #print 'For m', m, 'i_dic is', idic
            #print EnsB.shape
            #EnsBdic[idic] = EnsB 
            #print 'EnsBdic to be updated', EnsBdic[i_dic]
            #print 'EnsBdic', EnsBdic
            #print 'm', m
            #if m == 2:
                #EnsBdic['alpha'] = EnsB 
                #Edic['1'] = EnsB
                #print 'EnsBdic', EnsBdic
                #print 'Edic', Edic
            #else:
                #EnsBdic['beta'] = EnsB 
                #Edic['2'] = EnsB
                #print 'EnsBdic', EnsBdic
                #print 'Edic', Edic
            #print 'EnsBdic', EnsBdic
            
            ######## Calculate the mean and covariance of the ensemble#################
            ############ Mean ################    
            for a in range(D):
                EB[a] = np.mean(EnsB[a,:])
            #print 'The ensemble mean is', E 
            
            ############ Covariance ########## 
            for d in range(D):
                for e in range(Nens):
                    B[d,e] = EnsB[d,e] - EB[d]      
            #print 'B is', B.shape

        else:
            ####### Propagate the ensemble members ONLY from the next (M*tau)+n time step ahead #########
            random = np.zeros(D)
            
            ii_dic = m-1
            LastEnsB = EnsBdic[ii_dic]
            print 'LastEnsB', LastEnsB.shape

            for j in range(len(EnsB[0,:])):
                EnsB[:,j] = mod.lorenz96(LastEnsB[:,j],random,dt) 
                        

        #C = (np.dot(B,np.transpose(A)))/(Nens-1)   ## 1st try

        ##A_inv = np.linalg.pinv(A)                   ## 3rd try
        ##C = (np.dot(B,A_inv))/(Nens-1)  
        
        ###C = (np.dot(B,A_inv))                       ## 5th try  

        
        # Assuming covariances are zero (only variances considered), due to inversion ###
        ####*for f in range(D):
        ####*    for g in range(D):
        ####*        if f != g:
        ####*            C2[f,g] = 0  
        #print 'C2', C2.shape

        ### Testing different ways of calculating dsdx and dxds ####

        # 1st working formulation: B.At.(AAt)-1 (pseudoinverse) - primary results
        ##C1 = (np.dot(B,np.transpose(A)))         ## 2nd try
        ##C2 = (np.dot(A,np.transpose(A)))
        ##C2_inv = np.linalg.pinv(C2)
        ##C = (np.dot(C1,C2_inv))
        ##Jac = C 

        # 2nd working formulation: B.(A)-1 (pseudoinv) - Nancy's suggestion
        ##C3_inv = np.linalg.pinv(A)
        ##XX0i = (np.dot(B,C3_inv))  
        ##Jac = XX0i

        # 3rd NOT working formulation: straight to dxds = A.(B)-1.(h)-1 (pseudoinv) 
        ##B_inv = np.linalg.pinv(B)
        ##h_inv = np.linalg.pinv(h)
        ##idxss = m-2
        ##ABinv = np.dot(A,B_inv)
        ##dxds_new[:,idxss:idxss+L] = np.dot(ABinv,h_inv) 

        # 4th working formulation: dxds = A.(hB)-1 (pseudoinv)- PJ's
        # dsdx is constructed by h.B

        ########## Constructing S and Y vectors ########################   
        idxs = L*(m-1)        
        #print 'idxs', idxs        
        S[:,idxs:idxs+L] = np.dot(h,EB)        
        #print 'S at m', m, 'is', S
        #idy = n+(m-1)*nTau
        Y[:,idxs:idxs+L] = y[0:L,n+(m-1)*nTau] 
        #print 'Y at m', m, 'is', Y

        ################ Constructing dsdx ##############################
        # For 1st and 2nd working formulations:
        ##dsdx[idxs:idxs+L,:] = np.dot(h,Jac)

        # For 4th working formulation:
        dsdx[idxs:idxs+L,:] = np.dot(h,B)  
        #print 'dsdx', dsdx

        #idic = str(0)+str(m-1)
        #print 'm',  m
        #EnsBdic[idic] = EnsB 
        #print EnsBdic
        #for i = m:
        #('EnsB_%d' %m) = EnsB

    #print EnsBdic    

    #print 'dsdx', dsdx

    ##### Calculating the pseudoinverse (using SVD decomposition) #####
    #dxds = np.linalg.pinv(dsdx,rcond=pinv_tol)    
    
    U, G, V = mod.svd(dsdx)
    #print 'U', U.shape
    #print 'G', G                       
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
    rr = min(max_pinv_rank,sum(mask)) 
    #print 'rr is', rr
    g = G[:rr]**(-1) 
    #print 'g', g  
    ##Ginv = np.zeros((L*M, D))         # For the 1st and 2nd working formulations
    Ginv = np.zeros((L*M, Nens))        # For the 4th working formulation
    Ginv[:rr, :rr] = np.diag(g)
    #print 'Ginv', Ginv 
    ######Ginv = np.diag(Ginv)
    
    dxds1 = np.dot((np.transpose(V[:,:])),(np.transpose(Ginv)))   
    #print 'dxds1', dxds1.shape
    ########dxds = np.dot(dxds1,(np.transpose(U[:,:r])))  
    
    dxds = np.dot(dxds1,(np.transpose(U[:,:])))   # For the 1st, 2nd and 4th working formulations (3rd not working)

    #dxds_new = np.dot(A,dxds)       # Add this for the 4th formulation!

    #print 'dxds', dxds 
    #print 'Y', Y
    #print 'S', S
    
    #### Calculating the coupling term in coupled dynamics ######
    #dx1 = np.dot(K,dxds)             # For the 1st and 2nd working formulations
    #dx1 = np.dot(K,dxds_new)         # For the 4th working formulation (and the 3rd NOT working one)
    dx1 = np.dot(K_new,dxds)          # For the 5th working formulations
    #print 'dx1', dx1

    ######################## Implementing LOCALISATION #########################
    DX = np.zeros(D)    
    
    dYS = Y-S
    #print 'dYS', dYS

    dx2 = np.dot(A,dx1)

    #print 'dx2', dx2

    for i in range(D):
        #print 'var', i
        for j in range(M*L):
            #l1 = np.mod(j,L)*(L-1)
            l1 = np.mod(j,L)*(D/L)
            #print 'Initial l1', l1

            if l1 > D:
                #while l > D:                                   ###### to avoid negative distances #####
                for aa in range(1,L):
                    #l = l1 + aa*D
                    l = np.abs(l1) - aa*D 
                    #print 'new l1', l1
                    if l < D:
                        break
            else:
                l = l1
            #print 'Var:', i, 'measurement position:', j, 'l is', l
            #print 'Var:', i, 'l is', l

            dist1 = min(np.abs(i-l),D-np.abs(i-l))
            dist = (dist1)**2

            #print 'dist for var',i, ':', dist1
            #print 'dist', dist1

            if (dist1 > (3*loc)):
                dloc[j] = 0.
            else:
                dloc[j] = np.exp(-(dist/(2*(loc**2))))
            #print 'dloc', dloc
        #print 'dloc for var',i, ':', dloc
        
        for k in range(M*L):
            i_dYS = dYS[:,k]
            #print 'i_dYS', i_dYS
            #print 'i_dYS[0]', i_dYS[0]
            #print 'dloc[0]', dloc[0]
            #print 'i_dYS[0]', i_dYS[0]
            ##dYS[:,k] = dloc[k]*i_dYS[0]          # SCHUR Product!!
            #print 'dloc[k]', dloc[k]
            #print 'dYS[:,k]', dYS[:,k].shape
            i_newdYS = dloc[k]*i_dYS[0]            # SCHUR Product!! 
            newdYS[:,k] = i_newdYS
        #print 'New dYS to use in the variable equation to find var', i, ':', dYS

        dx3 = np.dot(dx2[i,:],np.transpose(newdYS)) 
        #print 'dx3 for var',i, ':', dx3

        DX[i] = dx3[0] 
        #dx2 = np.dot(Ks,np.transpose((Y-S)))    # For all formulations apart from 5th
        #dx2 = np.dot(dx1,np.transpose(dYS))    # For the 5th formulation

        #dx = np.dot(dx1,dx2)                     # For all formulations apart from 5th
        #dx = np.dot(A,dx2)                        # For the 5th formulation
        
        ##dx3 = dx3.reshape(D)        
        #dx = dx.reshape(D)
        #print 'dx', dx
        #print 'x[:,n]shape', x[:,n].shape
        #print 'x[:,n]', x[:,n]
    #print 'DX', DX
    #### Evolving variables with time #######
    random = np.zeros(D)
    #x[:,n+1] = mod.rk4_end(x[:,n],dx,dt) 
    x[:,n+1] = mod.rk4_end(x[:,n],DX,dt) 
                               
        #x[:,n+1] = mod.lorenz96(x[:,n],random,dt) + dx              

        #x[:,n+1] = mod.lorenz96(x[:,n],random,dt) + dt*dx                     

        #print 'x[:,n+1] at', n+1, 'is', x[:,n+1]

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
    
    ####### Storing SEs after the 1st minimum to take the mean and variance ############
    if n > 200:
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
#for i in range(D/2):  # for 20 vars
#for i in range(D/4):   # for 20-40 vars
#for i in range(D/20):   # for 100 vars
for i in range(D/100):   # for 1000 vars
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
#for k in range(D/4,D/2):  # for 20-40 vars
#for k in range(D/20,D/10):  # for 100 vars
for k in range(D/100,D/50):   # for 1000 vars
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
    

