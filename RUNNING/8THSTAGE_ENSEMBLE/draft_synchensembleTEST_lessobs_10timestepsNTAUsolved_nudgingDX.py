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
N = 2000#2300
Obs = 100
dt = 0.01    
fc = 1150#1900#2190

D = 1000 
F=8.17

M = 5
tau= 0.1
nTau = tau/dt
print 'D=', D, 'variables and M=', M ,'time-delays'

Nens = 30 # ensemble size 


###################### Seeding for 20 variables########################
#r=37 #for original codes 

#r=18 #for x[:,0] = xtrue[:,0]
#r=44  #for RK4 and 0.0005 uniform noise (for M = 10)
#r=39   #for RK4 and 0.0005 uniform noise (for M = 12)
#r=1
#r=10
#r=50
r=100
#r=24
#r=87

np.random.seed(r)  


#################### Constructing h (obs operator) ##################
observed_vars = range(250)    
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
#print 'K', K.shape

K_new = 1.0e2*np.diag(np.ones([Nens]))      #Increased to 100 to prevent dx to be divided by a factor of 10!!!! (what made Localisation work) # For the 5th formulation 
gtau1 = 0.3#1.0#0.9#2./3.        For increasing nudging = 0.1, for decreasing = 0.9         # Factor to be applied to the pseudoinv used in the 1st unobserved time step after an observation 
gtau2 = 1.0#0.7#1./3.                                                                       # Factor to be applied to the pseudoinv used in the 2nd unobserved time step after an observation 

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
#print 'xtrue[:,0]', xtrue[:,0]

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
#print 'x[:,0]', x[:,0]


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
#y = np.dot(h,xtrue) + np.random.normal(0,0.09,N+1)
sigma2 = 0.01
#y = np.dot(h,xtrue) + np.random.uniform(0,0.02,N+1)-0.01
#y = np.dot(h,xtrue) + np.random.uniform(0,0.01,N+1)-0.005
###(for seed=18)
#y = np.dot(h,xtrue) + np.random.normal(0,0.01,N+1)  

###### Noise that runs perfect until time step 1800 and 2300 (for seed=18, K=40, max_rank=7) 
#y = np.dot(h,xtrue) + np.random.uniform(0,0.002,N+1)-0.001
#y = np.dot(h,xtrue) + np.random.uniform(0,0.02,N+1)-0.01
#y = np.dot(h,xtrue) + np.random.normal(0,0.001,N+1)  

################################################################################
######### Implementing less obs, according to time steps #######################
################################################################################
################ ONLY at each other time step (odd time steps)##################
##for i in range(1,N+1): 
##    if np.mod(i,2) == 0:    
##        y[:,i] = 0 

#print 'y', y
#print 'xtrue', xtrue

####################### Obs after 2 time steps #################################
#obsgap = 3 #CHANGE LINES 286 AND 353 FOR THIS CASE!     ### obs at each 3 time steps: 1,4,7 etc
obsgap = 10 #CHANGE LINES 285 AND 354 FOR THIS CASE!     ### obs at each 10 time steps: 1,11,21 etc
lastobs = 1
for i in range(2,N+1): 
    diff_obs = i-lastobs
    if diff_obs < obsgap:    
        y[:,i] = 0 
    else:
        lastobs = i

#print 'y', y[0,0:20]

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

run = 1099

oo = np.zeros([1,run+1])         #for observability calculation
svmaxvec = np.zeros([1,run+1]) 
svmaxvec2 = np.zeros([1,run+1]) 

E = np.zeros([D])                ### ensemble mean ###
E_even = np.zeros([D])           ### ensemble mean for even obs time steps###
E_post = np.zeros([D])           ### ensemble mean for each after 2 obs time steps###
A = np.zeros([D,Nens])           ### for the covariance matrix ###
B = np.zeros([D,Nens])           ### for the covariance matrix ###
C = np.zeros([D,D])              ### covariance matrix ###   

loc = 3.
dloc = np.zeros(M*L)
dYS = np.zeros(M*L)
newdYS = np.zeros([1,M*L])

dlyaini = x[:,1] - xtrue[:,1]
dlyaini2 = (dlyaini)**2
########################### Main loop ##########################################

for n in range(1,run+1):
    ##### Running Ensynch only at odd time steps (when we have obs) #########
    #if np.mod(n,2) == 1:
    
    ##### Running Ensynch only at every 2 time steps (when we have obs) #####
    if np.all(y[:,n] != 0):
        #print 'Running Ensynch for time step', n

        lastnobs = n

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
        dsdx[0:L,:] = h4                ## Probably for all cases, included after the 4th formulation!!    
        #print 'dsdx', dsdx
      
        ###########Jac = Jac0
        #print 'Jac', Jac


        for m in range(2,M+1):
            ####### Propagate the ensemble members forward "tau" times #########
            for i in range(1,int(nTau)+1):          #FOR OBSGAP=10!!!!
            #**for i in range(1,int(nTau)):         #FOR OBSGAP=3!!!!!
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


             ########## Constructing S and Y vectors ########################   
            idxs = L*(m-1)        
            #print 'idxs', idxs        
            S[:,idxs:idxs+L] = np.dot(h,E)        
            #print 'S at m', m, 'is', S
            #idy = n+(m-1)*nTau

            ##### Arrange Y to get obs for the case of each 2 time steps!! #####
            #Y[:,idxs:idxs+L] = y[0:L,n+(m-1)*nTau-(m-1)]   ##FOR OBSGAP=3!!!!!
            Y[:,idxs:idxs+L] = y[0:L,n+(m-1)*nTau]          ##FOR OBSGAP=10!!!!

            #print 'Y at m', m, 'is', Y

            ################ Constructing dsdx ##############################
            # For 1st and 2nd working formulations:
            ##dsdx[idxs:idxs+L,:] = np.dot(h,Jac)

            # For 4th working formulation:
            dsdx[idxs:idxs+L,:] = np.dot(h,B)  
            #print 'dsdx', dsdx

            
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

        
        #### Calculating the coupling term in coupled dynamics ######
        #dx1 = np.dot(K,dxds)             # For the 1st and 2nd working formulations
        #dx1 = np.dot(K,dxds_new)         # For the 4th working formulation (and the 3rd NOT working one)
        dx1 = np.dot(K_new,dxds)          # For the 5th working formulations
        #print 'dx1', dx1

        ######################## Implementing LOCALISATION #########################
        ##### SUGGESTION of OPTIMISATION:dloc is the same during the whole run, so you can calculate it just once #####
        #############################################################################
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

                dist1 = min(np.abs(i-l),D-np.abs(i-l))
                dist = (dist1)**2


                if (dist1 > (3*loc)):
                    dloc[j] = 0.
                else:
                    dloc[j] = np.exp(-(dist/(2*(loc**2))))
                #print 'dloc', dloc
            #print 'dloc for var',i, ':', dloc
            
            for k in range(M*L):
                i_dYS = dYS[:,k]

                i_newdYS = dloc[k]*i_dYS[0]            # SCHUR Product!! 
                newdYS[:,k] = i_newdYS
            #print 'New dYS to use in the variable equation to find var', i, ':', dYS

            dx3 = np.dot(dx2[i,:],np.transpose(newdYS)) 
            #print 'dx3 for var',i, ':', dx3

            DX[i] = dx3[0] 
 
        #### Evolving variables with time #######
        random = np.zeros(D)
        #x[:,n+1] = mod.rk4_end(x[:,n],dx,dt) 
        x[:,n+1] = mod.rk4_end(x[:,n],DX,dt) 
                                   
 
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
            ##plt.figure(5)
            ##plt.plot(n+1,SEmean,'bo')
            #plt.yscale('log')
            ##plt.hold(True)
                
            ##plt.figure(6)
            ##plt.plot(n+1,SEvar,'rx')
            #plt.yscale('log')
            ##plt.hold(True)
            #print 'SEvar', SEvar

        ############  Lyapunov exponents ########################       
        dlya = x[:,n+1] - xtrue[:,n+1]
        dlya2 = (dlya)**2
        #print 'dlya', dlya
        #print 'dlya2', dlya2

        
        dl = abs(dlya/dlyaini)   
        leaddl = (np.sqrt(np.sum(dlya2))/np.sqrt(np.sum(dlyaini2)))
        #print 'dl', dl
        lya = (1/float(n))*(np.log(dl))
        leadinglya = (1/float(n))*(np.log(leaddl))
        #print 'Leading Lyapunov exponent', leadinglya

        ####### PLOTTING THE LEADING (AND CORRECT) Lyapunov exponents #######
#        fig = plt.figure(20)
#        ax = fig.add_subplot(1, 1, 1) # create an axes object in the figure
#        if leadinglya > 0:
#            plt.plot(n,leadinglya,'rx') 
#        if leadinglya == 0:
#            plt.plot(n,leadinglya,'gx') 
#        if leadinglya < 0:
#            plt.plot(n,leadinglya,'yx') 


#        plt.title('Leading Lyapunov exponents',fontsize=15)
#        plt.xlabel('Time units',fontsize=15)        
    
#        for tick in ax.xaxis.get_ticklabels():
#            tick.set_fontsize(15)
            #tick.set_fontsize('large')
            #tick.set_fontname('Times New Roman')
            #tick.set_color('blue')
            #tick.set_weight('bold')
#        for tick in ax.yaxis.get_ticklabels():
#            tick.set_fontsize(15)

    elif (n - lastnobs == 1):
        #print 'Using last dxds for time step', n
       
        newDX1 = gtau1*DX
        #newDX1 = gtau2*DX

        #print 'DX is', DX 
        #print 'newDX is', newDX1         
        x[:,n+1] = mod.rk4_end(x[:,n],newDX1,dt)

  
        ####### Calculating the synchronisation error (RMSE) #######
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
        
        ############  Lyapunov exponents ########################       
        dlya = x[:,n+1] - xtrue[:,n+1]
        dlya2 = (dlya)**2
        #print 'dlya', dlya
        #print 'dlya2', dlya2

        
        dl = abs(dlya/dlyaini)   
        leaddl = (np.sqrt(np.sum(dlya2))/np.sqrt(np.sum(dlyaini2)))
        #print 'dl', dl
        lya = (1/float(n))*(np.log(dl))
        leadinglya = (1/float(n))*(np.log(leaddl))
        #print 'Leading Lyapunov exponent', leadinglya

        ####### PLOTTING THE LEADING (AND CORRECT) Lyapunov exponents #######
#        fig = plt.figure(20)
#        ax = fig.add_subplot(1, 1, 1) # create an axes object in the figure
#        if leadinglya > 0:
#            plt.plot(n,leadinglya,'rx') 
#        if leadinglya == 0:
#            plt.plot(n,leadinglya,'gx') 
#        if leadinglya < 0:
#            plt.plot(n,leadinglya,'yx') 


#        plt.title('Leading Lyapunov exponents',fontsize=15)
#        plt.xlabel('Time units',fontsize=15)        
    
#        for tick in ax.xaxis.get_ticklabels():
#            tick.set_fontsize(15)
            #tick.set_fontsize('large')
            #tick.set_fontname('Times New Roman')
            #tick.set_color('blue')
            #tick.set_weight('bold')
#        for tick in ax.yaxis.get_ticklabels():
#            tick.set_fontsize(15)

        #gtau_count = 1    # For increasing nudging
        gtau_count = 0

        #gtau_count = 0     # For decreasing nudging

    else:
        #gtau_count = gtau_count + 0.5    #BAD
        #gtau_count = gtau_count + 1.5     #BAD
        #gtau_count = gtau_count + 0.9
        #gtau_count = gtau_count + 1.1
        gtau_count = gtau_count + 1                #WORKING BETTER     
        #print 'gtau_count for', n, '=', gtau_count 
        
        #if (n - lastnobs == 2):        
        #    newDX2 = gtau2*DX
        #    print 'This is time step', n, 'and DX will be reused again.'
        #else:
        newDX2 = gtau_count*gtau1*DX    # For increasing nudging  #WORKING BETTER

        #gfac = gtau1-(gtau_count*0.1)    # For decreasing nudging
        #print 'gfac for', n, '=', gfac   
        #newDX2 = gfac*DX                 # For decreasing nudging

        #newDX2 = gtau2*DX
        #print 'DX is', DX 
        #print 'newDX is', newDX2         
        x[:,n+1] = mod.rk4_end(x[:,n],newDX2,dt)


        ####### Calculating the synchronisation error (RMSE) #######
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
            ##plt.figure(5)
            ##plt.plot(n+1,SEmean,'bo')
            #plt.yscale('log')
            ##plt.hold(True)
                
            ##plt.figure(6)
            ##plt.plot(n+1,SEvar,'rx')
            #plt.yscale('log')
            ##plt.hold(True)
            #print 'SEvar', SEvar

        ############ Lyapunov exponents ########################       
        dlya = x[:,n+1] - xtrue[:,n+1]
        dlya2 = (dlya)**2
        #print 'dlya', dlya
        #print 'dlya2', dlya2

        
        dl = abs(dlya/dlyaini)   
        leaddl = (np.sqrt(np.sum(dlya2))/np.sqrt(np.sum(dlyaini2)))
        #print 'dl', dl
        lya = (1/float(n))*(np.log(dl))
        leadinglya = (1/float(n))*(np.log(leaddl))
        #print 'Leading Lyapunov exponent', leadinglya

        ####### PLOTTING THE LEADING (AND CORRECT) Lyapunov exponents #######
#        fig = plt.figure(20)
#        ax = fig.add_subplot(1, 1, 1) # create an axes object in the figure
#        if leadinglya > 0:
#            plt.plot(n,leadinglya,'rx') 
#        if leadinglya == 0:
#            plt.plot(n,leadinglya,'gx') 
#        if leadinglya < 0:
#            plt.plot(n,leadinglya,'yx') 


#        plt.title('Leading Lyapunov exponents',fontsize=15)
#        plt.xlabel('Time units',fontsize=15)        
    
#        for tick in ax.xaxis.get_ticklabels():
#            tick.set_fontsize(15)
            #tick.set_fontsize('large')
            #tick.set_fontname('Times New Roman')
            #tick.set_color('blue')
            #tick.set_weight('bold')
#        for tick in ax.yaxis.get_ticklabels():
#            tick.set_fontsize(15)

    
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

plt.savefig('2testgtau.png')
#plt.show()

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
#for i in range(D/10):   # for 100 vars
for i in range(D/100):   # for 1000 vars
#    plt.subplot(np.ceil(D/8.0),2,i+1)
    plt.subplot(5,2,i+1)
    ################## Plotting the observations ###########################
    i_obs = (D/L)
    if np.mod(i,i_obs) == 0:   
        i_y = i/i_obs
        plt.plot(y[i_y,:],'r.',label='obs')   ## create y with no zeros to plot correctly ###
        plt.hold(True)

    ##################### Plotting the variables ###########################    
           
    plt.plot(x[i,:],'g',label='X')
    plt.hold(True)
    #plt.plot(xtrue[i,:],'b-',linewidth=2.0,label='truth')
    #plt.plot(xtrue[i,:],'b-',linewidth=1.0,label='truth')
    plt.plot(xtrue[i,:],'b',label='truth')
    plt.hold(True)
        
    plt.ylabel('x['+str(i)+']')
    plt.xlabel('time steps')

plt.savefig('2testvargtau1.png')

plt.figure(figsize=(12, 10)).suptitle('Variables for D='+str(D)+', M='+str(M)+', r='+str(r)+', K='+str(K[0,0])+', max_pinv_rank= '+str(max_pinv_rank)+'')
#for k in range(D/2,D):  # for 20 vars
#for k in range(D/4,D/2):  # for 20-40 vars
#for k in range(D/10,D/5):  # for 100 vars
for k in range(D/100,D/50):   # for 1000 vars
#for i in range(D/3):
#for i in range(D):
    #plt.subplot(np.ceil(D/8.0),2,i+1)
    i2 = k - 10
    plt.subplot(5,2,i2+1)
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
    #plt.plot(xtrue[k,:],'b-',linewidth=2.0,label='truth')
    #plt.plot(xtrue[k,:],'b-',linewidth=1.0,label='truth')
    plt.plot(xtrue[k,:],'b',label='truth')
    plt.hold(True)
        
    plt.ylabel('x['+str(k)+']')
    plt.xlabel('time steps')

plt.savefig('2testvargtau2.png')
#plt.show()
    

