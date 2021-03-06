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
r=18 #for x[:,0] = xtrue[:,0]
#r=37 #for original code 
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
K = 1.e1*np.diag(np.ones([D]))      
Ks = 1.e0*np.diag(np.ones([L*M]))  
K1 = 11.e0*np.diag(np.ones([D]))

######### Setting tolerance and maximum for rank calculations ########
pinv_tol =  (np.finfo(float).eps)
max_pinv_rank = M             

################### Creating truth ###################################
xtrue = np.zeros([D,N+1])
xtrue[:,0] = np.random.rand(D)  
print 'xtrue[:,0]', xtrue[:,0]
dx0 = np.random.rand(D)-0.5     
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
 

H = np.zeros([M,M])
for i in range(M):
    H[i,i] = 1.

HH = np.zeros([M,D*M])
for i in range(M):
    #for j in range(0,D*M,D):
    HH[i,i*D] = 1.
#for l in range(M):
    #print 'HH', HH[l,:]


xx = np.zeros([D,1])      
xxx = np.zeros([D,nTau]) 

Jac = np.zeros([D,D])    

for i in range(D):
    Jac[i,i] = 1.

Jac0 = np.copy(Jac)  

#Trying tridiagonal initialisation of Jac
Jac_tri = np.zeros([D,D])    
it,jt = np.indices(Jac_tri.shape)
Jac_tri[it==jt] = 1.
Jac_tri[it==jt-1] = 0.1
Jac_tri[it==jt+1] = 0.1
#print 'Jac_tri', Jac_tri

#Jac0 = np.copy(Jac_tri)  

I = np.zeros([D,D*M])
for w in range(0,D*M,D):
    for k in range(D):
        I[k,k+w] = 1.
print 'I', I
#I = np.zeros([D,D])    
#for i in range(D):
#    I[i,i] = 1. 

run = 100

dlyaini = x[:,1] - xtrue[:,1]

Pa11 = Jac0

P1HT[:,0:L] = np.transpose(h)
HPHT_KS[0,0] = h[:,0]
################### Main loop ##########################################
for n in range(1,run+1):
    #t = (n-1)*dt

    ###S[:,0] = np.dot(h,x[:,n])   
    ###Y[:,0] = y[:,n]             
    #print 'S', S   
    #print 'Y', Y

    ###****P1HT[:,0:L] = np.transpose(h)
    ###****HPHT_KS[0,0] = h[:,0]

    P1HT[:,0:L] = Pa11[:,0:L]
    HPHT_KS[0,0] = Pa11[0,0]

    xx = x[:,n-1]

    Jac = Jac0
    
    P = {}
    ###****P['00'] = Jac0
    P['00'] = Pa11
    
    for s in range(1,M):
        idxs = s

        for m in range(1,M):
            ii = idxs - m
            iid = idxs + 1

            id1 = str(0)+str(idxs)
            id2 = str(0)+str(idxs-1)
            id21 = str(m-1)+str(idxs)

            id3 = str(iid-1)+str(ii)
            id4 = str(iid-2)+str(ii)

            id5 = str(iid-1)+str(iid-1)
            id6 = str(iid-2)+str(iid-2)
            #print 'Initial id21', id21

            if ii >= 0:
                Jac3 = P[id4]
                
                # Calculating the first row of Ps
                if  m == 1:
                    Jac2 = P[id2]

                    #########################
                    Jac2 = np.transpose(Jac2)
                    #########################

                    # Calculating all elements in the upper part of the diagonal#
                    Jacsize = D**2

                    Jacv2 = Jac2.reshape(Jacsize)       
                    Jacvec2 = Jacv2.reshape(Jacsize,1)  
         
                    Jac2 = mod.rk4_J3(Jacvec2,D,xx,dt)
                
                    Jac2 = np.transpose(Jac2)             
                    P[id1] = Jac2 

                    HPHT_KS[0,idxs] = Jac2[0,0]   

                if m > 1:
                    # Calculating all elements in the upper part of the diagonal#
                    Jacsize = D**2

                    Jacv2 = Jac2.reshape(Jacsize)       
                    Jacvec2 = Jacv2.reshape(Jacsize,1)  
         
                    Jac2 = mod.rk4_J3(Jacvec2,D,xx,dt)

                    P[id21] = Jac2

                    HPHT_KS[m-1,idxs] = Jac2[0,0]   

                # Calculating all elements in the lower part of the diagonal#
                Jacv3 = Jac3.reshape(Jacsize)       
                Jacvec3 = Jacv3.reshape(Jacsize,1)  
          
                Jac3 = mod.rk4_J3(Jacvec3,D,xx,dt) 

                P[id3] = Jac3

                # Calculating all elements in the diagonal#  
                Jacv4 = Jac2.reshape(Jacsize)       

                Jacvec4 = Jacv4.reshape(Jacsize,1)  
                
                Jac4 = mod.rk4_J3(Jacvec4,D,xx,dt)
                
                P[id5] = Jac4  
        
                ################# Constructing HPHT_KS matrix #################
                ###(uses only the first elements of the resulting matrices)####
                HPHT_KS[idxs,ii] = Jac3[0,0]

                HPHT_KS[idxs,idxs] = Jac4[0,0]
        
                #################### Constructing P1HT matrix #################
                if m == 1:                
                    col = Jac2[:,0]

                    P1HT[:,idxs] = col
                
            if m == (M-1):         
                random = np.zeros(D)
                xx = mod.lorenz96(xx,random,dt) 
                
                #print 'h', h.shape
                #print 'xx', xx.shape
                ########### Constructing S and Y vectors #######################   
                S[:,s-1] = np.dot(h,xx)

                Y[:,s-1] = y[:,n+(s-1)]         
    
                if s == 1:
                    xf = xx

        if s == (M-1):            
            random = np.zeros(D)
            xx = mod.lorenz96(xx,random,dt) 

            S[:,M-1] = np.dot(h,xx)
            Y[:,M-1] = y[:,n+s] 

    ########## Calculating the equivalent for KS structure##############
    #### Calculating the inverse of HPHT through SVD ####
    U, G, V = mod.svd(HPHT_KS)          # considering R=0
    print 'G', G

    #if n == run:
    if np.mod(n,100) == 0:
        plt.figure(10).suptitle('Singular values spectrum')
        plt.plot(G,'-') 
        #plt.yscale('log')
        plt.hold(True)

    ######## First and last 3 singular values ###########
    svmax = np.max(G) 

    svmin = np.min(G)
    
    svmin2 = G[M-2]
    
    svmin3 = G[M-3]

    svminlast = G[max_pinv_rank-1]

    ############# Condition number calculation ##########
    condnumber = svmax/svminlast

    ######## Option 1 - Dynamical rank according to tolerance ######
    toler = G/svmax
    #print 'tolerance', toler
    ##toler2 = len(G) * np.linalg.norm(G) * (np.finfo(float).eps)
    toler2 = len(G) * svmax * (np.finfo(float).eps)
    ##print 'toler2', toler2
    
    if np.mod(n,100) == 0:
        plt.figure(12).suptitle('Tolerance spectrum')
        plt.plot(toler,'-') 
        plt.yscale('log')
        plt.hold(True)

    tol = np.ones(len(toler))
    for l in range(len(toler)):
        if toler[l] > 1.e-6:
            tol[l] = 1
        else:
            tol[l] = 0
    max_pinv_rank = sum(tol)
    #print 'tol', tol
    print 'rank is', max_pinv_rank


    ### Option 2 - Dynamical rank according to sing values spectrum ##
    for j in range(len(G)-1):
        svdiff = G[j]-G[j+1]
        #print 'Svdiff', svdiff
        svgrad = abs(svdiff - G[j+1])
        #print 'svgrad', svgrad     
        ###if svgrad < G[j+1]:
        #if svdiff < G[j+1]:
            ###max_pinv_rank = j
            ###break
    ###print 'rank is', max_pinv_rank

    #### Modifying G to use the max_pinv_rank ###########
    mask = np.ones(len(G)) 
    for k in range(len(G)):
        if G[k] >= pinv_tol:
        #if G[k] >= 1.e0:    
            mask[k] = 1
        else:
            mask[k] = 0
        
    rr = min(max_pinv_rank,sum(mask)) 
    #print 'rank', rr
    
    g = G[:rr]**(-1) 
    #print 'g', g
    Ginv = np.zeros((M, M))
    Ginv[:rr, :rr] = np.diag(g)
    #print 'Ginv', Ginv   


    ############# Condition number calculation ##########
    #svminlast = Ginv[max_pinv_rank]     
    #condnumber = svmax/svminlast


    ############## Calculating the inverse ##############
    HPHTinv1 = np.dot((np.transpose(V[:,:])),(np.transpose(Ginv)))   
    
    HPHTinv = np.dot(HPHTinv1,(np.transpose(U[:,:])))  
    
    ##### Multiplying the whole term with (y - hx) ######
    dx1 = np.dot(P1HT,HPHTinv)  
    
    #print 'dx1', dx1.shape 


    ############# Update P matrix #######################
    ### (I - KH) ### 
    IKH = np.zeros([D,D*M])   
    KH = np.dot(dx1,HH)
    ##print 'KH', KH
    IKH = I - KH
    #print 'IKH', IKH.shape  
    #print 'I shape', I.shape

    ### (I - KH)Pf ###
    ### Transforming IKH into a dictionary to interact with P ###
    ikh2 = np.zeros([D,D])
    ikh_dic = {}
    idstr = 0
    for k in range(0,D*M,D):
        for i in range(D):
            for j in range(D):
                ikh2[i,j] = IKH[i,j+k]
        #print 'For k', k, 'ikh2', ikh2
        idk = str(0)+str(idstr)
        ikh_dic[idk] = ikh2 
        idstr = idstr + 1
        ikh2 = np.zeros([D,D])
    ##print 'IKH', IKH    
    ##print 'ikh', ikh_dic['00']                   
    
    ### (I - KH)Pf calculation of first entry of dic - P11 update ###
    #sum_ikh = np.zeros([D,D])
    Pa11 = np.zeros([D,D])
    for d in range(M):    
        str_a = str(0)+str(d)
        str_b = str(d)+str(0)
        Pa11 = Pa11 + np.dot(ikh_dic[str_a],P[str_b])
    #print 'Pa11', Pa11
    #####################################################
    #ddd = Y[:,0] - S[:,0]
    #print 'diff', diff
    #for i in range(M):
    #   diff[:,i] = ddd  
    #print diff
    
    #hx = np.zeros([1,M])     
    #hx[:,0] = ddd
    #print 'hx', hx

    #dys = Y - S
    #print 'Y-S', dys

    #Y[:,0] = ddd
    #print 'Y', Y
      
    dx = np.dot(dx1,np.transpose((Y-S)))
    dx = dx.reshape(D)  
    
    #dx = dx[:,0]

    ############ Running the coupled dynamics ###########
    random = np.zeros(D)

    x[:,n] = xf + dx
    

    ################ Calculating Lyapunov exponent #####################
    dlya = x[:,n] - xtrue[:,n]

    dl = abs(dlya/dlyaini)   

    lya = (1/float(n))*(np.log(dl))

   
    ################ Plotting Lyapunov exponent #####################
    #plt.figure(figsize=(12, 10)).suptitle('Lyapunov Exponents')
    #if [any(item) in lya >0 
    plt.figure(1).suptitle('Lyapunov Exponents for D=20, M='+str(M)+', r='+str(r)+', K='+str(K[0,0])+', max_pinv_rank= '+str(max_pinv_rank)+'')
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
            

    ########################## Calculating SE ############################
    dd = np.zeros([D,1])
    #dd[:,0] = xtrue[:,n+1] - x[:,n+1]
    dd[:,0] = xtrue[:,n] - x[:,n]
    SE = np.sqrt(np.mean(np.square(dd)))            
    ##print '*************************************'
    print 'SE for', n, 'is', SE
    ##print '*************************************'


    ######### Plotting SE and other variables ###########
    #plt.figure(figsize=(12, 10)).suptitle('Synchronisation Error')
    plt.figure(2).suptitle('Synchronisation Error for D=20, M=10, r='+str(r)+', K='+str(K[0,0])+', max_pinv_rank= '+str(max_pinv_rank)+'')
    ###plt.plot(n+1,SE,'b*') 
    plt.plot(n,SE,'b*') 
    plt.yscale('log')
    plt.hold(True)

    ###plt.plot(n+1,svmin,'yo') 
    plt.plot(n,svmin,'yo') 
    plt.hold(True)

    ###plt.plot(n+1,svmax,'go') 
    plt.plot(n,svmax,'go') 
    plt.hold(True)

    ###plt.plot(n+1,svminlast,'bo') 
    plt.plot(n,svminlast,'bo') 
    plt.hold(True)

    ###plt.plot(n+1,condnumber,'c*') 
    plt.plot(n,condnumber,'c*') 
    plt.yscale('log')
    plt.hold(True)

    plt.savefig('SE.png')
    #plt.show()   

 
    ################## Plotting P matrices ##############    
    #cmap=plt.get_cmap('RdBu')
    #plt.figure(3).suptitle('P Matrix')
    ####plt.imshow(P[:nx,:nx],cmap=cmap)
    #if np.mod(n,1000) == 0:
    #    for b in range(M):
    #        for a in range(M):
    #            c = M*b
    #            plt.subplot(M,M,c+a+1)
    #            idx = str(b)+str(a)
    #            plt.imshow(P[idx],cmap=cmap)
                #plt.colorbar()
    #            plt.hold(True)
    #            plt.xlabel('P['+str(idx)+']')
    #            plt.hold(True)
    #        plt.hold(True)
        #plt.colorbar()   
    #    plt.savefig('P_'+str(n)+'.png')
    ##plt.close('all')
#plt.show()
        
    cmap=plt.get_cmap('RdBu')
    plt.figure(4).suptitle('P Matrix at time 20')
    if n == 20:
        #for i in range(2):
        plt.subplot(1,3,1)
        plt.imshow(P['00'],cmap=cmap)
        plt.xlabel('P[00]')
        plt.colorbar()   
        plt.subplot(1,3,2)
        plt.imshow(P['09'],cmap=cmap)
        plt.xlabel('P[09]')
        plt.colorbar()   
        plt.subplot(1,3,3)
        plt.imshow(P['99'],cmap=cmap)
        plt.xlabel('P[99]')
        plt.colorbar()   
        plt.savefig('P_n20.png')
        #plt.show()

    #if np.mod(n,10) == 0:
    if n == run:
        cmap=plt.get_cmap('RdBu')
        plt.figure(5).suptitle('P Matrix at time'+str(n)+'')
    #if np.mod(n,10) == 0:
        #for i in range(2):
        plt.subplot(1,3,1)
        plt.imshow(P['00'],cmap=cmap)
        plt.xlabel('P[00]')
        plt.colorbar()   
        plt.subplot(1,3,2)
        plt.imshow(P['09'],cmap=cmap)
        plt.xlabel('P[09]')
        plt.colorbar()   
        plt.subplot(1,3,3)
        plt.imshow(P['99'],cmap=cmap)
        plt.xlabel('P[99]')
        plt.colorbar()   
        plt.savefig('P_n'+str(n)+'.png')
        #plt.show()

    #cmap=plt.get_cmap('RdBu')
    #plt.figure(6).suptitle('P - 1st row - 00-09 at time 10')
    #if n == 10:
    #    for i in range(M):
    #        plt.subplot(2,M/2.,i+1)
    #        idxx = str(0)+str(i)
    #        plt.imshow(P[idxx],cmap=cmap)
    #        plt.xlabel('P['+str(idxx)+']')
    #        plt.colorbar()   
    #        plt.savefig('P1_n10.png')
    
    #cmap=plt.get_cmap('RdBu')
    #plt.figure(7).suptitle('P - 1st row - 00-09 at time'+str(n)+'')
    #if n == run:
    #    for i in range(M):
    #        plt.subplot(2,M/2.,i+1)
    #        idxx = str(0)+str(i)
    #        plt.imshow(P[idxx],cmap=cmap)
    #        plt.xlabel('P['+str(idxx)+']')
    #        plt.colorbar()   
    #        plt.savefig('P1_n'+str(n)+'.png')

######################## Plotting variables ###############################
plt.figure(figsize=(12, 10)).suptitle('Variables for D=20, M='+str(M)+', r='+str(r)+', K='+str(K[0,0])+', max_pinv_rank= '+str(max_pinv_rank)+'')
for i in range(D/2):
    #plt.subplot(np.ceil(D/8.0),2,i+1)
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
plt.show()
    

