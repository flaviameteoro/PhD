# File containing all the functions used by mcmc comparison
import numpy as np
#from scipy import linalg as spl
import model as Mnl

#Resampling for the filter----------------------------------------------------------------------------------------
def resample(ensemble,weight):

    #calculate the actual weights of each particle
    weight = -weight
    maxwp = np.max(weight)
    Nsamp=len(ensemble[:,0])
    Ndim=len(ensemble[0,:])

    for m in range(Nsamp):
        weight[m] = np.exp(weight[m]-maxwp)
    
    #Normalize weights
    wsum=np.sum(weight)
    weight = weight/wsum

    effSamp = 1./(np.dot(weight,weight))
       
    #Stochastic universal resampling
    wpnew = np.zeros(Nsamp)
    newEns = np.zeros([Nsamp,Ndim])
    wpnew[0] = weight[0]
    for m in range(Nsamp-1):
        wpnew[m+1] = wpnew[m] + weight[m+1]
    rr = np.random.rand(1)/Nsamp
    nn=0
    for m in range(Nsamp):
        while (rr > wpnew[nn]):
            nn=nn+1
        newEns[m,:] = ensemble[nn,:]
        rr = rr + 0.9999/Nsamp

    return newEns, effSamp
#----------------------------------------------------------------------------------------------------------------


#SIR filter------------------------------------------------------------------------------------------------------
def sir(initial,Usir,MO,NM,scov_bg,scov_model,H,y,tobs,pinv_obs,F,dt):
    
    #Sizes
    Nsir = len(Usir[:,0,0])
    N = len(Usir[0,:,0])
    J = len(Usir[0,0,:])-1
    weights = np.zeros(Nsir)
    eff = np.zeros(NM)

    mcn = 0

    for t in range(J):

        for m in range(Nsir):
        
            if (t == 0):
            
                #random = np.random.randn(N)
                #force = np.dot(scov_bg,random)
                #Usir[m,:,t] = initial + force
                Usir[m,:,t] = initial[m,:]

            random = np.random.randn(N)
            force = np.dot(scov_model,random)
            Usir[m,:,t+1] = Mnl.lorenz96(Usir[m,:,t],force,dt)


        if (t+1 == tobs[mcn]): 

            print tobs[mcn]

            sum = 0
            for m in range(Nsir):

                diff = y[:,mcn]-np.dot(H,Usir[m,:,tobs[mcn]])
                hulp = np.dot(pinv_obs,diff)
                weights[m] = 1./2*np.dot(np.transpose(diff),hulp)
                sum = sum + weights[m]*weights[m]

            mcn = mcn+1

            Usir[:,:,t+1], eff = resample(Usir[:,:,t+1],weights)

    return Usir, eff

#----------------------------------------------------------------------------------------------------------------


#ewpf-equivalent-weights particle filter-------------------------------------------------------------------------
def ewpf(initial,Uew,MO,NM,scov_bg,scov_model,cov_model,pinv_model,H,y,tobs,cov_obs,pinv_obs,nudgeFac,F,dt):
    
    #Sizes
    New = len(Uew[:,0,0])
    N = len(Uew[0,:,0])
    J = len(Uew[0,0,:])-1
    ns = J/NM

    #Factors
    retain = 5
    epsilon = 0.001/New
    gamma_u = 1e-2
    gamma_n = 1e-5

    #Arrays
    weights = np.zeros(New) #
    extraWeight = np.zeros(New) #additional weights due to mixture density
    wpp = np.zeros(New) #storing nudge weights
    c = np.zeros(New) #storing maximum weights
    eff = np.zeros(NM)

    #Matrices needed
    HQHT = np.dot(H,np.dot(cov_model,np.transpose(H)))
    pinv_QN =  np.linalg.pinv(HQHT + cov_obs)
    kgain = np.dot(cov_model,np.dot(np.transpose(H),pinv_QN))

    mcn = 0

    for t in range(J):

        for m in range(New):
        
            if (t == 0):
            
                #random = np.random.randn(N)
                #force = np.dot(scov_bg,random)
                #Uew[m,:,t] = initial + force
                Uew[m,:,t] = initial[m,:]

            if(t+1 == tobs[mcn]):
  
                force = 0.
                Uew[m,:,t+1] = Mnl.lorenz96(Uew[m,:,t],force,dt)

            else:

                #nudge
                tau = (t-mcn*ns)/float(ns)
                diff = y[:,mcn]-np.dot(H,Uew[m,:,t])
                help1 = np.dot(pinv_obs,diff)
                help2 = np.dot(np.transpose(H),help1)
                help3 = np.dot(cov_model,help2)
                nudge = nudgeFac*tau*help3
                #random error
                random = np.random.randn(N)
                force = np.dot(scov_model,random)
                
                nudge = force + nudge
                Uew[m,:,t+1] = Mnl.lorenz96(Uew[m,:,t],nudge,dt)

                weight = np.dot(nudge,np.dot(pinv_model,nudge))-np.dot(force,np.dot(pinv_model,force))
                wpp[m] = wpp[m] + 0.5*weight
            
        #print wpp

        if (t+1 == tobs[mcn]):

            print tobs[mcn]

            #Calculate weights for best fit to observations for each member
            for m in range(New):
                diff = y[:,mcn]-np.dot(H,Uew[m,:,tobs[mcn]])
                c[m] = wpp[m] + 0.5*np.dot(diff,np.dot(pinv_QN,diff))

            ccsort = np.sort(c)
            #print ccsort
            cc = ccsort[(retain*New/10)-1]
      
            oldU = np.copy(Uew[:,:,tobs[mcn]])
            for m in range(New):
                if (c[m] <= cc):
                    diff = y[:,mcn]-np.dot(H,Uew[m,:,tobs[mcn]])
                    help = np.dot(H,np.dot(kgain,diff))
                    aaa = 0.5 * np.dot(diff,np.dot(pinv_obs,help))
                    bbb = 0.5 * np.dot(diff,np.dot(pinv_obs,diff))- cc + wpp[m]
                    alpha = 1 - np.sqrt(1. - bbb/aaa + 0.00000001)

                    Uew[m,:,tobs[mcn]] = Uew[m,:,tobs[mcn]] + alpha*np.dot(kgain,diff)

                #Add random error from mixture density
                u=np.random.rand(1)
                factor = epsilon/(1-epsilon)*(2/np.pi)**(N/2)*(gamma_u**N/gamma_n)
                if (u<epsilon):
                    random = gamma_n*np.random.randn(N)
                    #print 'random', random
                    addRandom = np.dot(scov_model,random)
                    extraWeight[m] = 1./(factor*np.exp(-(1./(2*gamma_n**2))*np.dot(random,random))) 
                else:
                    random=2*gamma_u**(1./2)*(np.random.rand(N)-0.5)
                    #print 'random', random
                    addRandom = np.dot(scov_model,random)
                    extraWeight[m] = 1./(1+factor*np.exp(-(1./(2*gamma_n**2))*np.dot(random,random)))

                Uew[m,:,tobs[mcn]] = Uew[m,:,tobs[mcn]] + addRandom
                    
            #Calculate final weights
            for m in range(New):
                diff = y[:,mcn]-np.dot(H,Uew[m,:,tobs[mcn]])
                xtest = Uew[m,:,tobs[mcn]]-oldU[m,:]
                help = 0.5*np.dot(diff,np.dot(pinv_obs,diff)) + 0.5*np.dot(xtest,np.dot(pinv_model,xtest))
                weights[m] = help + wpp[m] - extraWeight[m]
            print weights
      
            mcn = mcn+1

            Uew[:,:,t+1], eff = resample(Uew[:,:,t+1],weights)
            print 'Uew', Uew
            wpp[:]=0.

    return Uew, eff
#----------------------------------------------------------------------------------------------------------------
