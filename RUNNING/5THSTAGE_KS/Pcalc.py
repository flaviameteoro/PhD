#from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
#import functions as m
import model as mod

def Pmatrix(x,Jac0):
    #################### Initial settings ################################
    N = 10000
    Obs = 100
    dt = 0.01    #original value=0.01

    D = 20 
    F=8.17

    M = 40
   
    ##################### Seeding for 20 variables#######################
    r=37 #for x[:,0] = xtrue[:,0]
    np.random.seed(r)  

    #################### Constructing h (obs operator) ##################
    observed_vars = range(1)    
    L = len(observed_vars) 
    h = np.zeros([L,D])       
    for i in range(L):
        h[i,observed_vars[i]] = 1.0   

    xx = np.zeros([D,1])      
    
    Jac = np.zeros([D,D])    

    for i in range(D):
        Jac[i,i] = 1.

    #Jac0 = np.copy(Jac)  

    run = 1
    ################### Main loop ##########################################
    for n in range(1,run+1):
        xx = x
    
        #Jac = Jac0
        P = {}
        P['00'] = Jac0
    
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
                      
                    if m > 1:
                        # Calculating all elements in the upper part of the diagonal#
                        Jacsize = D**2

                        Jacv2 = Jac2.reshape(Jacsize)       
                        Jacvec2 = Jacv2.reshape(Jacsize,1)  
         
                        Jac2 = mod.rk4_J3(Jacvec2,D,xx,dt)
                
                        P[id21] = Jac2
                    
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
               
                    if m == (M-1):         
                        random = np.zeros(D)
                        xx = mod.lorenz96(xx,random,dt) 
               
    return P