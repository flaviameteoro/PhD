# File containing all the functions used by mcmc comparison
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as spl

#Lorenz 96 model--------------------------------------------------------------------------------------------------
def lorenz96(old,randomErr,dt):
    """This function computes the time evolution of the Lorenz 96 model.

    It is the general case for N variables; often N=40 is used.

    The Lorenz 1996 model is cyclical: dx[j]/dt=(x[j+1]-x[j-2])*x[j-1]-x[j]+F

    Inputs:  - old, original position.  
             - dt, the timestep 0.025 (the time step 1/40 (to guarantee stability with RK4))
    Outputs: - new, the new position"""

    # Initialize values for integration
    N = len(old)
    new = np.zeros(N)
    
    help, dummy = rk4(old,dt)
    new = help + randomErr

    return new

## Functions for the integration
def rk4(Xold,dt):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% Runge-Kutta 4 ODE numerical integration scheme      %
    #%   Xold - State variable                             %
    #%   dt - Time-step                                    %
    #%                                                     %
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 
     N=len(Xold)
     xmat = np.zeros([N,4])
 
     zz = Xold
     xmat[:,0] = zz
     qq = l95(zz,dt)
     xx = qq
    
     zz = Xold + 1/2.0*qq
     xmat[:,1] = zz;
     qq = l95(zz,dt)     
     xx = xx + 2*qq
     
     zz = Xold + 1/2.0*qq
     xmat[:,2] = zz
     qq = l95(zz,dt)     
     xx = xx + 2*qq
     
     zz = Xold + qq
     xmat[:,3] = zz
     qq = l95(zz,dt)      
     xx = xx + qq
     
     x = Xold + xx*(1/6.0)
     
     return x, xmat

def l95(x,dt):
    "The actual Lorenz 1996 model."
    N = len(x)
    F=8.17
    k=np.empty_like(x)
    k.fill(np.nan)
    
    for i in range(N): 
        #Periodic (cyclic) Boundary Conditions    
        ip1 = i+1
        if ip1>N-1: ip1 = ip1-N
        im1 = i-1
        if im1<0: im1 = im1+N
        im2 = i-2
        if im2<0: im2 = im2+N
        #RHS of L95
        k[i] = dt*(-x[im2]*x[im1] + x[im1]*x[ip1] - x[i] + F)

    return k


def df(y):                      
    "Function to find dF/dx."
    N = len(y)
    dkdx = np.zeros([N,N])
    for i in range(N):
        #for j in range(N):
        ip1 = i+1
        if ip1>N-1: ip1 = ip1-N
        im1 = i-1
        if im1<0: im1 = im1+N
        im2 = i-2
        if im2<0: im2 = im2+N
        dkdx[i,i] = -1
        dkdx[i,ip1] = y[im1]
        dkdx[i,im1] = y[ip1] - y[im2]
        dkdx[i,im2] = -y[im1]
    return dkdx


def svd(s):
    "Function to calculate the Singular Value Decomposition of a matrix."
    U, s, V = np.linalg.svd(s, full_matrices=True)
    return U, s, V

#-----------------------------------------------------------------------------------------------------------------



