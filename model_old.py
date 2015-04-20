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
    F=8
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

#-----------------------------------------------------------------------------------------------------------------


#Lorenz 96 tangent linear model-----------------------------------------------------------------------------------
def lorenz96_tl(old,linear,randomErr,dt):
    """This function computes the time evolution of the tangent linear Lorenz 96 model."""

    # Initialize values for integration
    N = len(old)
    new = np.zeros(N)
    
    help = rk4tl(old,linear,dt)
    new = help + randomErr
    
    return new
    
def rk4tl(dx,x,dt):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% Runge-Kutta 4 ODE numerical integration scheme (Tangent-linear) %
    #%  dx - Perturbed state                                           %
    #%   x - Linearisation state                                       %
    #%  dt - Time-step                                                 % 
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    help, xmat = rk4(x,dt)
    
    zz = dx
    qq = l95tl(xmat[:,0],zz,dt)
    xx = qq
            
    zz = dx + qq/2
    qq = l95tl(xmat[:,1],zz,dt)
    xx = xx + 2*qq
            
    zz = dx + qq/2
    qq = l95tl(xmat[:,2],zz,dt)
    xx = xx + 2*qq
            
    zz = dx + qq
    qq = l95tl(xmat[:,3],zz,dt)
    xx = xx + qq
            
    dx = dx + xx*(1./6)
    
    return dx

def l95tl(x,dx,dt):
    "The actual tangent linear Lorenz 1996 model."

    N = len(x)     #Number of variables automatically set by user
    
    dy=np.empty_like(x)
    dy.fill(np.nan)
 
    for i in range(N):
        #Periodic (cyclic) Boundary Conditions    
        ip1 = i+1
        if ip1>N-1: ip1 = ip1-N
        im1 = i-1
        if im1<0: im1 = im1+N
        im2 = i-2
        if im2<0: im2 = im2+N
        #RHS of L95 TL
        dy[i] = dt*(- dx[im2]*x[im1] + dx[im1]*(x[ip1] - x[im2]) - dx[i] + dx[ip1]*x[im1])
        
    return dy
#-----------------------------------------------------------------------------------------------------------------


#Lorenz 96 adjoint model------------------------------------------------------------------------------------------
def lorenz96_adj(old,linear,randomErr,dt):
    """This function computes the adjoint of the tangent linear Lorenz 96 model."""

    # Initialize values for integration
    N = len(old)
    new = np.zeros(N)
    
    help = rk4adj(old,linear,dt)
    new = help + randomErr
    
    return new
    
def rk4adj(dx,x,dt):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% Runge-Kutta 4 ODE numerical integration scheme (Adjoint version)  %
    #%   dx - Perturbed state variable                                   %
    #%    x - Linearisation state variable                               %
    #%   dt - Time-step                                                  %
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    help, xmat = rk4(x,dt)
    
    xx = dx*(1./6)
    
    qq = xx
    zz = l95adj(xmat[:,3],qq,dt)
    dx = dx + zz
    qq = zz
    
    qq = qq + 2*xx
    zz = l95adj(xmat[:,2],qq,dt);
    dx = dx + zz
    qq = 0.5*zz
        
    qq = qq + 2*xx
    zz = l95adj(xmat[:,1],qq,dt)
    dx = dx + zz
    qq = 0.5*zz
        
    qq = qq + xx
    zz = l95adj(xmat[:,0],qq,dt)
    dx = dx + zz
 
    return dx

def l95adj(x,dy,dt):
     
    N=len(x)
    dz=np.empty_like(x)
    dz.fill(np.nan)
 
    for i in range(N):
        #Periodic (cyclic) Boundary Conditions    
        ip1 = i+1
        if ip1>N-1: ip1 = ip1-N
        ip2 = i+2
        if ip2>N-1: ip2 = ip2-N  
        im1 = i-1
        if im1<0: im1 = im1+N
        im2 = i-2
        if im2<0: im2 = im2+N
        #RHS of L95 ADJOINT
        dz[i] = dt*(dy[im1]*x[im2] - dy[i] + dy[ip1]*(x[ip2] - x[im1]) - dy[ip2]*x[ip1])
        
    return dz
#-----------------------------------------------------------------------------------------------------------------    
 
