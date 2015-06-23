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
    
    help = rk4(old,dt)
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
     #print 'xx1', xx
     
     zz = Xold + 1/2.0*qq
     xmat[:,1] = zz;
     qq = l95(zz,dt)     
     xx = (xx + 2*qq)
     #print 'xx2', xx
     
     zz = Xold + 1/2.0*qq
     xmat[:,2] = zz
     qq = l95(zz,dt)     
     xx = (xx + 2*qq)
     #print 'xx3', xx
     
     zz = Xold + qq
     xmat[:,3] = zz
     qq = l95(zz,dt)      
     xx = (xx + qq)
     #print 'xx4', xx
     x = Xold + xx*(1/6.0)
     
     return x

def dxdt(xx,Jvec,Jlen,dt):
    "Evaluate the variational equation and the model equation together (in tandem)"
    J = Jvec.reshape(Jlen,Jlen)
    
    f, df = l95_J(xx,dt)
    #print 'f shape is', f.shape

    dJ = np.dot(df,J)

    Jacsize = (Jlen)**2
    Jacv = dJ.reshape(Jacsize) 
    #Jacvec = Jacv.reshape(Jacsize,1) 
    #print 'Jacv shape is', Jacv.shape

    dxdt = np.concatenate((f,Jacv),axis=0)

    return dxdt

def l95(x,dt):
    "The actual Lorenz 1996 model."
    N = len(x)
    F=8.17
    k=np.empty_like(x)
    #k.fill(np.nan)
    
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

def rk4_J(Xold,Jacv,dt):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% Runge-Kutta 4 ODE numerical integration scheme      %
    #%   Xold - State variable                             %
    #%   dt - Time-step                                    %
    #%                                                     %
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 
     N=len(Xold)
     #Jacsize = N**2
     #Jacvec = Jac.reshape(Jacsize) 
     Jac = Jacv.reshape(N,N)

     zz = Xold
     Jz = Jac
     #xnew,dfdx = l95(zz,Jz,dt)
     xnew,dfdx = l95_J(zz,dt)
     Jq = np.dot(dfdx,Jz)
     xx = xnew
     JJ = Jq
    
     zz = Xold + 1/2.0*xnew
     Jz = Jac + 1/2.0*Jq
     xnew,dfdx = l95_J(zz,dt)
     Jq = np.dot(dfdx,Jz)
     xx = xx + 2*xnew
     JJ = JJ + 2*Jq

     zz = Xold + 1/2.0*xnew
     Jz = Jac + 1/2.0*Jq
     xnew,dfdx = l95_J(zz,dt)
     Jq = np.dot(dfdx,Jz)
     xx = xx + 2*xnew
     JJ = JJ + 2*Jq

     zz = Xold + xnew
     Jz = Jac + Jq
     xnew,dfdx = l95_J(zz,dt)
     Jq = np.dot(dfdx,Jz)   
     xx = xx + xnew
     JJ = JJ + Jq

     #x = Xold + xx*(1/6.0)
     J = Jac + JJ*(1/6.0)
     
     return J

def rk4_J2(Xold,df,dt):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% Runge-Kutta 4 ODE numerical integration scheme      %
    #%   Xold - State variable                             %
    #%   dt - Time-step                                    %
    #%                                                     %
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 
     N=len(Xold)
     #Jacsize = N**2
     #Jacvec = Jac.reshape(Jacsize) 
     #Jac = Jacv.reshape(N,N)

     zz = Xold
     fz = df
     #xnew,dfdx = l95(zz,Jz,dt)
     xnew,dfdx = l95_J(zz,dt)
     #Jq = np.dot(dfdx,Jz)
     xx = xnew
     ff = dfdx
    
     zz = Xold + 1/2.0*xnew
     fz = df + 1/2.0*ff
     xnew,dfdx = l95_J(zz,dt)
     #Jq = np.dot(dfdx,Jz)
     xx = xx + 2*xnew
     ff = ff + 2*dfdx

     zz = Xold + 1/2.0*xnew
     fz = df + 1/2.0*ff
     xnew,dfdx = l95_J(zz,dt)
     #Jq = np.dot(dfdx,Jz)
     xx = xx + 2*xnew
     ff = ff + 2*dfdx

     zz = Xold + xnew
     fz = df + ff
     xnew,dfdx = l95_J(zz,dt)
     #Jq = np.dot(dfdx,Jz)   
     xx = xx + xnew
     ff = ff + dfdx

     #x = Xold + xx*(1/6.0)
     dfdx = df + ff*(1/6.0)
     
     return dfdx

def l95_J(x,dt):
    "The actual Lorenz 1996 model."
    N = len(x)
    F=8.17
    k=np.empty_like(x)
    #k.fill(np.nan)
    
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

    #N = len(x)
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
        dkdx[i,ip1] = k[im1]
        dkdx[i,im1] = k[ip1] - k[im2]
        dkdx[i,im2] = -k[im1]

    #dksize = N**2
    #dkdxvec = dkdx.reshape(dksize)

    return k, dkdx    


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

def dfk(y):                      
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

    dksize = N**2
    dkdxvec = dkdx.reshape(dksize)

    return dkdxvec

def svd(s):
    "Function to calculate the Singular Value Decomposition of a matrix."
    U, s, V = np.linalg.svd(s, full_matrices=True)
    return U, s, V


#-----------------------------------------------------------------------------------------------------------------



