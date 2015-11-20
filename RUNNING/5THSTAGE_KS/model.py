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
     #xmat[:,0] = zz
     qq = l95(zz,dt)
     xx = qq
     #print 'xx1', xx
     
     zz = Xold + 1/2.0*qq
     #xmat[:,1] = zz;
     qq = l95(zz,dt)     
     xx = xx + 2*qq
     #print 'xx2', xx
     
     zz = Xold + 1/2.0*qq
     #xmat[:,2] = zz
     qq = l95(zz,dt)     
     xx = xx + 2*qq
     #print 'xx3', xx
     
     zz = Xold + qq
     #xmat[:,3] = zz
     qq = l95(zz,dt)      
     xx = xx + qq
     #print 'xx4', xx
     x = Xold + xx*(1/6.0)
     
     return x

def l95(x,dt):
    "The actual Lorenz 1996 model."
    N = len(x)
    F=8.17  #5.0117 
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

def dxdt(xx,Jvec,Jlen,dt):
    "Evaluate the variational equation and the model equation together (in tandem)"
    J = Jvec.reshape(Jlen,Jlen)
    
    f, df = l95_J(xx,dt)
    #print 'f shape is', f.shape

    dJ = np.dot(df,J)

    Jacsize = (Jlen)**2
    Jacv = dJ.reshape(Jacsize) 
    Jacvec = Jacv.reshape(Jacsize,1) 
    #print 'Jacv shape is', Jacv.shape

    dxdt = np.concatenate((f,Jacv),axis=0)
    #print dxdt

    return dxdt#, dJ
    #return f, dJ

def l95_J(x,dt):
    "The actual Lorenz 1996 model."
    N = len(x)
    F=8.17  #5.0117 
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

def rk4_end(Xold,dx,dt):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% Runge-Kutta 4 ODE numerical integration scheme      %
    #%   Xold - State variable                             %
    #%   dt - Time-step                                    %
    #%                                                     %
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 
     N=len(Xold)
     xmat = np.zeros([N,4])
 
     zz = Xold 
     #xmat[:,0] = zz
     qq = l95(zz,dt) + dt*dx
     xx = qq
     #print 'xx1', xx
     
     zz = (Xold + 1/2.0*qq) 
     #xmat[:,1] = zz;
     qq = l95(zz,dt) + dt*dx   
     xx = (xx + 2*qq)
     #print 'xx2', xx
     
     zz = (Xold + 1/2.0*qq) 
     #xmat[:,2] = zz
     qq = l95(zz,dt) + dt*dx   
     xx = (xx + 2*qq)
     #print 'xx3', xx
     
     zz = (Xold + qq) 
     #xmat[:,3] = zz
     qq = l95(zz,dt) + dt*dx     
     xx = (xx + qq)
     #print 'xx4', xx
     x = Xold + xx*(1/6.0)
     
     return x

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

def rk4_J3(Jvec,JN,y,dt):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% Runge-Kutta 4 ODE numerical integration scheme      %
    #%   Jvec - Initial Jacobian (vector)                  %
    #%   dt - Time-step                                    %
    #%                                                     %
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
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

     #dkdxT = np.transpose(dkdx)

     #JN=len(Jvec)
     #Jacsize = JN**2
     #Jacvec = Jac.reshape(Jacsize) 

     Jold = Jvec.reshape(JN,JN)

     Jz = Jold
     Jq = dt*(np.dot(dkdx,Jz))
     J = Jq
         
     Jz = Jold + 1/2.0*Jq
     Jq = dt*(np.dot(dkdx,Jz))
     J = J + 2*Jq
     
     Jz = Jold + 1/2.0*Jq
     Jq = dt*(np.dot(dkdx,Jz))
     J = J + 2*Jq

     Jz = Jold + Jq
     Jq = dt*(np.dot(dkdx,Jz))  
     J = J + Jq

     Jac = Jold + J*(1/6.0)

               
     return Jac

def rk4_J4(JvecD,JN,y,dt):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% Runge-Kutta 4 ODE numerical integration scheme      %
    #%   Jvec - Initial Jacobian (vector)                  %
    #%   dt - Time-step                                    %
    #%                                                     %
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
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

     #dkdxT = np.transpose(dkdx)

     #JN=len(Jvec)
     #Jacsize = JN**2
     #Jacvec = Jac.reshape(Jacsize) 

     Jold = JvecD.reshape(JN,JN)
     
     Jold = np.dot(dkdx,Jold)

     Jz = Jold
     Jq = dt*(np.dot(dkdx,Jz))
     J = Jq
         
     Jz = Jold + 1/2.0*Jq
     Jq = dt*(np.dot(dkdx,Jz))
     J = J + 2*Jq
     
     Jz = Jold + 1/2.0*Jq
     Jq = dt*(np.dot(dkdx,Jz))
     J = J + 2*Jq

     Jz = Jold + Jq
     Jq = dt*(np.dot(dkdx,Jz))  
     J = J + Jq

     JacD = Jold + J*(1/6.0)

               
     return JacD

def rk4_J5(JvecD,JN,y,dt):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% Runge-Kutta 4 ODE numerical integration scheme      %
    #%   Jvec - Initial Jacobian (vector)                  %
    #%   dt - Time-step                                    %
    #%                                                     %
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
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

     #dkdxT = np.transpose(dkdx)

     #JN=len(Jvec)
     #Jacsize = JN**2
     #Jacvec = Jac.reshape(Jacsize) 

     Jold = JvecD.reshape(JN,JN)
     
     #Jold = np.dot(dkdx,Jold)

     Jz = Jold
     Jq = dt*(np.dot(dkdx,(np.dot(dkdx,Jz))))
     J = Jq
         
     Jz = Jold + 1/2.0*Jq
     Jq = dt*(np.dot(dkdx,(np.dot(dkdx,Jz))))
     J = J + 2*Jq
     
     Jz = Jold + 1/2.0*Jq
     Jq = dt*(np.dot(dkdx,(np.dot(dkdx,Jz))))
     J = J + 2*Jq

     Jz = Jold + Jq
     Jq = dt*(np.dot(dkdx,(np.dot(dkdx,Jz))))
     J = J + Jq

     JacD = Jold + J*(1/6.0)

               
     return JacD

def rk4_J6(Jvec,JN,y,dt):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #% Runge-Kutta 4 ODE numerical integration scheme      %
    #%   Jvec - Initial Jacobian (vector)                  %
    #%   dt - Time-step                                    %
    #%                                                     %
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
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

     dkdxT = np.transpose(dkdx)

     #JN=len(Jvec)
     #Jacsize = JN**2
     #Jacvec = Jac.reshape(Jacsize) 

     Jold = Jvec.reshape(JN,JN)

     Jz = Jold
     Jq = dt*(np.dot(Jz,dkdxT))
     J = Jq
         
     Jz = Jold + 1/2.0*Jq
     Jq = dt*(np.dot(Jz,dkdxT))
     J = J + 2*Jq
     
     Jz = Jold + 1/2.0*Jq
     Jq = dt*(np.dot(Jz,dkdxT))
     J = J + 2*Jq

     Jz = Jold + Jq
     Jq = dt*(np.dot(Jz,dkdxT))  
     J = J + Jq

     Jacd = Jold + J*(1/6.0)

               
     return Jacd

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

def svd(mat):
    "Function to calculate the Singular Value Decomposition of a matrix."
    U, s, V = np.linalg.svd(mat, full_matrices=True)
    return U, s, V


#-----------------------------------------------------------------------------------------------------------------



