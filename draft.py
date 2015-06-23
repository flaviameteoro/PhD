import numpy as np
import matplotlib.pyplot as plt
import functions as m
import model as mod

N = 10000
Obs = 100
dt = 0.01

D = 20 
F=8.17

M = 10
tau= 0.1
ntau = tau/dt

observed_vars = range(1)    
L = len(observed_vars) 
H = np.zeros([L,D])       
for i in range(L):
    H[i,observed_vars[i]] = 1.0   

K = 1.e1*np.diag(np.ones([D]))  
Ks = 1.e0*np.diag(np.ones([L*M]))  

pinv_tol = 2.2204e-16
max_pinv_rank = D

xtrue = np.zeros([D,N+1])
xtrue[:,0] = np.random.rand(D)
dx0 = np.random.rand(D)-0.5
x[:,0] = xtrue[:,0] + dx0

nTD = N + (M-1)*nTau
#t = np.zeros([1,nTD])
t = np.dot(dt,list(xrange(nTD)))
for j in range(N):
    force = np.zeros(D)                                 
    xtrue[:,j+1] = mod.lorenz96(xtrue[:,j],force,dt)   
print 'truth created'
