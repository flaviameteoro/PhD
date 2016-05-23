#from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

########## Defining variables ##########
#r = np.zeros([1,2]) # seeds
#L = np.zeros([1,2]) # number of obs
TD5_1 = np.zeros([1,7]) # time-delays

############# Defining variables and values ##################
r = [18, 37]
L = [5, 10]
TD = [3, 4, 5, 6, 7, 8, 9, 10]
TD5_1 = ([4, 5, 6, 7, 8, 9, 10])
TD5_2 = [5, 6, 7, 8, 9, 10]

sigma2 = 0.01
threshold = np.sqrt(sigma2)*5

rlen = len(r)
Llen = len(L)

TDlen5_1 = len(TD) - 1
TDlen5_2 = len(TD) - 2
TDlen10 = len(TD)

predic_ranges_synch_L5 = np.zeros([rlen,TDlen5_1]) # prediction ranges for synch code
predic_ranges_synch_L10 = np.zeros([rlen,TDlen10]) # prediction ranges for synch code

predic_ranges_sherman_L5 = np.zeros([rlen,TDlen5_2]) # prediction ranges for sherman code
predic_ranges_sherman_L10 = np.zeros([rlen,TDlen10]) # prediction ranges for sherman code

############# Synch values ################
predic_ranges_synch_L5[0,:] = [109, 194, 191, 164, 207, 220, 326]
predic_ranges_synch_L5[1,:] = [146, 262, 289, 267, 277, 296, 254]

predic_ranges_synch_L10[0,:] = [114, 181, 213, 199, 197, 246, 231, 281]
predic_ranges_synch_L10[1,:] = [229, 336, 389, 364, 306, 300, 335, 385]

############# Sherman values ################
predic_ranges_sherman_L5[0,:] = [113, 191, 192, 275, 189, 192]
predic_ranges_sherman_L5[1,:] = [105, 120, 152, 135, 164, 162]

predic_ranges_sherman_L10[0,:] = [42, 133, 157, 216, 135, 204, 208, 213]
predic_ranges_sherman_L10[1,:] = [62, 101, 248, 168, 233, 255, 322, 265]


################ Plotting ###################

######## For L=5 / r=18 ##########
#Ltitle = L[:,0]
#rtitle = r[:,0]
#plt.figure(figsize=(1, 10)).suptitle('Prediction ranges in time steps (L = '+str(L[0])+', r = '+str(r[0])+', max #RMSE:'+str(threshold)+')')

plt.figure(1)
plt.title('Prediction ranges in time steps (L = '+str(L[0])+', r = '+str(r[0])+', max RMSE:'+str(threshold)+')')
#for i in range(rlen):
#    plt.plot(TD5_1,predic_ranges_synch_L5[i,:],'b*') 

plt.plot(TD5_1,predic_ranges_synch_L5[0,:],'b*-')     
plt.hold(True)
plt.plot(TD5_2,predic_ranges_sherman_L5[0,:],'m*-') 
plt.hold(True)

plt.legend(['synch','sherman'], loc='upper left')
plt.xlabel('Time delays')
plt.ylabel('Time steps')

######## For L=5 / r=37 ##########
#plt.figure(figsize=(2, 10)).suptitle('Prediction ranges in time steps (max RMSE:'+str(threshold)+')')

plt.figure(2)
plt.title('Prediction ranges in time steps (L = '+str(L[0])+', r = '+str(r[1])+', max RMSE:'+str(threshold)+')')

plt.plot(TD5_1,predic_ranges_synch_L5[1,:],'b*-')     
plt.hold(True)
plt.plot(TD5_2,predic_ranges_sherman_L5[1,:],'m*-') 
plt.hold(True)

plt.legend(['synch','sherman'], loc='upper left')
plt.xlabel('Time delays')
plt.ylabel('Time steps')

######## For L=10 / r=18 ##########
#plt.figure(figsize=(2, 10)).suptitle('Prediction ranges in time steps (max RMSE:'+str(threshold)+')')

plt.figure(3)
plt.title('Prediction ranges in time steps (L = '+str(L[1])+', r = '+str(r[0])+', max RMSE:'+str(threshold)+')')

plt.plot(TD,predic_ranges_synch_L10[0,:],'b*-')     
plt.hold(True)
plt.plot(TD,predic_ranges_sherman_L10[0,:],'m*-') 
plt.hold(True)

plt.legend(['synch','sherman'], loc='upper left')
plt.xlabel('Time delays')
plt.ylabel('Time steps')

######## For L=10 / r=37 ##########
#plt.figure(figsize=(2, 10)).suptitle('Prediction ranges in time steps (max RMSE:'+str(threshold)+')')

plt.figure(4)
plt.title('Prediction ranges in time steps (L = '+str(L[1])+', r = '+str(r[1])+', max RMSE:'+str(threshold)+')')

plt.plot(TD,predic_ranges_synch_L10[1,:],'b*-')     
plt.hold(True)
plt.plot(TD,predic_ranges_sherman_L10[1,:],'m*-') 
plt.hold(True)

plt.legend(['synch','sherman'], loc='upper left')
plt.xlabel('Time delays')
plt.ylabel('Time steps')

plt.show()
