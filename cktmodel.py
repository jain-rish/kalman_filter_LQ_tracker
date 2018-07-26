#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on Tue June 1 09:56:09 2018

Circuit model
@author: Rishabh Jain
"""

import numpy as np
import ode
import pandas as pd
import matplotlib.pyplot as plt



tStart  = 1.0
tStop   = 200.0


def system_dyn(t, vectorState, **kwargs):
    # evaluate state dynamics with guilleman model
    iL_out = (1/kwargs['L'])*(-vectorState[1] + kwargs['k']*np.log(vectorState[2])+ kwargs['V0'])
    Vc_out = (1/kwargs['C'])*(vectorState[0] + kwargs['Vu']/kwargs['Rs'] -vectorState[1]/kwargs['Rs'])
    qB_out = -vectorState[0]
    
    vector_out= np.array([iL_out, Vc_out, qB_out])
    return vector_out



####INPUTS
    
# init values of the elements
C=  10e-1
L=  5e-1

Vu= 100
V0= 200

k= 10


# init states
x0 = np.array([-10, 80., 500])



# numerical constants
outputNum = 1
integrateTol = 1e-6
integrateMaxIter = 5000


# controls
uTimes = np.arange(tStart, tStop, 4)
uTimes = np.append(uTimes,tStop)
duty_vals = np.random.rand(len(uTimes), 1)
# convert controls to dynamic resistor values
Rsmax= 10e+3
Rsmin= 10
dyn_res = duty_vals*Rsmax + (1-duty_vals)*Rsmin

uControl = pd.DataFrame(dyn_res, index= uTimes)
uControl.index.names = ['Time']


# noise parameters
mu = 0.
sigma = 0.2

# 2D list to record the state, as well as the time array
list_Result = []
vector_out=  x0




#### Run the simulation

tStart_i= tStart
while tStart_i < tStop - 1e-12:
    
    vectorStateStart= x0
    
    for i in range(len(uTimes)-1): # iteration i: the time interval of two adjacent observation
        list_Result.append([]) 
        tStop_i = uTimes[i+1]        
       
        # evenly divide the time interval between two adjacent observation 
        #     in order to output filtered state values at those time points
        stepSize = (tStop_i-tStart_i)/outputNum
        tStart_j = tStart_i   
        # initiate variable for counting total rows of integration output of the jth iteration
        sumCountRows = 0
        
        #print ("Var. checking: tStart_i={}, tStop_i={} sec. ".format(tStart_i, tStop_i))
        
        for j in range(outputNum):  
            # setup time and initiate dictionary for recording the results of the jth iteration
            state_j = {"Time" : [], "countRows": 0, "BattState" : []} 
            initFlag = True
            tStop_j = tStart_j + stepSize
            tbegin = tStart_j
            
            while tbegin < tStop_j - 1e-12: # 1e-12: give some tolerance for numerical purpose
                # pull control data from uControl
                #       - note:  set it to be backward constant, i.e., find rows of the first 
                #                time index > tStart_j, for numerical purpose, add epsilon as tolerance
                uTimeIndex = uControl[uControl.index > tbegin+1e-10].index[0]
                uControlData = uControl.loc[uTimeIndex]
                tEnd = min(uTimeIndex, tStop_j)
                
                #print ("Var. checking: tbegin={}, tStop_j={} sec. ".format(tbegin, tStop_j, tEnd))
                    
                # add noise (varying to each element?)
                ns = np.random.normal(mu, sigma, len(vectorStateStart))
                vectorStateStart = vectorStateStart + ns
                
                # call the ode integrator ...
                fn= system_dyn
                vector_out, t, failFlag, iter_i = \
                    ode.ode_rk23(fn, tbegin, tEnd, vectorStateStart, 
                                 integrateTol, integrateMaxIter, C=C, L=L, k=k, V0=V0, Vu= Vu, Rs=np.float(uControlData))
                
                print ("integration happening from {} - {} sec. with ctrl at {}".format(tbegin, tEnd, uTimeIndex))
                
                # records results with nicer format, 
                #   note: do not record the starting point since it is given, not computed
                countRows = len(t)
                sumCountRows = sumCountRows+countRows-1
                if initFlag:
                    state_j['Time'] = t[1:countRows]                
                    state_j['State'] = vector_out[1:countRows,:]
                    initFlag = False
                else:
                    state_j['Time'] = np.append(state_j['Time'],t[1:countRows])
                    state_j['State'] = np.append(state_j['State'], vector_out[1:countRows,:], axis=0)
                
                # update tStart and the start point, i.e. vectorStateStart
                tbegin = tEnd
                vectorStateStart = vector_out[-1]
                
            # add results of jth iteration to the list, and update start time of j loop
            list_Result[i].append(state_j)
            tStart_j = tStop_j
        # update start time of i loop
        tStart_i = tStop_i
    




# construct results for output - evenly divide the time interval by outputNum 
#                                between two adjacent observation and output 
#                                filtered state and covariance values at those time points 
modelTimeIndex = np.array([tStart])
modelState = np.array([x0])


for m in range(len(list_Result)):
    for n in range(len(list_Result[m])):
        modelTimeIndex = np.append(modelTimeIndex, list_Result[m][n]['Time'][-1])
        modelState = np.append(modelState, np.array([list_Result[m][n]['State'][-1]]), axis=0)
    

# construct time sequence of the battery model state over time interval [t0, t1] 
model = {"timeIndex" : modelTimeIndex, \
             "iL" : modelState[:,0], \
             "Vc" : modelState[:,1],\
             "Q"  : modelState[:,2]}
# make a pandas dataframe
df_model = pd.DataFrame(data= model)



# do some graphics
plt.plot(model['timeIndex'], model['iL'], 'o-')
plt.plot(model['timeIndex'], model['Vc'], 'o-')
plt.plot(model['timeIndex'], model['Q'], 'o-')

#plt.plot(uTimes, uControl/10, 'o--')
plt.gca().legend(('iL','Vc','Q'))

plt.show()