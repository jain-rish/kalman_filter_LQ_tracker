#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 09:22:12 2018

State estimator for Edge Controller 
"""

import pandas as pd
import numpy as np
import ode

def kalman_filter_continuous(t0, t1, x0, stateCov0, stateDim, controlDim, observeDim, uControl, 
                             yObserve, dynParA, dynParB, dynParF, observeParH, dynCovW, 
                             observeCovV, outputNum, integrateTol, integrateMaxIter):
    '''
    %--------------------------------------------------------------------------
    %  Purpose  : implement Kalman filter continuous version for the following 
    %             model over the time interval [t0,t1] 
    %
    %                 State Dynamics: dx(t)/dt=A*x(t)+B*u(t)+f+Omega(t)
    %                 Observation   : y(t) = H*x(t)+Theta(t)
    %
    %             where
    %                 x(t) is a vector of state variable
    %                 u(t) is a known time sequence of control vector
    %                 y(t) is a a known time sequence of observation vector
    %                 Omega(t) is a Gaussian noise vector with zero mean and covariance matrix W
    %                 Theta(t) is a Gaussian noise vector with zero mean and covariance matrix V
    %                 A, B, f are parameters of the state dynamics model
    %                 H is the matrix of observation model
    %
    %             The Kalman filter continuous version of the model:
    %                 dynamics of the state conditional mean -
    %                     dx(t)/dt=A*x(t)+B*u(t)+f+K(t)(y(t)-H*x(t))
    %                   with
    %                     x(t0) = x0
    %                 dynamics of the state covariance
    %                     dp(t)/dt=A*p(t)+p(t)*transpose(A)-K(t)*H*p(t)+W
    %                   with
    %                     p(t0) = stateCov0
    %                 where K(t)=p(t)*transpose(H)*inv(V) is Kalman gain 
    %
    %   
    %  Inputs:
    %   t0, t1          :  starting and ending time of the Kalman filter model
    %                      type - float
    %   x0              :  initial value of conditional mean of the state vector at time t0
    %                      type - 1D numpy.ndarray, dimension: stateDim x stateDim
    %   stateCov0       :  initial value of covariance of the state vector at time t0, which is positive definite
    %                      type - 1D numpy.ndarray, dimension: stateDim x 1
    %   uControl        :  a time sequence of piecewise constant control signal over time [t0,t1]
    %                      type - pandas dataframe with 'Time' to be indext, and 
    %                             the vector of control signal to be data
    %                      dimension: (k_u) x controlDim, where k_u is the number of piecewise constant
    %                      Note: (1) the time of the first data point should be t0,
    %                                and the time of the last data point should be t1
    %                            (2) the piecewise constant of a given time segment will 
    %                                take the control signal value at the end of the time segment
    %   yObserve        :  a time sequence of observation signal over time [t0,t1]
    %                      type - pandas dataframe with 'Time' to be indext, and 
    %                             the vector of observation signal to be data
    %                      dimension: (k_y) x observeDim, where k_y is the number of observations
    %                      Note: (1) the time of the first data point should be t0,
    %                                and the time of the last data point should be t1
    %                            (2) the piecewise constant of a given time segment will 
    %                                take the observation signal value at the end of the time segment
    %   stateDim        :  dimension of state vector 
    %                      type - integer > 0
    %   controlDim      :  dimension of control signal 
    %                      type - integer > 0, note: controlDim <= stateDim
    %   observeDim      :  dimension of observation signal 
    %                      type - integer > 0, note: observeDim <= stateDim
    %   dynParA         :  parameter of the dynamics dx/dt 
    %                      type - 2D numpy.ndarray, dimension: stateDim x stateDim
    %   dynParB         :  parameter of the dynamics dx/dt 
    %                      type - 2D numpy.ndarray, dimension: stateDim x controlDim
    %   dynParF         :  parameter of the dynamics dx/dt 
    %                      type - 1D numpy.ndarray, dimension: stateDim  x 1
    %   observeParH     :  parameter of the observation model
    %                      type - 2D numpy.ndarray, dimension: observeDim x stateDim
    %   dynCovW         :  covariance matrix of the dynamics noise, which is positive definite
    %                      type - 2D numpy.ndarray, dimension: stateDim x stateDim
    %   observeCovV     :  covariance matrix of the observation noise, which is positive definite 
    %                      type - 2D numpy.ndarray, dimension: observeDim x observeDim
    %   outputNum       :  the number of time points evenly dividing each time interval
    %                      defined by two adjasent observation signals, and at each 
    %                      time point the conditional mean and its covariance of the state signal are recorded
    %                      type - integer > 0
    %   integrateTol    :  numerical integration process local error tolerance for the solution
    %                      type - float > 0
    %   integrateMaxIter:  maximal iterations allowed for executing nemerical integration
    %                      type - integer > 0
    %    
    %   (TO DO: may need to add input for controlling integration method)
    %
    %  Output:
    %   kfcStateAndCov   :  a dictionary stores Kalman filtered state vector and state covariance matix with keys  
    %                           "timeIndex", "kfcState" and "kfcStateCov" 
    %                      where, 
    %                            "timeIndex"  - a sequence of time points at which the filtered state 
    %                                           and its covariance are available for the time interval [t0,t1]
    %                                   type  - numpy array with the dimension (k_y*outputNum) x 1
    %                            "kfcState"    - a time sequence of the filtered state vector for the time interval [t0,t1]
    %                                   type  - a (k_y*outputNum) x 1 list with each element being a numpy array which records 
    %                                           values of a state vector with dimension stateDim x 1 
    %                            "kfcStateCov" - a (k_y*outputNum) x 1 list with each element being a numpy array which records
    %                                           values of the state covariance matrix with dimension stateDim x stateDim 
    %                      Note: the time of the first data point should be t0, the time of the last data point should be t1
    '''   
    # To do: need to check dimensions of the input, and raise error if dimensions are not match
    
    # To do: need to check initial time and terminal time of uControl and yObserve (i.e. =t0, and =t1 respectively)

    # To do: need to add defaul integrateTol, integrateMaxIter    

    # To do: need to raise exception if integration is fail
    listFilterResult = kfc_propagate_state_and_cov(t0, t1, x0, stateCov0, stateDim, 
                                                   controlDim, observeDim, uControl, 
                                                   yObserve, dynParA, dynParB, dynParF, 
                                                   observeParH, dynCovW, observeCovV, 
                                                   outputNum, integrateTol, integrateMaxIter)
                        
 
    # construct results for output - evenly divide the time interval by outputNum 
    #                                between two adjacent observation and output 
    #                                filtered state and covariance values at those time points 
    kfcTimeIndex = np.array([t0])
    kfcState = np.array([x0])
    kfcStateCov = np.array([stateCov0])
    for i in range(len(listFilterResult)):
        for j in range(len(listFilterResult[i])):
            kfcTimeIndex = np.append(kfcTimeIndex, listFilterResult[i][j]['Time'][-1])
            kfcState = np.append(kfcState, np.array([listFilterResult[i][j]['vectorState'][-1]]), axis=0)
            kfcStateCov = np.append(kfcStateCov, np.array([listFilterResult[i][j]['matrixCov'][-1]]), axis=0)
   
    ### construct time sequence of Kalman filter state over time interval [t0, t1] 
    kfcStateAndCov = {"timeIndex" : kfcTimeIndex, "kfcState" : kfcState, "kfcStateCov" : kfcStateCov}
  
    
    return kfcStateAndCov


def kfc_propagate_state_and_cov(t0, t1, x0, stateCov0, stateDim, controlDim, observeDim, 
                                uControl, yObserve, dynParA, dynParB, dynParF, observeParH, dynCovW, 
                                observeCovV, outputNum, integrateTol, integrateMaxIter):
    '''                    
    # propagate state and state covariance for continuous Kalman Filter by forward integration 
    #   (note: propagate state and covariance of the state simultaneously) 
    '''
    #### define the inital condition of covariance and state
    ####   note: need to reshape covariance as a vector, and combine covariance vector and state vector as one vector
    covVectorDim = stateDim*stateDim
    vectorCovStateStart = construct_cov_state_vector(stateCov0, x0, covVectorDim)    
    
    ### solve integration eqs of state and covariance of the state simultaneously
    ###    note: (1) evenly divide the time interval between two adjacent observation 
    ###              and output filtered state values at those time points
    ###          (2) time index of uControl yObserve may not align each other, 
    ###              therefore the time interval of two adjacent time points given by (1) 
    ###              may be split into multiple segments which each has different uControl data
    listFilterResult = [] # it will be a 2D list to record the filtered state and state covariance as well as the time array
    tStart_i = t0
    for i in range(len(yObserve)-1): # iteration i: the time interval of two adjacent observation
        listFilterResult.append([]) 
        tEnd_i = yObserve.index[i+1]        
        # pull observation data from yObserve for the time interval [tStart_i, tEnd_i], 
        #      note:  set it to be backward constant, i.e. observation data = yObserve(tEnd_i)
        yObserveData = np.array(yObserve.iloc[i+1])        
        # evenly divide the time interval between two adjacent observation 
        #     in order to output filtered state values at those time points
        stepSize = (tEnd_i-tStart_i)/outputNum
        tStart_j = tStart_i   
        # initiate variable for counting total rows of integration output of the jth iteration
        sumCountRows = 0
        
        for j in range(outputNum):  
            # setup time and initiate dictionary for recording the results of the jth iteration
            stateAndCov_j = {"Time" : [], "countRows": 0, "vectorState" : [], "matrixCov" : []} 
            initFlag = True
            tEnd_j = tStart_j + stepSize
            tStart = tStart_j
            
            while tStart < tEnd_j - 1e-12: # 1e-12: give some tolerance for numerical purpose
                # pull control data from uControl
                #       - note:  set it to be backward constant, i.e., find rows of the first 
                #                time index > tStart_j, for numerical purpose, add epsilon as tolerance
                uTimeIndex = uControl[uControl.index > tStart+1e-10].index[0]
                uControlData = np.array(uControl.loc[uTimeIndex])
                tEnd = min(uTimeIndex, tEnd_j)
                
                # To do: need to raise exception when integration is fail
                vectorCovState, t, failFlag, iter_i = \
                    ode.ode_rk23(continuous_kalman_filter_dyn, tStart, tEnd, vectorCovStateStart, 
                                 integrateTol, integrateMaxIter, dynParA = dynParA, dynParB=dynParB, 
                                 dynParF = dynParF, observeParH=observeParH, uControlData=uControlData,
                                 yObserveData=yObserveData, dynCovW=dynCovW, observeCovV=observeCovV,                             
                                 covVectorDim = covVectorDim, stateDim = stateDim)
                    
                # records results with nicer format, 
                #   note: do not record the starting point since it is given, not computed
                countRows = len(t)
                sumCountRows = sumCountRows+countRows-1
                if initFlag:
                    stateAndCov_j['Time'] = t[1:countRows]                
                    stateAndCov_j['vectorState'] = vectorCovState[1:countRows,-stateDim:]
                    stateAndCov_j['matrixCov'] = vectorCovState[1:countRows,0:covVectorDim].reshape((countRows-1, stateDim,stateDim))
                    initFlag = False
                else:
                    stateAndCov_j['Time'] = np.append(stateAndCov_j['Time'],t[1:countRows])
                
                    stateAndCov_j['vectorState'] = np.append(stateAndCov_j['vectorState'],
                                 vectorCovState[1:countRows,-stateDim:], axis=0)
                    stateAndCov_j['matrixCov'] = np.append(stateAndCov_j['matrixCov'],
                                 vectorCovState[1:countRows,0:covVectorDim].reshape((countRows-1, stateDim,stateDim)), axis=0)
                
                # update tStart and the start point, i.e. vectorCovStateStart
                tStart = tEnd
                vectorCovStateStart = vectorCovState[-1]
                
            # add results of jth iteration to the list, and update start time of j loop
            listFilterResult[i].append(stateAndCov_j)
            tStart_j = tEnd_j
        # update start time of i loop
        tStart_i = tEnd_i
        
    return listFilterResult


def construct_cov_state_vector(matrixCov, vectorState, covVectorDim):
    '''
    #reshape covariance matrix as a vector (i.e. row by row), and append state vector to the end of 
    #the covariance vector as one vector
    # inputs: matrixCov - nxn numpy 2D array
    #         vectorState - nx1 numpy 1D array
    #         covVectorDim - integer with value=n*n
    # output: vectorCovState - (n*n+n)x1 2D array with first n*n entries are 
    #                           elements of matrixCov (row-byrow), and the last
    #                           n entries are elements vectorState
    '''
    vectorCovState = np.append(matrixCov.reshape((covVectorDim,)),vectorState)
    return vectorCovState

def continuous_kalman_filter_dyn(t, vectorCovState, **kwargs):
    # evaluate state dynamics and covariance dynamics for continuous Kalman filter model
    ### separate vectorCovState into vectorCov and vectorState
    vectorState = vectorCovState[-kwargs['stateDim']:]
    vectorCov = vectorCovState[0:kwargs['covVectorDim']]
    ### reshape vectorCov to stateDim x stateDim 2D numpy array
    matrixCov = vectorCov.reshape((kwargs['stateDim'],kwargs['stateDim']))
    ### compute Kalman gain
    hByCov = kwargs['observeParH'].dot(matrixCov)
    kGain = (np.transpose(hByCov)).dot(np.linalg.inv(kwargs['observeCovV']))
    ### compute values of Kalman filter covariance dynamics, i.e. matrixCovDot
    parAByCov = kwargs['dynParA'].dot(matrixCov)
    matrixCovDot = parAByCov+np.transpose(parAByCov)-kGain.dot(hByCov)+kwargs['dynCovW']
    ### compute values of Kalman filter state dynamics, i.e. vectorStateDot
    vectorStateDot = kwargs['dynParA'].dot(vectorState)+\
                    kwargs['dynParB'].dot(kwargs['uControlData'])+kwargs['dynParF']+\
                    kGain.dot(kwargs['yObserveData']-kwargs['observeParH'].dot(vectorState))
    ### reshape matrixSigmaDot as a vector, and make sigma_dot vector and vectorPhiDot as one vector
    vectorCovStateDot =construct_cov_state_vector(matrixCovDot, vectorStateDot, kwargs['covVectorDim'])
    return vectorCovStateDot