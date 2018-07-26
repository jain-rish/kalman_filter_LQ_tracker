#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 08:16:43 2018

Collection of different variation of LQ tracker
"""
import pandas as pd
import numpy as np
import ode
  
def lqtracker_pid(t0, t1, x0, u0, stateDim, controlDim, signalTrack, dynParA, 
                  dynParB, dynParF, trackParQ0, trackParQ1, trackParR0, trackParR1, 
                  trackParF0, trackParF1, outputNum, integrateTol, integrateMaxIter): 
    '''
    %--------------------------------------------------------------------------
    %  Purpose  : Compute the optimal feedback control of the following 
    %             PID version of LQ tracking problem for the time interval [t0,t1] 
    %
    %                 min (integrate from t0 to t1)CriterionFunction(x(t),x_dot(t),u(t),u_dot(t))dt+(terminal_term)
    %                 s.t. dx/dt=A*x(t)+B*u(t)+f
    %                      dx_dot/dt=A*x_dot(t)+B*u_dot(t)
    %                      du/dt = v(t)
    %
    %             with
    %                   x(t0) = x0
    %                   u(t0) = u0
    %                   x(t0)_dot = A*x0+B*u0+f
    %
    %             where
    %                 CriterionFunction(x(t),x_dot(t),u(t),u_dot(t))=
    %                    0.5*{Transpose(x(t)-zx(t))*Q0*(x(t)-z(t))+Transpose(x_dot(t)-zx_dot(t))*Q1*(x_dot(t)-z_dot(t))
    %                            +Transpose(u(t)-zu(t))*R0*(u(t)-z(t))+transpose(u_dot(t))*R1*u_dot(t)}
    %                 terminal_term = 0.5*Transpose(x(t1)-z(t1))*F0*(x(t1)-z(t1))+0.5*Transpose(x_dot(t1)-z_dot(t1))*F1*(x_dot(t1)-z_dot(t1))
    %                 x(t) is a vector of state variable
    %                 u(t) is a vector of control variable
    %                 v(t) is a vector of changing rate of control variable
    %                 z(t) is a vector of given tracking signal
    %                 A, B, f, Q0, Q1, R0, R1, F0, F1 are parameters of the LQ tracking problem
    %
    %   
    %  Inputs:
    %   t0, t1          :  starting and ending time which define the time interval of
    %                      the LQ tracking problem defined on
    %                      type - float
    %   x0              :  initial value of state vector at time t0
    %                      type - 1D numpy.ndarray, dimension: stateDim x 1
    %   u0              :  initial value of control vector at time t0
    %                      type - 1D numpy.ndarray, dimension: controlDim x 1
    %   stateDim        :  dimension of state vector 
    %                      type - integer > 0
    %   controlDim      :  dimension of control vector 
    %                      type - integer > 0, note: controlDim <= stateDim
    %   signalTrack     :  dictionary stores tracking signals with keys "timeIndex", 
    %                           "zTrack", "zDotTrack" and "uTrack" 
    %                      where, 
    %                            "timeIndex" - a sequence of time points at which tracking signal is available for the time interval [t0,t1]
    %                                   type - numpy array with the dimension (k_z) x 1
    %                            "zTrack"    - piecewise constant tracking signal of the state vector for the time interval [t0,t1]
    %                                   type - numpy array with the dimension (k_z) x stateDim
    %                            "zDotTrack" - piecewise constant tracking signal of the state changing rate vector for the time interval [t0,t1] 
    %                                   type - numpy array with the dimension (k_z) x stateDim
    %                            "uTrack"    - piecewise constant tracking signal of the control vector for the time interval [t0,t1] 
    %                                   type - numpy array with the dimension (k_z) x controlDim
    %                      Note: the time of the first data point should be t0, the time of the last data point should be t1
    %                            the tracking signals at time t1 is used for terminal condition of LQ tracking
    %   dynParA         :  parameter of the dynamics dx/dt 
    %                      type - 2D numpy.ndarray, dimension: stateDim x stateDim
    %   dynParB         :  parameter of the dynamics dx/dt 
    %                      type - 2D numpy.ndarray, dimension: stateDim x controlDim
    %   dynParF         :  parameter of the dynamics dx/dt 
    %                      type - 1D numpy.ndarray, dimension: stateDim  x 1
    %   trackParQ0      :  LQ tracking weight matrix for state vector
    %                      type - 2D numpy.ndarray diagnal matrix with positive 
    %                           elements, dimension: stateDim x stateDim 
    %   trackParQ1      :  LQ tracking weight matrix for state vector changing rate (i.e. dx/dt)
    %                      type - 2D numpy.ndarray diagnal matrix with positive 
    %                           elements, dimension: stateDim x stateDim 
    %   trackParR0      :  LQ tracking weight matrix for control vector 
    %                      type - 2D numpy.ndarray diagnal matrix with positive 
    %                           elements, dimension: controlDim x controlDim
    %   trackParR1      :  LQ tracking weight matrix for control vector changing rate (i.e. du/dt)
    %                      type - 2D numpy.ndarray diagnal matrix with positive 
    %                           elements, dimension: controlDim x controlDim
    %   trackParF0      :  LQ tracking weight matrix for terminal term of the state vector
    %                      type - 2D numpy.ndarray diagnal matrix with positive 
    %                           elements, dimension: stateDim x stateDim 
    %   trackParF1      :  LQ tracking weight matrix for terminal term of the state vector changing rate (i.e.dx/dt)
    %                      type - 2D numpy.ndarray diagnal matrix with positive 
    %                           elements, dimension: stateDim x stateDim 
    %   outputNum       :  the number of time points evenly dividing each time interval
    %                      defined by two adjasent tracking signal zTrack, and at each 
    %                      time point the state and control signal are recorded
    %                      type - integer > 0
    %   integrateTol    :  numerical integration process local error tolerance for the solution
    %                      type - float > 0
    %   integrateMaxIter:  maximal iterations allowed for executing nemerical integration
    %                      type - integer > 0
    %    
    %   (TO DO: may need to add input for controlling integration method)
    %
    %  Output:
    %   lqtOptimalControl:  collection of optimal control vector at a sequence of time 
    %                       points between t0 and t1 
    %                       type - pandas dataframe with 'Time' to be indext and 
    %                              the vector of control signal at each time to be data, data dimension (k_z*outputNum)xn
    %   lqtOptimalState  :  collection of optimal state vector at a sequence of time 
    %                       points between t0 and t1 
    %                       type - pandas dataframe with 'Time' to be indext and 
    %                              the vector of sate signal at each time to be data, data dimension (k_z*outputNum)xn
    %
    '''   
    # To do: need to check dimensions of the input, and raise error if dimensions are not match
    
    # To do: need to check initial time and terminal time of zTrack (i.e. =t0, and =t1 respectively)

    # To do: need to add defaul integrateTol, integrateMaxIter

    
    # reform LQT PID controller problem into regular LQT problem which can be solved by lqtracker_basic()
    ### constrcut initial condition of state vector for reformed problem, i.e. tildeX0 = [x0, x0Dot, u0]
    x0Dot = dynParA.dot(x0) + dynParB.dot(u0) + dynParF
    tildeX0 = np.append(np.append(x0, x0Dot),u0)
    tildeStateDim = len(tildeX0)    
    ### construct tracking signals for reformed problem, i.e. tildeZtrack = [zTrack, zDotTrack, uTrack]
    tildeZTrack = pd.DataFrame(np.append(np.append(signalTrack['zTrack'], 
                                                   signalTrack['zDotTrack'],axis=1),
                                                   signalTrack['uTrack'],axis=1), 
                                                   index=signalTrack['timeIndex'])        
    ### construct parameters of the state dynamics for reformed problem, i.e. tildeDynParA, 
    ###     tildeDynParB, tildeDynParF
    ###     TO dO: set sparse matrics for tildeDynParA, tildeDynParB, tildeDynParF
    tildeDynParA = np.zeros((tildeStateDim,tildeStateDim))
    tildeDynParA[0:stateDim,0:stateDim] = dynParA
    tildeDynParA[0:stateDim,2*stateDim:tildeStateDim] = dynParB
    tildeDynParA[stateDim:2*stateDim,stateDim:2*stateDim] = dynParA
    tildeDynParB = np.zeros((tildeStateDim,controlDim))
    tildeDynParB[stateDim:2*stateDim,:] = dynParB
    tildeDynParB[2*stateDim:tildeStateDim,:] = np.eye(controlDim)
    tildeDynParF = np.zeros((tildeStateDim,))
    tildeDynParF[0:stateDim] = dynParF        
    ### construct LQ tracking weight matrices for reformed problem, i.e. tildeTrackParQ, 
    ###    tildeTrackParR, tildeTrackParF
    #TO dO: set sparse matrics for tildeTrackParQ, tildeTrackParR, tildeTrackParF
    tildeTrackParQ = np.zeros((tildeStateDim,tildeStateDim))
    tildeTrackParQ[0:stateDim,0:stateDim] = trackParQ0
    tildeTrackParQ[stateDim:2*stateDim,stateDim:2*stateDim] = trackParQ1
    tildeTrackParQ[2*stateDim:tildeStateDim,2*stateDim:tildeStateDim] = trackParR0
    tildeTrackParR = trackParR1
    tildeTrackParF = np.zeros((tildeStateDim,tildeStateDim))
    tildeTrackParF[0:stateDim,0:stateDim] = trackParF0
    tildeTrackParF[stateDim:2*stateDim,stateDim:2*stateDim] = trackParF1

    # call lqtracker_basic to solve reformed LQ tracking problem
    lqtTildeOptimalControl, lqtTildeOptimalState = lqtracker_basic(t0, t1, tildeX0, 
        tildeStateDim, controlDim, tildeZTrack, tildeDynParA, tildeDynParB, tildeDynParF, 
        tildeTrackParQ, tildeTrackParR, tildeTrackParF, outputNum, integrateTol, integrateMaxIter)
    
    # write results as original definition of state and control variables
    lqtOptimalState = lqtTildeOptimalState.iloc[:,0:stateDim]
    lqtOptimalControl = lqtTildeOptimalState.iloc[:,2*stateDim:tildeStateDim]
    lqtOptimalControl.columns = range(controlDim) # reset column index from 0
    
    return lqtOptimalControl, lqtOptimalState
    
    
    
def lqtracker_basic(t0, t1, x0, stateDim, controlDim, zTrack, dynParA, dynParB, dynParF, 
                    trackParQ, trackParR, trackParF, outputNum, integrateTol, integrateMaxIter):
        
    '''
    %--------------------------------------------------------------------------
    %  Purpose  : Compute the optimal feedback control of the following 
    %             LQ tracking problem for the time interval [t0,t1] 
    %
    %                 min (integrate from t0 to t1)CriterionFunction(x(t),u(t))dt+(terminal_term)
    %                 s.t. dx/dt=A*x(t)+B*u(t)+f
    %
    %             with
    %                   x(t0) = x0
    %                   u(t0) = u0
    %
    %             where
    %                 CriterionFunction(x(t),u(t))=0.5*{Transpose(x(t)-z(t))*Q*(x(t)-z(t))+transpose(u(t))*R*u(t)}
    %                 terminal_term = 0.5*Transpose(x(t1)-z(t1))*F*x(t1-z(t1))
    %                 x(t) is a vector of state variable
    %                 u(t) is a vector of control variable
    %                 z(t) is a vector of given tracking signal
    %                 A, B, f, Q, R are parameters of the LQ tracking problem
    %
    %   
    %  Inputs:
    %   t0, t1          :  starting and ending time which define the time interval of
    %                      the LQ tracking problem defined on
    %                      type - float
    %   x0              :  initial value of state vector at time t0
    %                      type - 1D numpy.ndarray, dimension: stateDim x 1
    %   stateDim        :  dimension of state vector 
    %                      type - integer > 0
    %   controlDim      :  dimension of control vector 
    %                      type - integer > 0, note: controlDim <= stateDim
    %   zTrack          :  piecewise constant tracking signal of the state vector for the time interval [t0,t1]
    %                      type - pandas dataframe with 'Time' to be indext, and 
    %                             the vector of tracking signal to be data, dimension: (k_z) x stateDim
    %                      Note: the time of the first data point should be t0, the time of the last data point should be t1
    %                            the zTrack vector at time t1 is used for terminal condition of LQ tracking
    %   dynParA         :  parameter of the dynamics dx/dt 
    %                      type - 2D numpy.ndarray, dimension: stateDim x stateDim
    %   dynParB         :  parameter of the dynamics dx/dt 
    %                      type - 2D numpy.ndarray, dimension: stateDim x controlDim
    %   dynParF         :  parameter of the dynamics dx/dt 
    %                      type - 1D numpy.ndarray, dimension: stateDim  x 1
    %   trackParQ       :  LQ tracking weight matrix for state vector
    %                      type - 2D numpy.ndarray diagnal matrix with positive 
    %                           elements, dimension: stateDim x stateDim 
    %   trackParR       :  LQ tracking weight matrix for control vector 
    %                      type - 2D numpy.ndarray diagnal matrix with positive 
    %                           elements, dimension: controlDim x controlDim
    %   trackParF       :  LQ tracking weight matrix for terminal term of the state vector
    %                      type - 2D numpy.ndarray diagnal matrix with positive 
    %                           elements, dimension: stateDim x stateDim 
    %   outputNum       :  the number of time points evenly dividing each time interval
    %                      defined by two adjasent tracking signal zTrack, and at each 
    %                      time point the state and control signal are recorded
    %                      type - integer > 0
    %   integrateTol    :  numerical integration process local error tolerance for the solution
    %                      type - float > 0
    %   integrateMaxIter:  maximal iterations allowed for executing nemerical integration
    %                      type - integer > 0
    %    
    %   (TO DO: may need to add input for controlling integration method)
    %
    %  Output:
    %   lqtOptimalControl:  collection of optimal control vector at a sequence of time 
    %                       points between t0 and t1 
    %                       type - pandas dataframe with 'Time' to be indext and 
    %                              the vector of control signal at each time to be data, data dimension (k_z*outputNum)xn
    %   lqtOptimalState  :  collection of optimal state vector at a sequence of time 
    %                       points between t0 and t1 
    %                       type - pandas dataframe with 'Time' to be indext and 
    %                              the vector of sate signal at each time to be data, data dimension (k_z*outputNum)xn
    %
    %
    %
    '''   
    # To do: need to check dimensions of the input, and raise error if dimensions are not match
    
    # To do: need to check initial time and terminal time of zTrack (i.e. =t0, and =t1 respectively)

    # To do: need to add defaul integrateTol, integrateMaxIter
    
    # solve riccati  equations 
    ### compute BInvRB appeared in riccati equations
    riccatiInvRB = (np.linalg.inv(trackParR)).dot(np.transpose(dynParB))
    riccatiBInvRB = dynParB.dot(riccatiInvRB)    
    ### setup algorithm parameters for the chosen ode method
    #tol = 1e-8
    #maxIter = 10000    
    ### solve riccati equations by backward integration 
    listRiccati = lqt_solve_riccati(t1, stateDim, zTrack, dynParA, dynParF, 
                                   trackParF, trackParQ, riccatiBInvRB, 
                                   outputNum, integrateTol, integrateMaxIter)
    
    # forward integration of state dynamics with results of riccati equations as input
    listStateAndControl = lqt_propogate_state(x0, t0, t1, stateDim, controlDim,
                                             listRiccati, riccatiInvRB, dynParA, 
                                             dynParB, dynParF, integrateTol, integrateMaxIter)
    
    # construct results for output - between two zTrack points of time, output 
    #     restules with every (zTrack(i+1)-zTrack(i))/outputNum time distance
    lqtResultTimeIndex = np.array([listStateAndControl[0][0]['Time'][0]])
    lqtResultState = np.array([listStateAndControl[0][0]['vectorState'][0]])
    lqtResultControl = np.array([listStateAndControl[0][0]['vectorControl'][0]])
    for i in range(len(listStateAndControl)):
        for j in range(len(listStateAndControl[i])):
            lqtResultTimeIndex = np.append(lqtResultTimeIndex, listStateAndControl[i][j]['Time'][-1])
            lqtResultState = np.append(lqtResultState, 
                                       np.array([listStateAndControl[i][j]['vectorState'][-1]]), 
                                       axis=0)
            lqtResultControl = np.append(lqtResultControl, 
                                       np.array([listStateAndControl[i][j]['vectorControl'][-1]]), 
                                       axis=0)
    lqtOptimalState = pd.DataFrame(lqtResultState, index=lqtResultTimeIndex)
    lqtOptimalState.index.names = ['Time']      
    lqtOptimalControl = pd.DataFrame(lqtResultControl, index=lqtResultTimeIndex)
    lqtOptimalControl.index.names = ['Time']
    
    return lqtOptimalControl, lqtOptimalState
    
    
            
def lqt_solve_riccati(t1, stateDim, zTrack, dynParA, dynParF, trackParF, 
                    trackParQ, riccatiBInvRB, outputNum, integrateTol, integrateMaxIter):
    '''
    # solve riccati  equations 
    #    i.e. backward integrate sigma and phi dynamics together:
    #      sigma_dot = -sigma*dynParA-Transpose(dynParA)*sigma+sigma*dynParB*Inv(trackParR)*Transpose(dynParB)-trackParQ
    #      phi_dot = (-Transpose(dynParA)+sigma*dynParB*Inv(trackParR)*Transpose(dynParB))*phi-sigma*dynParF+trackParQ*zTrack
    #    with terminal condition sigma_end=trackParF, phi_end=(nx1 zeros)
    #    Note: sigma is a symmetric 2D matrix
    '''    
    
    #### define the terminal condition of sigma and phi
    ####   note: need to reshape sigma as a vector, and make sigma vector and phi as one vector
    sigmaVectorDim = stateDim*stateDim
    vectorSigmaPhiBackStart = construct_sigma_phi_vector(trackParF, -trackParF.dot(np.array(zTrack.iloc[-1])), sigmaVectorDim)    
    
    ### solve riccati equations by backward integration 
    ####  i.e. for each piecewise constant of zTrack time interval,
    ####       call ode method to barckward integrate riccati equations
    iterNum = len(zTrack)-1
    tStart = t1
    listRiccati = [] # it will be a 2D list to record the results of riccati equations 
    for i in range(iterNum-1,-1,-1): # iteration i: work in the ith time interval in which zTrack is a constant        
        listRiccati.append([])
        listRiccatiRow = iterNum-i-1
        tEnd = zTrack.index[i]
        zTrackData = np.array(zTrack.iloc[i])
        
        # evenly divided time interval [tStart_i, tEnd_i) by outputNum
        tStep = (tEnd-tStart)/outputNum
        
        # backward integration for each tStep
        tStart_j = tStart
        for j in range(outputNum):
            # iteration j: work in the jth sub-time-interval within the zTrack constant interval
            tEnd_j = tStart_j+tStep
            
            # TO DO: need raise failure message when integration fail
            # backward propagate riccati equations
            vectorSigmaPhiBack, t, failFlag, iter_i = \
                ode.ode_rk23(riccati_eqs, tStart_j, tEnd_j, vectorSigmaPhiBackStart, 
                             integrateTol, integrateMaxIter, dynParA = dynParA, dynParF = dynParF, 
                             trackParQ = trackParQ, riccatiBInvRB = riccatiBInvRB,
                             zTrackData = zTrackData, sigmaVectorDim = sigmaVectorDim,
                             stateDim = stateDim)
                
            # records results with nicer format for forward integration, 
            #   note: do not record the starting point since it is given, not computed
            countRows = len(vectorSigmaPhiBack)
            riccatiResults = {"Time" : t[1:countRows], "countRows":countRows-1, 
                              "matrixSigma" : vectorSigmaPhiBack[1:countRows,0:sigmaVectorDim].reshape((countRows-1,stateDim,stateDim)), 
                              "vectorPhi" : vectorSigmaPhiBack[1:countRows,-stateDim:]}
    
            listRiccati[listRiccatiRow].append(riccatiResults)
            
            # update tStart_j and the start point, i.e. vectorSigmaPhiBackStart
            tStart_j = tEnd_j
            vectorSigmaPhiBackStart = vectorSigmaPhiBack[-1]
            
        # update tStart for outer loop i
        tStart = tEnd
        
    return listRiccati



def lqt_propogate_state(x0, t0, t1, stateDim, controlDim, listRiccati,
                      riccatiInvRB, dynParA, dynParB, dynParF, integrateTol, integrateMaxIter):
    '''
    #forward integration of state dynamics with results of riccati equations as input
    #     note: form a given time interval time to be processed, 
    #           set results of riccati equations to be constants equal the 
    #           values at the end of the time interval
    '''
    vectorStateStart = x0
    tStart = t0
    listStateAndControl = [] # it will be a 2D list to record the results of state
    startFlag = True
    iterNum = len(listRiccati)
    for i in range(iterNum): # iteration i: work in the ith time interval in which zTrack is a constant
        listStateAndControl.append([])   
        listRiccatiRow = iterNum-i-1
        if i+1 < iterNum:
            tEnd = listRiccati[listRiccatiRow-1][-1]['Time'][-1] #zTrack.index[i+1]
        else:
            tEnd = t1                    
        # evenly divided time interval [tStart_i, tEnd_i) by outputNum
        #tStep = (tEnd-tStart)/outputNum
        ### outputNum = len(listRiccati[listRiccatiRow-1])
        
        # forward integration for each tStep
        riccatiSteps_j = len(listRiccati[listRiccatiRow-1])
        tStart_j = tStart     
        for j in range(riccatiSteps_j): # iteration j: work in the jth sub-time-interval within the zTrack constant interval            
            listRiccatiCol = riccatiSteps_j-j-1
            if listRiccatiCol > 0:
                tEnd_j = listRiccati[listRiccatiRow][listRiccatiCol-1]['Time'][-1] # tStart_j+tStep
            else:
                tEnd_j = tEnd
            # initiate dictionary for recording the results of the jth iteration
            stateAndControl_j = {"Time" : [], "countRows": 0, "vectorState" : [], "vectorControl" : []} 
            initFlag = True
            sumCountRows = 0
            tStart_k = tStart_j
            for k in range(listRiccati[listRiccatiRow][listRiccatiCol]['countRows']-1,-1,-1): 
                # iteration k: the smallest time interval in which ricatti numerical results are constant
                if k > 0:
                    tEnd_k = listRiccati[listRiccatiRow][listRiccatiCol]['Time'][k-1]
                else:
                    tEnd_k = tEnd_j
                # construct parameters from results of riccati equations for forward integration                
                riccatiKlq = -riccatiInvRB.dot(listRiccati[listRiccatiRow][listRiccatiCol]['matrixSigma'][k])
                riccatiPsi = -riccatiInvRB.dot(listRiccati[listRiccatiRow][listRiccatiCol]['vectorPhi'][k])                
                stateAPlusBKlq = dynParA+dynParB.dot(riccatiKlq)
                stateBPsiPlusF = dynParB.dot(riccatiPsi)+dynParF

                # TO DO: need raise failture message when integration fail
                # propogate state dynamics with feedback policy
                vectorState, t, failFlag, iter_i = \
                    ode.ode_rk23(feedback_policy_state_dot, tStart_k, tEnd_k, vectorStateStart, 
                                 integrateTol, integrateMaxIter, stateAPlusBKlq=stateAPlusBKlq, 
                                 stateBPsiPlusF=stateBPsiPlusF)
                    
                # compute optimal feedback control
                listControl = []
                for tmp_i in range(len(t)):
                    listControl.append(riccatiKlq.dot(vectorState[tmp_i])+riccatiPsi)
                vectorControl = np.array(listControl)
                
                # records results with nicer format for forward integration, 
                #   note: if at time tStart_k=t0, record the state and control values at starting time point  
                #         if at tStart_k>t0, no need to record the values at tStart_k, 
                #         since it has been recorded at the previous iteration
                countRows = len(t)
                if startFlag:
                    startRow = 0
                    sumCountRows = sumCountRows+countRows                    
                    startFlag = False
                else:
                    startRow = 1
                    sumCountRows = sumCountRows+countRows-1
                if initFlag:
                    stateAndControl_j['Time'] = t[startRow:countRows]                
                    stateAndControl_j['vectorState'] = vectorState[startRow:countRows]
                    stateAndControl_j['vectorControl'] = vectorControl[startRow:countRows]
                    initFlag = False
                else:
                    stateAndControl_j['Time'] = np.append(stateAndControl_j['Time'],t[startRow:countRows])
                
                    stateAndControl_j['vectorState'] = np.append(stateAndControl_j['vectorState'],
                                 vectorState[startRow:countRows], axis=0)
                    stateAndControl_j['vectorControl'] = np.append(stateAndControl_j['vectorControl'],
                                 vectorControl[startRow:countRows], axis=0)
                
                # update tStart_k and the start point, i.e. vectorStateStart
                tStart_k = tEnd_k
                vectorStateStart = vectorState[-1]
                
            # update tStart_j for outer loop j
            stateAndControl_j['countRows']=sumCountRows
            listStateAndControl[i].append(stateAndControl_j)
            tStart_j = tEnd_j
            
        # update tStart for outer loop i
        tStart = tEnd 
        
    return listStateAndControl


    
def construct_sigma_phi_vector(matrixSigma, vectorPhi, sigmaVectorDim):
    '''
    #reshape sigma as a vector (i.e. row by row), and append phi to the end of 
    #the sigma vector as one vector
    # inputs: matrixSigma - nxn numpy 2D array
    #         vectorPhi - nx1 numpy 1D array
    #         sigmaVectorDim - integer with value=n*n
    # output: vectorSigmaPhi - (n*n+n)x1 2D array
    '''
    vectorSigmaPhi = np.append(matrixSigma.reshape((sigmaVectorDim,)),vectorPhi)
    return vectorSigmaPhi


def riccati_eqs(t, vectorSigmaPhi, **kwargs):
    # riccati equations for the LQ tracking problem solved by lqtracker_basic 
    ### separate vectorSigmaPhi into vectorSigma and vectorPhi
    vectorPhi = vectorSigmaPhi[-kwargs['stateDim']:]
    vectorSigma = vectorSigmaPhi[0:kwargs['sigmaVectorDim']]
    ### reshape vectorSigma to stateDim x stateDim 2D numpy array
    matrixSigma = vectorSigma.reshape((kwargs['stateDim'],kwargs['stateDim']))
    ### compute values of riccati equations, i.e. matrixSigmaDot and vectorPhiDot
    matrixSigmaDot = -matrixSigma.dot(kwargs['dynParA'])-np.transpose(kwargs['dynParA']).dot(matrixSigma)+\
                    matrixSigma.dot(kwargs['riccatiBInvRB']).dot(matrixSigma)-kwargs['trackParQ']
    vectorPhiDot = (-np.transpose(kwargs['dynParA'])+matrixSigma.dot(kwargs['riccatiBInvRB'])).dot(vectorPhi)-\
                   matrixSigma.dot(kwargs['dynParF'])+kwargs['trackParQ'].dot(kwargs['zTrackData'])
    ### reshape matrixSigmaDot as a vector, and make sigma_dot vector and vectorPhiDot as one vector
    vectorSigmaPhiDot = construct_sigma_phi_vector(matrixSigmaDot, vectorPhiDot, kwargs['sigmaVectorDim'])
    return vectorSigmaPhiDot


def feedback_policy_state_dot(t, vetorState, **kwargs):
    #  state dynamics with feedback policy constructed from the results of riccati eqs are given
    vetorStateDot = kwargs['stateAPlusBKlq'].dot(vetorState)+kwargs['stateBPsiPlusF']
    return vetorStateDot




  