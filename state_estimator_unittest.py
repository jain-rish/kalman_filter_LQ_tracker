#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 08:19:20 2018

@author: home
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:47:51 2018

unit tests for testing state_estimator.py, i.e. numerical methods for implementing Kalman filter
"""

import unittest   
from state_estimator import kalman_filter_continuous as kfc
import numpy as np
import pandas as pd

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

class FunctionalityTestCase(unittest.TestCase):
    '''
    test forward integration of 1x1 vector dynamics (i.e. in the dx/dt function x is 1x1 vector)
    '''
    def setUp(self):         
        #self.myfun = dynamicFunSimpleVector
        pass

    def test_kfc_normalCase(self):
        # setup parameters, data  and initial point for testing function
        x0 = np.array([0.1,-0.3,1.0]) 
        stateCov0 = np.diag(np.array([0.01,0.02,0.03]))
        stateDim = len(x0)
        controlDim = 2
        observeDim = 2
        t0 = 2.0
        t1 = 3.0
        
        uTimeIndex = np.array([2.0,2.5,3.0])
        uData = np.array([[0.010000,  0.030000],[0.510645,  0.642798],[0.576875,  0.792619]])
        uControl = pd.DataFrame(uData, index = uTimeIndex)
        uControl.index.names = ['Time']
        
        yTimeIndex = np.array([2.0,2.2,2.4,2.6,2.8,3.0])
        yData = np.array([[0.5393,-0.0555],[0.4788,-0.017 ],[0.4624,0.0294],[0.436,0.062],[0.3458,0.103 ],[ 0.3908,0.112 ]])
        yObserve = pd.DataFrame(yData, index = yTimeIndex)
        yObserve.index.names = ['Time']
        
        dynParA = np.array([[-1.0, 0.1, -0.3], [0.2, -1, 0.1], [0.1, 0.4, -1]])
        dynParB = np.array([[0.5, -0.6], [-0.7, 0.6], [0.4, 0.8]])        
        dynParF = np.array([-0.1, -0.02, -0.2])
        
        observeParH = np.array([[0.1,-0.06,0.5],[-0.6,0.3,0.1]])
        
        dynCovW = 1e-1*np.array([[3.0,0.01,-0.006],[0.01,4.2,-0.002],[-0.006,-0.002,2.8]])
        observeCovV = 1e-8*np.array([[6.0,-0.04],[-0.04,42.0]])
        
        outputNum = 2
        integrateTol = 1e-6
        integrateMaxIter = 5000      
        
        # call LQ tracker basic (i.e. veriation 1 in Edge Controller doc)
        kfcStateAndCov = kfc(t0, t1, x0, stateCov0, stateDim, controlDim, observeDim, 
                                uControl, yObserve, dynParA, dynParB, dynParF, observeParH, 
                                dynCovW, observeCovV, outputNum, integrateTol, integrateMaxIter)
        
        kfcEstObserve = np.transpose(observeParH.dot(np.transpose(kfcStateAndCov['kfcState'])))
        
        # compare with expected results 
        expectTime = np.array([2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0])
        expectKfcEstObserv = np.array([[ 0.528     , -0.05      ],\
                                       [ 0.47851375, -0.01635951],\
                                       [ 0.47851396, -0.01635955],\
                                       [ 0.46213485,  0.02995149],\
                                       [ 0.4621352 ,  0.02995168],\
                                       [ 0.43576168,  0.06248145],\
                                       [ 0.43582257,  0.06259666],\
                                       [ 0.3457088 ,  0.1034757 ],\
                                       [ 0.34570941,  0.10347577],\
                                       [ 0.3906694 ,  0.11248594],\
                                       [ 0.39066927,  0.11248611]])
        expectLastStateCov = np.array([[ 0.03714016,  0.0728391 ,  0.00138818],\
                                       [ 0.0728391 ,  0.14543557,  0.00290476],\
                                       [ 0.00138818,  0.00290476,  0.00032642]])

        self.assertTrue(max(abs(kfcStateAndCov['timeIndex']-expectTime))<1e-12, msg='test here0')
        self.assertTrue(np.amax(abs(kfcEstObserve-expectKfcEstObserv))<1e-6, msg='test here1')
        self.assertTrue(np.amax(abs(kfcStateAndCov['kfcStateCov'][-1]-expectLastStateCov))<1e-6, msg='test here')
        


def suite_test():
    """
        Gather all the tests from this module in a test suite.
    """
    suite = unittest.TestSuite()
    suite.addTest(FunctionalityTestCase('test_kfc_normalCase'))
    return suite


if __name__ == '__main__':
   
    #mySuite=suite_vector()
    mySuite=suite_test()
      
    #runner=unittest.TextTestRunner()
    #runner.run(mySuit)
    
    result = unittest.result.TestResult()
    mySuite.run(result)
    print (result)