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

unit tests for testing lqtracker.py, i.e. numerical methods for solving 
LQ tracking problem
"""

import unittest   
import lqtracker as lqt
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
    
    def test_lqtracker_pid_simpleCase(self):
        # setup parameters, data  and initial point for testing function
        x0 = np.array([0.1,-0.3,1.0]) 
        u0 = np.array([0.01,0.03])
        stateDim = len(x0)
        controlDim = len(u0)
        t0 = 2.0
        t1 = 3.0 #4.0

        timeIndex = np.array([2,2.5,3.0])
        zData = np.array([[0.12,-0.27,1.1],[0.123,-0.276,1.08],[0.002,0.003,0.04]])
        zDotData = np.array([[0.1,-0.01,0.001],[0.0123,-0.0276,0.0108],[0.001,0.002,0.003]])
        uData = np.array([[0.02,0.003],[-0.03,0.04],[0.004,0.005]])        
        signalTrack = {"timeIndex" : timeIndex, "zTrack" : zData, "zDotTrack" : zDotData, "uTrack": uData}

        dynParA = np.array([[-1.0, 0.1, -0.3], [0.2, -1, 0.1], [0.1, 0.4, -1]])
        dynParB = np.array([[0.5, -0.6], [-0.7, 0.6], [0.4, 0.8]])        
        dynParF = np.array([-0.1, -0.02, -0.2])
        trackParQ0 = np.diag(np.array([10.0,20.0,30.0]))
        trackParQ1 = np.diag(np.array([0.01,0.02,0.03]))
        trackParR0 = np.diag(np.array([5.0,2.0]))
        trackParR1 = np.diag(np.array([0.01,0.02]))
        trackParF0 = np.diag(np.array([10.0,10.0,10.0]))
        trackParF1 = np.diag(np.array([1.5,1.0,0.6]))
        outputNum = 5
        integrateTol = 1e-6
        integrateMaxIter = 5000
        
        # call LQ tracker basic (i.e. veriation 1 in Edge Controller doc)
        lqtOptimalControl, lqtOptimalState = lqt.lqtracker_pid(t0, t1, x0, 
                                                           u0, stateDim, 
                                                           controlDim, signalTrack, 
                                                           dynParA, dynParB, 
                                                           dynParF, trackParQ0, 
                                                           trackParQ1, trackParR0, 
                                                           trackParR1, trackParF0, 
                                                           trackParF1, outputNum, 
                                                           integrateTol, integrateMaxIter) 
        # compare with expected results 
        expectTime = np.array([2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0])
        expectOptControl = np.array([[0.010000,  0.030000],\
                                    [0.510645,  0.642798],\
                                    [0.576875,  0.792619],\
                                    [0.574660,  0.762913],\
                                    [0.542027,  0.656152],\
                                    [0.463768,  0.505740],\
                                    [0.358086,  0.319013],\
                                    [0.241455,  0.080780],\
                                    [0.092229, -0.176567],\
                                    [-0.050328, -0.311178],\
                                    [0.266625,  0.125075]])
        expectOptState = np.array([[0.100000, -0.300000,  1.000000],\
                                    [0.044646, -0.263027,  0.919384],\
                                    [-0.013033, -0.225661,  0.881214],\
                                    [-0.065396, -0.192239,  0.851861],\
                                    [-0.108290, -0.166191,  0.819705],\
                                    [-0.140905, -0.147646,  0.779183],\
                                    [-0.164143, -0.134740,  0.726345],\
                                    [-0.176247, -0.129042,  0.658345],\
                                    [-0.176858, -0.130332,  0.572504],\
                                    [-0.169837, -0.134301,  0.472500],\
                                    [-0.166108, -0.133343 , 0.389243]])

        self.assertTrue(np.amax(abs(lqtOptimalControl.values-expectOptControl))<1e-6, msg=None)
        self.assertTrue(np.amax(abs(lqtOptimalState.values-expectOptState))<1e-6, msg=None)
        self.assertTrue(max(abs(lqtOptimalControl.index-expectTime))<1e-12, msg=None)
        self.assertTrue(max(abs(lqtOptimalState.index-expectTime))<1e-12, msg=None)

                
    def test_lqtracker_basic_simulate(self):
        # setup parameters, data  and initial point for testing function
        x0 = np.array([0.1,-0.3,1.0])#np.array([0.1,-0.3,1.0])       
        stateDim = len(x0)
        controlDim = 2
        #t0 = 0.0
        #t1 = 0.3 
        #zTimeIndex = np.array([0.0, 0.2])
        #zData = np.array([[-0.12, -0.45,  0.9],[-0.12, -0.45,  0.9]])
        #zTrack = pd.DataFrame(zData, index=zTimeIndex)
        #zTrack.index.names = ['Time']
        dynParA = np.array([[-1.0, 0.1, -0.3], [0.2, -1, 0.1], [0.1, 0.4, -1]])
        dynParB = np.array([[0.5, -0.1], [-0.2, 0.6], [0.4, 0.8]])        
        dynParF = np.array([-0.1, -0.02, -0.2])
        trackParQ = np.diag(np.array([100.0,100.0,100.0]))
        trackParR = np.diag(np.array([0.1, 0.1]))
        trackParF = np.diag(np.array([50.0,50.0,50.0]))
        outputNum = 100
        integrateTol = 1e-8
        integrateMaxIter = 10000    
        
        zData = np.array([[-0.2, -0.2,  0.5],[-0.45, 0.35,  0.85],[-2.0, 0.12,  -1.8],
                      [-1.1, -0.1,  -1.1],[0.2, 0.2,  -0.1]])
        zTimeIndex = np.array([[0.0, 0.2],[0.2,0.4],[0.4,0.6],[0.6,0.8],[0.8,1.0]])
        
        # call LQ tracker basic (i.e. veriation 1 in Edge Controller doc)
        for i in range(len(zData)):
            t0 = zTimeIndex[i,0]
            t1 = zTimeIndex[i,1]
            zTrack = pd.DataFrame(np.array([zData[i],zData[i]]), index=zTimeIndex[i])
            zTrack.index.names = ['Time']
            lqtOptimalControl, lqtOptimalState = lqt.lqtracker_basic(t0, t1, x0, stateDim, controlDim,  
                                                                 zTrack, dynParA, dynParB, 
                                                                 dynParF, trackParQ, 
                                                                 trackParR, trackParF, outputNum,
                                                                 integrateTol, integrateMaxIter)    
            #update x0
            x0 = lqtOptimalState.values[-1]

        # TO dO: add and compare with expected results
        self.assertTrue(True) 
 
        
    
    def test_lqtracker_basic_simpleCase(self):
        # setup parameters, data  and initial point for testing function
        x0 = np.array([0.1,-0.3,1.0])       
        stateDim = len(x0)
        controlDim = 2
        t0 = 2.0
        t1 = 3.0 #4.0
        #zTimeIndex = np.array([2,2.5,3.0,3.5])
        #zData = np.array([[0.12,-0.27,1.1],[0.123,-0.276,1.08],[0.127,-0.274,1.11],[0.13,-0.28,1.13]])
        zTimeIndex = np.array([2,2.5,3.0])
        zData = np.array([[0.12,-0.27,1.1],[0.123,-0.276,1.08],[0.0,0.0,0.0]])
        zTrack = pd.DataFrame(zData, index=zTimeIndex)
        zTrack.index.names = ['Time']
        dynParA = np.array([[-1.0, 0.1, -0.3], [0.2, -1, 0.1], [0.1, 0.4, -1]])
        dynParB = np.array([[0.5, -0.6], [-0.7, 0.6], [0.4, 0.8]])        
        dynParF = np.array([-0.1, -0.02, -0.2])
        trackParQ = np.diag(np.array([10.0,20.0,30.0]))
        trackParR = np.diag(np.array([5.0,2.0]))
        trackParF = np.diag(np.array([1.5,1.0,0.6]))
        outputNum = 5
        integrateTol = 1e-8
        integrateMaxIter = 5000
        
        # call LQ tracker basic (i.e. veriation 1 in Edge Controller doc)
        lqtOptimalControl, lqtOptimalState = lqt.lqtracker_basic(t0, t1, x0, stateDim, controlDim,  
                                                             zTrack, dynParA, dynParB, 
                                                             dynParF, trackParQ, 
                                                             trackParR, trackParF, outputNum,
                                                             integrateTol, integrateMaxIter)
                
        # compare with expected results 
        expectTime = np.array([2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0])
        expectOptControl = np.array([[0.538091,  0.954148],\
                                    [0.589635,  0.903814],\
                                    [0.617540,  0.858735],\
                                    [0.624641,  0.812464],\
                                    [0.611282,  0.757599],\
                                    [0.576994,  0.686039],\
                                    [0.522384,  0.615610],\
                                    [0.442203,  0.509925],\
                                    [0.332558,  0.353409],\
                                    [0.187862,  0.123700],\
                                    [-0.001342, -0.216655]])
        expectOptState = np.array([[0.100000, -0.300000,  1.000000],\
                                    [0.024270, -0.247597,  0.968192],\
                                    [-0.038381, -0.207075,  0.938349],\
                                    [-0.090490, -0.175505,  0.909373],\
                                    [-0.133836, -0.150765,  0.879820],\
                                    [-0.169521, -0.131414,  0.847822],\
                                    [-0.198761, -0.115911,  0.812124],\
                                    [-0.222223, -0.103239,  0.770928],\
                                    [-0.239124, -0.093713,  0.720384],\
                                    [-0.247798, -0.088376,  0.655380],\
                                    [-0.245268, -0.089365,  0.568777]])
        self.assertTrue(np.amax(abs(lqtOptimalControl.values-expectOptControl))<1e-6, msg=None)
        self.assertTrue(np.amax(abs(lqtOptimalState.values-expectOptState))<1e-6, msg=None)
        self.assertTrue(max(abs(lqtOptimalControl.index-expectTime))<1e-12, msg=None)
        self.assertTrue(max(abs(lqtOptimalState.index-expectTime))<1e-12, msg=None)
        


def suite_test():
    """
        Gather all the tests from this module in a test suite.
    """
    suite = unittest.TestSuite()
    suite.addTest(FunctionalityTestCase('test_lqtracker_basic_simpleCase'))
    suite.addTest(FunctionalityTestCase('test_lqtracker_basic_simulate'))
    suite.addTest(FunctionalityTestCase('test_lqtracker_pid_simpleCase'))
    return suite


if __name__ == '__main__':
   
    #mySuite=suite_vector()
    mySuite=suite_test()
      
    #runner=unittest.TextTestRunner()
    #runner.run(mySuit)
    
    result = unittest.result.TestResult()
    mySuite.run(result)
    print result