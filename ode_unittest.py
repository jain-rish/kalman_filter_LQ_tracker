#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:47:51 2018

unit tests for testing ode.py, i.e. numerical methods for solving ordinary 
differential equations
"""

import unittest   
import ode
import math
import numpy as np

#import configtest # first test


class SimpleTestCaseVector(unittest.TestCase):
    '''
    test forward integration of 1x1 vector dynamics (i.e. in the dx/dt function x is 1x1 vector)
    '''
    def setUp(self):         
        self.myfun = dynamicFunSimpleVector
        
    
    def test_ode_rk23_simpleCase(self):
        # setup parameters and initial point for testing function
        x0=np.array([4.0])
        par_a=-0.2
        
        # setup time interval [a,b] for integration 
        a=2.0
        b=4.0
        
        # setup algorithm parameters for the chosen ode method to be tested
        tol=1e-8
        maxIter=300 
        
        # call integration methos
        x, t, failFlag, iter_i = ode.ode_rk23(self.myfun, a, b, x0, 
                                          tol, maxIter, par_a=par_a)
        
        # compare with expected results 
        x_expect = math.exp(-0.4)*4
        self.assertTrue(abs(x_expect-x[-1,0])<1e-6, msg=None)
        
def dynamicFunSimpleVector(t, x, **kwargs):
    '''
    # simple dynamic function for testing ode_rk23
    #        dx/dt=par_a*x
    #    if integrate dx/dt from [t0,t1], then solution of the dynamic is:
    #        x(t1)=exp(par_a*(t1-t0))*x(t0)
    # x: a real number
    # t: time
    # par_a: a real number
    #
    '''
    x_dot = kwargs['par_a']*x
    return x_dot


class NormalTestCaseVector(unittest.TestCase):
    '''
    test forward integration of 3x1 vector dynamics (i.e. in the dx/dt function x is 3x1 vector)
    '''
    
    def setUp(self):         
        self.myfun = dynamicsFunNormalVector
            
    def test_ode_rk23_normalCase(self):
        # setup parameters and initial point for testing function
        x0=np.array([0.1,-0.3,1.0])
        matrixA=np.array([[-1.0, 0.1, -0.3], [0.2, -1, 0.1], [0.1, 0.4, -1]])
        matrixB=np.array([[0.5, -0.6], [-0.7, 0.6], [0.4, 0.8]])
        vectorU=np.array([0.2,0.3])
        vectorF=np.array([-0.1, 0-0.02, -0.2])
        
        # setup time interval [a,b] for integration        
        a=0.0
        b=5.0
        
        # setup algorithm parameters for the chosen ode method to be tested
        tol=1e-8
        maxIter=300
        
        # call integration methos
        x, t, failFlag, iter_i = ode.ode_rk23(self.myfun, a, b, x0, tol, 
                                              maxIter, matrixA=matrixA, 
                                              matrixB=matrixB, 
                                              vectorU=vectorU, vectorF=vectorF)
        
        # compare with expected results 
        # (note: the values here is not the real expected result, but copied 
        #       from the output of ode_rk23, need a way to get expected result)
        x_expect = np.array([ -3.91136351e-01,  -1.45401145e-01,  -5.53579890e-01])        
        self.assertTrue(max(abs(x_expect-x[-1]))<1e-6, msg=None)

def dynamicsFunNormalVector(t, vectorX, **kwargs):
    '''
    # define dynamics to be integrated  
    # x: 3X1 vector ; t: 1X1 vector ; A: 3X3 matrix ; u: 1X2 vector ; B: 2X3 vector ; F: 3x1 vector
    #
    # in *args: A, B, u, F
    '''
    x_dot=kwargs['matrixA'].dot(vectorX)+kwargs['matrixB'].dot(kwargs['vectorU'])+kwargs['vectorF']*t
    return x_dot  
        

class TestCaseMatrix(unittest.TestCase):
    
    def setUp(self):         
        self.myfun = dynamicsFunMatrix
        
    
    def test_ode_rk23_simpleCase(self):
        '''
        test forward integration of 2x2 matrix dynamics (i.e. in the dx/dt function x is 2x2 matrix)
        '''
        # setup parameters and initial point for testing function
        matrixX0=np.array([[0.5,-0.2],[1.3,2.4]])
        matrixA=np.array([[0.2,-0.1], [-0.4,0.7]])
        matrixB=np.array([[0.006, -0.003], [-0.002, 0.008]])
        matrixQ=np.array([[0.006, -0.002], [-0.001, 0.005]])
        
        # setup time interval [a,b] for integration 
        a=0
        b=5
        
        # setup algorithm parameters for the chosen ode method to be tested
        tol=1e-8
        maxIter=1000
        
        # reshape matrixX0 from 2x2 matrix to 4x1 vector
        dim_matrixX0=matrixX0.shape
        vectorX0=matrixX0.reshape((dim_matrixX0[0]*dim_matrixX0[1],))
        # call integration methos
        vectorX, t, failFlag, iter_i = ode.ode_rk23(self.myfun, a, b, vectorX0, tol, 
                                              maxIter, matrixA=matrixA, 
                                              matrixB=matrixB, matrixQ=matrixQ)
        
        # reshape result from 4x1 to 2x2, and compare with expected result
        # (note: the values here is not the real expected result, but copied 
        #       from the output of ode_rk23, need a way to get expected result)
        matrixX_result = vectorX[-1].reshape(matrixX0.shape)
        matrixX_expect = np.array([[ 0.47734137,  0.10362487],[0.11596301,  0.00871929]])       
        self.assertTrue(np.amax(abs(matrixX_expect-matrixX_result))<1e-6, msg=None)
        
    def test_ode_rk23_normaleCase(self):
        '''
        test forward integration of 9x9 matrix dynamics (i.e. in the dx/dt function x is 9x9 matrix)
        '''        
        # setup parameters and initial point for testing function
        matrixX0=np.array([[ 0.76,  0.22,  0.67,  0.19,  0.71,  0.26,  0.53,  0.14,  0.25],\
                   [ 0.45,  0.17,  0.74,  0.28,  0.17,  0.25,  0.8 ,  0.08,  0.43],\
                   [ 0.69,  0.02,  0.6 ,  0.29,  0.92,  0.32,  0.08,  0.56,  0.32],
                   [ 0.99,  0.08,  0.84,  0.24,  0.42,  0.25,  0.24,  0.52,  0.95],\
                   [ 0.31,  0.33,  0.06,  0.13,  0.38,  0.6 ,  0.22,  0.83,  0.51],\
                   [ 0.2 ,  0.21,  0.1 ,  0.48,  0.89,  0.96,  0.09,  0.48,  0.25],\
                   [ 0.36,  0.16,  0.34,  0.49,  0.54,  0.32,  0.5 ,  0.8 ,  0.24],\
                   [ 0.69,  0.37,  0.69,  0.88,  0.89,  0.26,  0.36,  0.83,  0.72],\
                   [ 0.8 ,  0.64,  0.93,  0.01,  0.59,  0.85,  0.84,  0.13,  0.92]])
        matrixX0=matrixX0+np.eye(9)*10
    
     
        matrixA=np.array([[ 3.54,  0.5 ,  0.3 ,  0.06,  0.63,  0.68,  0.57,  0.41,  0.13],\
                       [ 0.04,  3.05,  0.46,  0.42,  0.57,  0.45,  0.49,  0.32,  0.27],\
                       [ 0.32,  0.79,  3.66,  0.19,  0.11,  0.37,  0.32,  0.63,  0.1 ],\
                       [ 0.24,  0.35,  0.59,  3.12,  0.93,  0.51,  0.86,  0.76,  0.48],\
                       [ 0.97,  0.86,  0.4 ,  0.92,  3.35,  0.74,  0.62,  0.42,  0.37],\
                       [ 0.79,  0.02,  0.64,  0.64,  0.17,  3.42,  0.44,  0.26,  0.96],\
                       [ 0.48,  0.49,  0.4 ,  0.45,  0.3 ,  0.61,  3.24,  0.26,  0.65],\
                       [ 0.7 ,  0.17,  0.91,  0.5 ,  0.65,  0.55,  0.01,  3.69,  0.11],\
                       [ 0.15,  0.14,  0.39,  0.25,  0.44,  0.53,  0.93,  0.84,  3.  ]])
                       
        matrixB=np.array([[-9.68,  0.5 ,  0.5 ,  0.33,  0.8 ,  0.  ,  0.93,  0.91,  0.85],\
               [ 0.79, -9.02,  0.55,  0.56,  0.64,  0.  ,  0.06,  0.46,  0.25],\
               [ 0.62,  0.47, -9.82,  0.18,  0.41,  0.7 ,  0.03,  0.84,  0.26],\
               [ 0.32,  0.05,  0.59, -9.42,  0.63,  0.99,  0.47,  0.37,  0.2 ],\
               [ 0.59,  0.38,  0.71,  0.27, -9.75,  0.82,  0.4 ,  1.  ,  0.38],\
               [ 0.85,  0.2 ,  0.18,  0.46,  0.22, -9.05,  0.61,  0.97,  0.76],\
               [ 0.75,  0.93,  0.53,  0.74,  0.52,  0.54, -9.8 ,  0.31,  0.63],\
               [ 0.32,  0.62,  0.  ,  0.62,  0.65,  0.9 ,  0.74, -9.68,  0.07],\
               [ 0.59,  0.4 ,  0.57,  0.49,  0.58,  0.5 ,  0.86,  0.  , -9.94]])
    
        matrixQ=np.array([[ 0.57,  0.08,  0.05,  0.  ,  0.  ,  0.06,  0.01,  0.02,  0.06],\
               [ 0.07,  0.5 ,  0.01,  0.06,  0.02,  0.09,  0.02,  0.09,  0.06],\
               [ 0.02,  0.01,  0.57,  0.01,  0.07,  0.07,  0.05,  0.07,  0.06],\
               [ 0.  ,  0.06,  0.02,  0.54,  0.06,  0.03,  0.08,  0.09,  0.02],\
               [ 0.03,  0.  ,  0.05,  0.  ,  0.54,  0.04,  0.06,  0.03,  0.02],\
               [ 0.05,  0.06,  0.04,  0.02,  0.05,  0.6 ,  0.07,  0.08,  0.05],\
               [ 0.  ,  0.03,  0.07,  0.09,  0.06,  0.02,  0.52,  0.04,  0.01],\
               [ 0.05,  0.06,  0.03,  0.1 ,  0.05,  0.03,  0.03,  0.6 ,  0.02],\
               [ 0.02,  0.02,  0.04,  0.09,  0.05,  0.01,  0.03,  0.  ,  0.5 ]])
        
        # setup time interval [a,b] for integration 
        a=0.0
        b=1.0
        
        # setup algorithm parameters for the chosen ode method to be tested
        tol=1e-8
        maxIter=1000
        
        # reshape matrixX0 from 9x9 matrix to 81x1 vector
        dim_matrixX0=matrixX0.shape
        vectorX0=matrixX0.reshape((dim_matrixX0[0]*dim_matrixX0[1],))
        # call integration methos
        vectorX, t, failFlag, iter_i = ode.ode_rk23(self.myfun, a, b, vectorX0, tol, 
                                              maxIter, matrixA=matrixA, 
                                              matrixB=matrixB, matrixQ=matrixQ)
        
        # reshape result from 81x1 to 9x9, and compare with expected result
        # (note: the values here is not the real expected result, but copied 
        #       from the output of ode_rk23, need a way to get expected result)
        matrixX_result = vectorX[-1].reshape(matrixX0.shape)
        matrixX_expect = np.array([
                [-0.05614609, -0.02764642,  0.00442386, -0.03128679,  0.0379626 ,\
                -0.00210206,  0.02055731,  0.01708753, -0.01079675],\
               [ 0.01076375, -0.075299  ,  0.04635011, -0.00160557,  0.02930959,\
                -0.01371929, -0.01172549, -0.03845662,  0.00592987],\
               [-0.01827535,  0.02592572, -0.06526755, -0.01455321, -0.00696632,\
                 0.0123018 ,  0.00875977,  0.03707331, -0.02007888],\
               [-0.01917939, -0.04928098,  0.01923522, -0.07813147,  0.05341208,\
                 0.0136149 , -0.00085168,  0.00241213,  0.0005026 ],\
               [ 0.00643727,  0.06033435, -0.04005195,  0.0275839 , -0.07949216,\
                 0.00268929, -0.00399012,  0.01671832, -0.00206837],\
               [ 0.01743537,  0.00612319, -0.0026365 ,  0.01606572, -0.01925   ,\
                -0.04091118, -0.00317609, -0.0098151 ,  0.00212008],\
               [ 0.00996543,  0.00499091, -0.00098197,  0.02704245, -0.01822727,\
                -0.00234791, -0.04023058, -0.00917272, -0.00128094],\
               [ 0.03103352,  0.002872  ,  0.01410333,  0.03743825, -0.0293213 ,\
                -0.01646853, -0.01985861, -0.07105586,  0.026175  ],\
               [-0.0198831 , -0.00450292,  0.00266541, -0.03444674,  0.01946599,\
                 0.00968283,  0.01935828,  0.01385639, -0.04228292]])       
        self.assertTrue(np.amax(abs(matrixX_expect-matrixX_result))<1e-6, msg=None)
        
    def test_ode_rk23_backwardCase(self):
        '''
        test backward integration of 2x2 matrix dynamics (i.e. in the dx/dt function x is 2x2 matrix)
        Note: this test case is the reversed test case of 
              TestCaseMatrix('test_ode_rk23_simpleCase')
              i.e. use result of TestCaseMatrix('test_ode_rk23_simpleCase') as 
              inital point, and integrate from the ending time of 
              TestCaseMatrix('test_ode_rk23_simpleCase')
              and all the way back to the starting time of 
              TestCaseMatrix('test_ode_rk23_simpleCase').
              The expected result of this test case should be very close to the 
              initial point of TestCaseMatrix('test_ode_rk23_simpleCase')
        '''
        # setup parameters and initial point for testing function
        matrixX0=np.array([[ 0.47734137,  0.10362487],[0.11596301,  0.00871929]]) #np.array([[0.5,-0.2],[1.3,2.4]])
        matrixA=np.array([[0.2,-0.1], [-0.4,0.7]])
        matrixB=np.array([[0.006, -0.003], [-0.002, 0.008]])
        matrixQ=np.array([[0.006, -0.002], [-0.001, 0.005]])
        
        # setup time interval [a,b] for integration  (a>b, backward integration)
        a=5.0
        b=0.0
        
        # setup algorithm parameters for the chosen ode method to be tested
        tol=1e-8
        maxIter=1000
        
        # reshape matrixX0 from 2x2 matrix to 4x1 vector
        dim_matrixX0=matrixX0.shape
        vectorX0=matrixX0.reshape((dim_matrixX0[0]*dim_matrixX0[1],))
        # call integration methos
        vectorX, t, failFlag, iter_i = ode.ode_rk23(self.myfun, a, b, vectorX0, tol, 
                                              maxIter, matrixA=matrixA, 
                                              matrixB=matrixB, matrixQ=matrixQ)
        
        # reshape result from 4x1 to 2x2, and compare with expected result
        # (note: the expected result values here is not the reversed values of real expected result, but copied 
        #       from the output of ode_rk23, need a way to get expected result)
        matrixX_result = vectorX[-1].reshape(matrixX0.shape)
        matrixX_expect = np.array([[0.5,-0.2],[1.3,2.4]]) #np.array([[ 0.47734137,  0.10362487],[0.11596301,  0.00871929]])       
        self.assertTrue(np.amax(abs(matrixX_expect-matrixX_result))<1e-4, msg=None)
        

def dynamicsFunMatrix(t, vectorX, **kwargs):
    '''
    # define dynamics to be integrated  
    # x: (n^2)x1 vector ; t: 1x1 vector ; A: nxn matrix ; B: nxn vector ; Q: nxn
    #
    # in *args: A, B, Q
    '''
    # reshape vectorX from (n^2)x1 vector to nxn matrix
    matrixX=vectorX.reshape(kwargs['matrixA'].shape)
    # compute dynamics of matrixX
    matrixX_dot=-matrixX.dot(kwargs['matrixA'])- \
        np.transpose(kwargs['matrixA']).dot(matrixX)+ \
        np.transpose(matrixX).dot(kwargs['matrixB']).dot(matrixX)-kwargs['matrixQ']*t
    
    # reshape matrixX_dot from nxn matrix to (n^2)x1 vector
    x_dot=matrixX_dot.reshape(vectorX.shape)
    return x_dot 


def suite_vector():
    """
        Gather all the tests from this module in a test suite.
    """
    suite = unittest.TestSuite()
    suite.addTest(SimpleTestCaseVector('test_ode_rk23_simpleCase'))
    suite.addTest(NormalTestCaseVector('test_ode_rk23_normalCase'))
    return suite

def suite_matrix():
    """
        Gather all the tests from this module in a test suite.
    """
    suite = unittest.TestSuite()
    suite.addTest(TestCaseMatrix('test_ode_rk23_simpleCase'))
    suite.addTest(TestCaseMatrix('test_ode_rk23_normaleCase'))
    suite.addTest(TestCaseMatrix('test_ode_rk23_backwardCase'))    
    return suite

if __name__ == '__main__':
   
    #mySuite=suite_vector()
    mySuite=suite_matrix()
      
    #runner=unittest.TextTestRunner()
    #runner.run(mySuit)
    
    result = unittest.result.TestResult()
    mySuite.run(result)
    print result