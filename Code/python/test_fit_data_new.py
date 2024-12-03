import unittest
from rate_kinetics_final import fit_data
import numpy as np
import pandas as pd

class TestRateKinetics(unittest.TestCase):
    def test_fit_data_baseline_steadystate(self):
    
        time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        response = np.array([0, 0.5, 0.9, 1.4, 1.8, 2.1, 2.6, 3.0, 3.4, 3.7, 4.1])
    
  
        p0 = [0.1, 0.1, 0.1] 
        assumption = "baseline+steadystate"
    
        expected_params = np.array([0.017, 17.556, 0.026])  
    
        data = pd.DataFrame({'time': time, 'RU 1nM': response})
    
        params, _ = fit_data(time, response, p0, assumption, data)
    
        np.testing.assert_allclose(params, expected_params, atol=1e-1)

    def test_fit_data_response_to_zero(self):
        
        time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        response = np.array([0, 0.5, 0.9, 1.4, 1.8, 2.1, 2.6, 3.0, 3.4, 3.7, 4.1])
        
    
        p0 = [0.1, 0.1, 0.1, 0.1] 
        assumption = "response to zero"
        
        expected_params = np.array([0.017, 17.556])  
        
        data = pd.DataFrame({'time': time, 'RU 1nM': response})
        
        params, _ = fit_data(time, response, p0, assumption, data)
        
        np.testing.assert_allclose(params, expected_params, atol=1e-1)
        
    def test_fit_data_response_to_steady_state(self):
            
        time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        response = np.array([0, 0.5, 0.9, 1.4, 1.8, 2.1, 2.6, 3.0, 3.4, 3.7, 4.1])
            
        
        p0 = [0.1, 0.1, 0.1, 0.1, 0.1] 
        assumption = "response to steady state"
            
        expected_params = np.array([0.026])  
            
        data = pd.DataFrame({'time': time, 'RU 1nM': response})
            
        params, _ = fit_data(time, response, p0, assumption, data)
            
        np.testing.assert_allclose(params, expected_params, atol=1e-1)
    
    def test_fit_data_association(self):
            
        time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        response = np.array([0, 0.5, 0.9, 1.4, 1.8, 2.1, 2.6, 3.0, 3.4, 3.7, 4.1])
                
            
        p0 = [0.1, 0.1, 0.1, 0.1] 
        assumption = "typical_association"
                
        expected_params = np.array([0.017, 0.026])  
                
        data = pd.DataFrame({'time': time, 'RU 1nM': response})
                
        params, _ = fit_data(time, response, p0, assumption, data)
                
        np.testing.assert_allclose(params, expected_params, atol=1e-1)

    def test_fit_data_dissociation(self):
                    
        time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        response = np.array([0, 0.5, 0.9, 1.4, 1.8, 2.1, 2.6, 3.0, 3.4, 3.7, 4.1])
                        
                    
        p0 = [0.1, 0.1] 
        assumption = "typical_dissociation"
                        
        expected_params = np.array([0.017])  
                        
        data = pd.DataFrame({'time': time, 'RU 1nM': response})
                        
        params, _ = fit_data(time, response, p0, assumption, data)
                        
        np.testing.assert_allclose(params, expected_params, atol=1e-1)
if __name__ == '__main__':
    unittest.main()
