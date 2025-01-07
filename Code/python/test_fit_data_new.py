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
        
        time = np.array([0.395939572892549, 1.731947822943837, 3.066704400027353, 4.400423004405889, 5.733179957601867, 7.065219488487051, 8.396633182888348, 9.727466833719214, 11.057918876937954, 12.38792825532663, 13.717617083321121, 15.047153268270762, 16.376460488653127, 17.70561506599064, 19.034723850414696, 20.363649463184938, 21.692544547346206, 23.021363309985052, 24.2895522552094, 25.61869156824243, 26.947540859490243, 28.276405415042543, 29.605254706290356, 30.934119261842657, 32.263044874612895, 33.5920010159921, 35.5855039174311, 36.914536380332734, 38.24362990045231, 39.57272342057188, 42.2310478395514, 44.22488655568907, 46.21884738626261, 47.54821566386292, 48.8775992057677, 50.20692169045456, 52.20112674989986, 53.530571349022594, 55.52485272999033, 56.85432785772202, 58.84873135312563, 60.843073791311305, 62.83743149380147, 64.16707452888251, 66.16160013872201, 68.15600363412561, 69.48576878364253, 70.81548814024599, 71.87313653370846 ])
        response = np.array([1.1242542878448951, 2.0680462341536163, 2.8207494407158826, 3.414988814317674, 3.8624161073825505, 4.20031692766592, 4.442673378076062, 4.596476510067113, 4.6920208799403404, 4.719985085756896, 4.69901193139448, 4.654735272184936, 4.575503355704697, 4.47296793437733, 4.363441461595823, 4.225950782997764, 4.083799403430273, 3.9299962714392223, 3.7482289336316192, 3.6433631618195363, 3.4942207307979096, 3.347408650261002, 3.1982662192393754, 3.0514541387024607, 2.913963460104398, 2.7811334824757665, 2.59237509321402, 2.471196868008949, 2.35934004474273, 2.247483221476511, 2.0447427293064884, 1.9072520507084256, 1.7884041759880702, 1.7184936614466828, 1.6509134973900075, 1.5740119313944838, 1.4924496644295324, 1.434190902311709, 1.3642803877703216, 1.3106823266219259, 1.2594146159582422, 1.1988255033557067, 1.1405667412378833, 1.1126025354213276, 1.0799776286353477, 1.0287099179716641, 1.0193885160328158, 1.0030760626398205, 0.9774422073079805 ])
        
    
        p0 = [1, 1, 0.1, 0.09] 
        assumption = "response to zero"
        
        expected_params = np.array([1, 1, 0.1, 0.09])  
        
        data = pd.DataFrame({'time': time, 'RU 1nM': response})
        
        params, _ = fit_data(time, response, p0, assumption, data)
        
        np.testing.assert_allclose(params, expected_params, atol=5e-1)
        
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
                        
                    
        p0 = [0.1] 
        assumption = "typical_dissociation"
                        
        expected_params = np.array([0.017])  
                        
        data = pd.DataFrame({'time': time, 'RU 1nM': response})
                        
        params, _ = fit_data(time, response, p0, assumption, data)
                        
        np.testing.assert_allclose(params, expected_params, atol=5)
if __name__ == '__main__':
    unittest.main()
