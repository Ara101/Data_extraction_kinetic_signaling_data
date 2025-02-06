import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import pandas as pd

class KineticDataAnalysis:
    def __init__(self, data_path):
        self.data_path = data_path
        os.chdir(self.data_path)
    
    @staticmethod
    def import_clean_data(file_name, sheet=None, columns=None, is_csv=False):
        if is_csv:
            data = pd.read_csv(file_name, header=None)
        else:
            data = pd.read_excel(file_name, sheet_name=sheet)
        return data.iloc[:, columns].dropna()
    
    @staticmethod
    def baseline_steadystate_response(t, y_initial, y_final, kon):
        return y_final * (1 - np.exp(-kon * t)) + y_initial

    @staticmethod
    def response_to_zero(t, C, y_initial, kon, koff):
        return (C / (kon - koff)) * (np.exp(-koff * t) - np.exp(-kon * t)) + y_initial

    @staticmethod
    def response_to_steady_state(t, y_initial, y_final, D, kon, koff):
        return y_final * ((1 - D * np.exp(-kon * t)) + (D - 1) * np.exp(-koff * t)) + y_initial

    @staticmethod
    def typical_association(t, y_final, conc, kon, koff):
        kd = koff / kon
        return ((y_final * conc) / (kd + conc)) * (1 - np.exp((-1 * (kon * conc + koff)) * t))

    @staticmethod
    def typical_dissociation(t, y_initial, koff):
        return y_initial * np.exp(-koff * t)
    
    def plot_fitted_curve(self, function, data, params):
        plt.plot(data.iloc[:, 0], data.iloc[:, 1], 'o', label='Experimental Data')
        fitted_response = function(data.iloc[:, 0], *params)
        plt.plot(data.iloc[:, 0], fitted_response, '-', label='Fitted Curve')
        plt.xlabel('Time (t)')
        plt.ylabel('Response')
        plt.title('Data and Fitted Curve')
        plt.legend()
        plt.show()
        print("Fitted parameters:", params)
    
    def fit_data(self, time, response, p0, assumption, data):
        function_map = {
            "baseline+steadystate": self.baseline_steadystate_response,
            "response to zero": self.response_to_zero,
            "response to steady state": self.response_to_steady_state,
            "typical_association": self.typical_association,
            "typical_dissociation": self.typical_dissociation
        }
        
        function = function_map.get(assumption)
        if function is None:
            raise ValueError("Invalid assumption type")
        
        params, pcov = curve_fit(function, time, response, p0=p0)
        self.plot_fitted_curve(function, data, params)
        return params, pcov
