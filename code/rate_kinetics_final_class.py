import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import pandas as pd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

class KineticAnalysis:
    def __init__(self, file_name=None, sheet=None, columns=None):
        self.file_name = file_name
        self.sheet = sheet
        self.columns = columns
        self.data = None

    # Import Data + Clean Data
    def import_clean_data(self):
        data = pd.read_excel(self.file_name, sheet_name=self.sheet)
        self.data = data.iloc[:, self.columns].dropna()
        return self.data

    def import_clean_data_csv(self):
        data = pd.read_csv(self.file_name, header=None)
        self.data = data.iloc[:, self.columns].dropna()
        return self.data

    # Function 1: Baseline falling to steady state response
    def baseline_steadystate_response(self, t, y_initial, y_final, kon):
        """
        Function to find the kon value from the data. 
        Assuming we know the baseline and the steady state response, we can find the kon value.
        The equation is y(t) = y_final * (1 - exp(-kon * t)) + y_intial

        Parameters
        ----------
        data : pandas dataframe
            Dataframe containing the time and response values
        y_intial : float
            The baseline value of the response
        y_final : float
            The steady state value of the response
        t : float
            The time value

        Returns
        -------
        function
            The function that can be used to calculate the response
        """
        return y_final * (1 - np.exp(-kon * t)) + y_initial

    # Function 2: Response falling to zero
    def response_to_zero(self, t, C, y_initial, kon, koff):
        """
        Function to find the kon and koff values from the data
        Assuming we know that the response goes to zero
        
        Parameters
        ----------
        data : pandas dataframe
            Dataframe containing the time and response values
        C : float
            The initial rate of signaling
        y_initial : float
            The baseline value of the response
        t : float
            The time value
        
        Returns
        -------
        function
            The function that can be used to calculate the response
        """
        return (C / (kon - koff)) * (np.exp(-koff * t) - np.exp(-kon * t)) + y_initial

    # Function 3: Response falling to steady state response
    def response_to_steady_state(self, t, y_initial, y_final, D, kon, koff):
        """
        Function to find the kon and koff values from the data
        Assuming we know the steady state response
        
        Parameters
        ----------
        data : pandas dataframe
            Dataframe containing the time and response values
        y_initial : float
            The initial rate of signaling
        y_final : float
            The final rate of signaling
        t : float
            The time value
        
        Returns
        -------
        function
            The function that can be used to calculate the response
        """
        return y_final * (1 - D * np.exp(-kon * t) + (D - 1) * np.exp(-koff * t)) + y_initial

    # Function 4: Typical association
    def typical_association(self, t, y_final, conc, kon, koff):
        """
        Function to find the kon and koff values from the data
        Assuming it is a typical association function
        
        Parameters
        ----------
        data : pandas dataframe
            Dataframe containing the time and response values
        y_final : float
            The final rate of signaling
        t : float
            The time value
        conc : float
            The concentration of the substance
        
        Returns
        -------
        function
            The function that can be used to calculate the response
        """
        kd = koff / kon
        return ((y_final * conc) / (kd + conc)) * (1 - np.exp((-1 * (kon * conc + koff)) * t))

    # Function 5: Typical dissociation
    def typical_dissociation(self, t, y_initial, koff):
        """
        Function to find the koff values from the data
        Assuming it is a typical dissociation function
        
        Parameters
        ----------
        data : pandas dataframe
            Dataframe containing the time and response values
        y_initial : float
            The initial rate of signaling
        t : float
            The time value
        
        Returns
        -------
        function
            The function that can be used to calculate the response
        """
        return y_initial * np.exp(-koff * t)
