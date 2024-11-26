
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import pandas as pd

# Import Data + Clean Data
def importCleanData(file_name, sheet, columns):
    data = pd.read_excel(file_name, sheet_name = sheet)
    cleanData = data.iloc[:, columns].dropna()
    return cleanData

def importCleanDataCSV(file_name, columns):
    data = pd.read_csv(file_name, header = None)
    cleanData = data.iloc[:, columns].dropna()
    return cleanData

justin = 'C:/Users/dhlpablo_m2/Desktop/Git/Data_extraction_kinetic_signaling_data/data/train_data/vegf_testdata'
Lionel = 'C:/Users/Imoukhuede lab/OneDrive - UW/Desktop/GitHub/Data_extraction_kinetic_signaling_data/data/train_data/vegf_testdata'
os.chdir(justin)


# Function 1: Baseline falling to steady state response
def baseline_steadystate_response(t, y_initial, y_final, kon):
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
def response_to_zero(t, C, y_initial,  kon, koff):
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
def response_to_steady_state(t, y_initial, y_final,  D, kon, koff):
    """
    Function to find the kon and koff values from the data
    Assuming we know the steady state response

    Parameters
    ----------
    data : pandas dataframe
        Dataframe containing thetime and response values
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
    return y_final * ( 1 - D * np.exp(-kon * t) + (D - 1) * np.exp(-koff * t)) + y_initial

# Function 4: Typical association
def typical_association(t, y_final, conc, kon, koff):
    """
    Function to find the kon and koff values from the data
    Assuming it is a typical association function

    Parameters
    ----------
    data : pandas dataframe
        Dataframe containing thetime and response values
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
    kd = koff/kon
    return ( (y_final * conc) / (koff/kon + conc) ) * (1 - np.exp( (-1*(kon * conc + koff)) * t) )

# Function 5: Typical dissociation
def typical_dissociation(t, y_initial, koff):
    """
    Function to find the koff values from the data
    Assuming it is a typical dissociation function

    Parameters
    ----------
    data : pandas dataframe
        Dataframe containing thetime and response values
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


    # Fitting the data to its appropriate function
def fit_data(time, response, p0, assumption, testFile):
    """
    Function to fit the data to the function

    Parameters
    ----------
    time : pandas Series
        The series containing the time values
    response : pandas Series
        The series containing the response values
    p0 : list
        The initial guess of the parameters
    assumption : string
        The assumption of the function
    testFile : pandas dataframe
        The dataframe containing the test data

    Returns
    -------
    param_k
        The parameters of the function
    pcov_k  
        The covariance of the parameters
    """
 
    if assumption == "baseline+steadystate":
        def baseline_steadystate_response(t, y_initial, y_final, kon):
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
        function = baseline_steadystate_response
        param_k, pcov_k = curve_fit(function, time, response, p0 = p0)
        
        plt.plot(testFile.iloc[:, 0], testFile.iloc[:, 1], 'o', label='Experimental Data')

        fitted_response = baseline_steadystate_response(testFile.iloc[:, 0], *param_k)

        plt.plot(testFile.iloc[:, 0], fitted_response, '-', label='Fitted Curve')


        plt.xlabel('Time (t)')
        plt.ylabel('Response')
        plt.title('Data and Fitted Curve')

        plt.legend()

        plt.show()

        print("Fitted parameters: ", param_k)


    elif assumption == "response to zero":
        def response_to_zero(t, C, y_initial, kon, koff):
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
            return (C / (kon - koff)) * (np.exp(-koff * t) - np.exp(-kon * t)) +y_initial
        
        function = response_to_zero
        param_k, pcov_k = curve_fit(function, time, response, p0 = p0)

        plt.plot(testFile.iloc[:, 0], testFile.iloc[:, 1], 'o', label='Experimental Data')

        fitted_response = response_to_zero(testFile.iloc[:, 0], *param_k)
        plt.plot(testFile.iloc[:, 0], fitted_response, '-', label='Fitted Curve')


        plt.xlabel('Time (t)')
        plt.ylabel('Response')
        plt.title('Data and Fitted Curve')

        plt.legend()

        plt.show()

        print("Fitted parameters: ", param_k)
        

    elif assumption == "response to steady state":
        def response_to_steady_state(t, y_initial, y_final, D, kon, koff):
            """
            Function to find the kon and koff values from the data
            Assuming we know the steady state response

            Parameters
            ----------
            data : pandas dataframe
                Dataframe containing thetime and response values
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
            return y_final * ((1 - D * np.exp(-kon * t)) + (D - 1) * np.exp(-koff * t)) + y_initial
        
        function = response_to_steady_state
        param_k, pcov_k = curve_fit(function, time, response, p0 = p0)

        plt.plot(testFile.iloc[:, 0], testFile.iloc[:, 1], 'o', label='Experimental Data')

        fitted_response = response_to_steady_state(testFile.iloc[:, 0], *param_k)
        plt.plot(testFile.iloc[:, 0], fitted_response, '-', label='Fitted Curve')


        plt.xlabel('Time (t)')
        plt.ylabel('Response')
        plt.title('Data and Fitted Curve')

        plt.legend()

        plt.show()

        print("Fitted parameters: ", param_k)

    elif assumption == "typical_association":
        def typical_association(t, y_final, conc, kon, koff):
            """
            Function to find the kon and koff values from the data
            Assuming it is a typical association function

            Parameters
            ----------
            data : pandas dataframe
                Dataframe containing thetime and response values
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
            kd = koff/kon
            return ( (y_final * conc) / (koff/kon + conc) ) * (1 - np.exp( (-1*(kon * conc + koff)) * t) )
        function = typical_association
        param_k, pcov_k = curve_fit(function, time, response, p0 = p0)
        plt.plot(testFile.iloc[:, 0], testFile.iloc[:, 1], 'o', label='Experimental Data')

        fitted_response = typical_association(testFile.iloc[:, 0], *param_k)
        plt.plot(testFile.iloc[:, 0], fitted_response, '-', label='Fitted Curve')


        plt.xlabel('Time (t)')
        plt.ylabel('Response')
        plt.title('Data and Fitted Curve')

        plt.legend()

        plt.show()

        print("Fitted parameters: ", param_k)


    elif assumption == "typical_dissociation":
        def typical_dissociation(t, y_initial, koff):
            """
            Function to find the koff values from the data
            Assuming it is a typical dissociation function

            Parameters
            ----------
            data : pandas dataframe
                Dataframe containing thetime and response values
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
        function = typical_dissociation
        param_k, pcov_k = curve_fit(function, time, response, p0 = p0)
        
        plt.plot(testFile.iloc[:, 0], testFile.iloc[:, 1], 'o', label='Experimental Data')

        fitted_response = typical_dissociation(testFile.iloc[:, 0], *param_k)
        plt.plot(testFile.iloc[:, 0], fitted_response, '-', label='Fitted Curve')


        plt.xlabel('Time (t)')
        plt.ylabel('Response')
        plt.title('Data and Fitted Curve')

        plt.legend()

        plt.show()

        print("Fitted parameters: ", param_k)


    return param_k, pcov_k