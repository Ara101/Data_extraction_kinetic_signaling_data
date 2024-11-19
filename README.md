# Data Fitting
A python package to assist researchers in fitting experimental data to mathematical models for analyzing response behaviors over time.
Scientific experiments regarding rate kinetics often generate time-dependent response data which must be fitted to models in order to extract meaningful kinetic parameters. This can be difficult as rate kinetics can be unique as in they cannot be modelled using simple functions. This tool simplifies the fitting process by using predefined functions for common scenarios. Researchers can input their time and response data along with initial parameter guesses, and this will return optimized parameters and their covariances.

# Setup
pip install numpy scipy pandas

# Defining the Fitting Function
The fit_data function fits the time-reponse data to the appropriate model based on given assumptions. The user can decide between different assumptions about their data including baseline to a response, zero to a response, or a response to a steady-state behavior.

Example:
import numpy as np
from scipy.optimize import curve_fit

time = np.array([0, 1, 2, 3, 4])
response = np.array([0, 1, 2, 3, 4])
p0 = [1, 1, 1, 1, 1]

params, covariance = fit_data(time, response, p0, assumption = "baseline+steadystate")

print("Fitted Parameters:", params)
print("Covariance:", covariance)

# Available Assumptions
Baseline + Steady-State Response
Response to Zero
Response to Steady-State
Typical Association
Typical Dissociation