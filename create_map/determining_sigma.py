import pandas as pd
import numpy as np
from formatting_improved import format_gpx, convert_to_speeds
from scipy.optimize import minimize_scalar
import math
import numpy as np
import matplotlib.pyplot as plt

def calculate_equation_components(df):
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    
    errors = []
    alphas = []
    big_ts = []
    
    for i in range(1, len(df) - 1):
        t_prev = df.loc[i - 1, 'time'].timestamp()
        t_next = df.loc[i + 1, 'time'].timestamp()
        t_cur  = df.loc[i, 'time'].timestamp()
        
        try:
            big_t = t_next - t_prev
            alpha = (t_cur - t_prev) / big_t
            
            x_prev = df.loc[i - 1, 'x']
            x_next = df.loc[i + 1, 'x']
            y_prev = df.loc[i - 1, 'y']
            y_next = df.loc[i + 1, 'y']
            
            x_pred = x_prev + alpha * (x_next - x_prev)
            y_pred = y_prev + alpha * (y_next - y_prev)
            
            # Calculate Euclidean distance between predicted and real points
            dist = np.sqrt((x_pred - df.loc[i, 'x'])**2 + (y_pred - df.loc[i, 'y'])**2) # !!!!!!!!!!!!!!!!!! haversine
            alphas.append(alpha)
            errors.append(dist)
            big_ts.append(big_t)
        except:
            pass
    
    return errors, alphas, big_ts






def determine_sigmas(df_list):

    def log_likelihood(standard_dev):
        if standard_dev == 0:
            return 0
        
        variance = standard_dev**2

        log_total = 0
        for i in range(0,len(error_list)):
            d = error_list[i]
            alpha = alpha_list[i]
            big_t = big_t_list[i]
            if alpha != 0 and alpha != 1 and big_t != 0:
                scaled_var = variance * alpha * (1-alpha) * big_t
                log_total += -d**2 / (2 * scaled_var) + math.log(1/scaled_var)

        return max(0,log_total)



    values = []
    for r in df_list:
        error_list, alpha_list, big_t_list = calculate_equation_components(r)

        result = minimize_scalar(
            lambda v: -log_likelihood(v),
            bounds=(0, 0.001),  # Avoid zero by starting from a very small positive number
            method='bounded'
        )

        values.append(result.x)

    return values

#______________________ maximum likelihood estimation________________________________





"""
The log form of the equation in the paper
taken ver batim (I believe it should be twice the size but oh well)
"""