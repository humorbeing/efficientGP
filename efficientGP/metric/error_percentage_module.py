import numpy as np

def error_percentage(y_true, y_pred):
    error_abs = np.abs(y_true - y_pred)
    error_p = np.mean(error_abs/y_true) * 100
    return error_p



    
