import numpy as np

def average_displacement_error(df):
    displacement_errors = np.sqrt((df['xCenter_pred'] - df['xCenter_gt'])**2 +
                                  (df['yCenter_pred'] - df['yCenter_gt'])**2)
    return displacement_errors

def final_displacement_error(df):
    last_row = df.iloc[-1]
    return np.sqrt((last_row['xCenter_pred'] - last_row['xCenter_gt'])**2 +
                   (last_row['yCenter_pred'] - last_row['yCenter_gt'])**2)
