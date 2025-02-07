# main_bicycle.py

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.data_preprocessing import load_and_filter_data
from utils.metrics import average_displacement_error, final_displacement_error
from models.combined_models import BicycleModel, bicycle_model
import os

start_time = time.time()

# Load and preprocess data
recording_id = "25"
skip_width = 0
num_frames = 25
start_row = 0
dt = (skip_width + 1)/25

data = load_and_filter_data(recording_id, skip_width=skip_width)

if not os.path.exists('results'):
    os.makedirs('results')

data['wheelbase'] = data['length'] * 0.90
headings = data['heading']
delta_heading = headings.diff().fillna(0)

data['delta_heading'] = delta_heading
data['velocity'] = np.sqrt(data['xVelocity']**2 + data['yVelocity']**2)
data['estimated_steering_angles'] = np.arctan((data['wheelbase'] * data['delta_heading']) / (data['velocity'] * (1/25)))    

def run_bicycle_model(data, start_row, num_frames=25, dt=1/25):
    # Get the row for the specified start_row number
    data_row = data.iloc[start_row]

    # Extract necessary data
    current_x = data_row['xCenter']
    current_y = data_row['yCenter']
    current_vx = data_row['xVelocity']
    current_vy = data_row['yVelocity']
    current_ax = data_row['xAcceleration']
    current_ay = data_row['yAcceleration']
    current_heading = np.deg2rad(data_row['heading'])
    heading_change = np.deg2rad(data_row['delta_heading'])
    wheelbase = data_row['wheelbase']
    estimated_steering_angle = data_row['estimated_steering_angles']

    predicted_x_values = []
    predicted_y_values = []
    predicted_heading_values = []

    # Perform multiple iterations using the bicycle model
    for _ in range(num_frames):
        new_x, new_y, new_heading = bicycle_model(current_x, current_y, current_vx, current_vy,
                                                   current_ax, current_ay, current_heading,
                                                   heading_change, wheelbase, dt, estimated_steering_angle)

        predicted_x_values.append(new_x)
        predicted_y_values.append(new_y)
        predicted_heading_values.append(new_heading)

        # Update the current state for the next iteration
        current_x = new_x
        current_y = new_y
        current_heading = new_heading

    return predicted_x_values, predicted_y_values, predicted_heading_values

# Run the bicycle model to get predictions
predicted_x, predicted_y, predicted_heading = run_bicycle_model(data, start_row, num_frames)

# Create DataFrame to store the results
output_data = pd.DataFrame({
        'xCenter_pred': predicted_x,
        'yCenter_pred': predicted_y,
        'heading_pred': predicted_heading,
        'xCenter_gt': data.iloc[start_row:start_row + num_frames]['xCenter'].values,
        'yCenter_gt': data.iloc[start_row:start_row + num_frames]['yCenter'].values,
        'heading': data.iloc[start_row:start_row + num_frames]['heading'].values
    })

# Save predictions to CSV
output_csv_path = 'bicycle_predictions.csv'
output_data = output_data.round(3)
output_data.to_csv(output_csv_path, index=False)#, sep=';')
print(output_data.head())

# Calculate errors
ade_values = average_displacement_error(output_data)
ade_value = np.mean(ade_values)

# Calculate cumulative averages for error
cumulative_averages = []
cumulative_sum = 0
cumulative_count = 0

for value in ade_values:
    cumulative_sum += value
    cumulative_count += 1
    cumulative_average = cumulative_sum / cumulative_count
    cumulative_averages.append(cumulative_average)

fde_value = final_displacement_error(output_data)

# Step 4: Print the evaluation results
print("Average Displacement Error (ADE):", ade_value)
print("Final Displacement Error (FDE):", fde_value)
print(f'Data with distances has been saved to {output_csv_path}.')

# Visualize errors
timeVec = np.arange(0, dt * num_frames, dt)
plt.figure(figsize=(10, 6))
plt.plot(timeVec, cumulative_averages, marker='o', linestyle='-', label='Average Distance Error')
plt.plot(timeVec, ade_values, marker='x', linestyle='-', label='Deviated Distance')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Error in location prediction [m]')
plt.title('Bicycle Model Prediction Error')
plt.tight_layout()
plt.savefig('Bicycle.png', dpi=300, bbox_inches='tight', format='png')
plt.show()