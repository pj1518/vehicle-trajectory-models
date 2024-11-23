import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from utils.data_processing import load_and_filter_data
from utils.evaluation_metrics import calculate_ade, calculate_fde

# Start timer
start_time = time.time()

def constant_acceleration_prediction(recording_id, skip_width=0, num_frames=25, start_row=0):
    # Load and filter data
    data = load_and_filter_data(recording_id, skip_width)

    # Initial conditions
    initial_row = data.iloc[start_row]
    current_x, current_y = initial_row['xCenter'], initial_row['yCenter']
    current_vx, current_vy = initial_row['xVelocity'], initial_row['yVelocity']
    current_ax, current_ay = initial_row['xAcceleration'], initial_row['yAcceleration']

    # Time step
    dt = (skip_width + 1) / 25

    # Predictions
    predicted_x, predicted_y = [current_x], [current_y]

    for _ in range(1, num_frames):
        # Update velocities and positions using constant acceleration model
        current_vx += current_ax * dt
        current_vy += current_ay * dt
        current_x += current_vx * dt
        current_y += current_vy * dt
        predicted_x.append(current_x)
        predicted_y.append(current_y)

    # Ground truth for comparison
    ground_truth_x = data.iloc[start_row:start_row + num_frames]['xCenter'].values
    ground_truth_y = data.iloc[start_row:start_row + num_frames]['yCenter'].values

    # Calculate ADE and FDE
    ade_value = calculate_ade(predicted_x, predicted_y, ground_truth_x, ground_truth_y)
    fde_value = calculate_fde(predicted_x, predicted_y, ground_truth_x, ground_truth_y)

    # Save predictions
    prediction_df = pd.DataFrame({
        'xCenter_gt': ground_truth_x,
        'xCenter_pred': predicted_x,
        'yCenter_gt': ground_truth_y,
        'yCenter_pred': predicted_y
    })
    prediction_df.to_csv('constant_acceleration_predictions.csv', index=False)

    # Plot results
    time_vec = np.arange(0, num_frames * dt, dt)
    displacement_errors = np.sqrt((np.array(predicted_x) - ground_truth_x) ** 2 +
                                  (np.array(predicted_y) - ground_truth_y) ** 2)

    plt.figure(figsize=(10, 6))
    plt.plot(time_vec, displacement_errors, label="Displacement Error", marker='x')
    plt.axhline(y=ade_value, color='r', linestyle='--', label=f"ADE: {ade_value:.2f}")
    plt.axhline(y=fde_value, color='g', linestyle='--', label=f"FDE: {fde_value:.2f}")
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('Constant Acceleration Model Prediction Error')
    plt.legend()
    plt.tight_layout()
    plt.savefig('constant_acceleration_error_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print results
    print("Average Displacement Error (ADE):", ade_value)
    print("Final Displacement Error (FDE):", fde_value)
    elapsed_time = time.time() - start_time
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
