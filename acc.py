import time
import pandas as pd
import matplotlib.pyplot as plt
from utils.data_preprocessing import load_and_filter_data
from utils.metrics import average_displacement_error, final_displacement_error
from models.combined_models import acc_model
import os
import numpy as np

start_time = time.time()

# Load and preprocess data
recording_id = "25"
future_sequence_length = 25
skip_width = 0

# Downsampling and filtering
filtered_data = load_and_filter_data(recording_id, skip_width=0)

if not os.path.exists('results'):
    os.makedirs('results')

# Use the model to predict the future positions
predictions = acc_model(x=filtered_data, skip_width=skip_width, future_sequence_length=future_sequence_length)

predictions_df = pd.DataFrame(predictions, columns=["xCenter_pred", "yCenter_pred"])

# Add ground truth to the DataFrame
predictions_df['xCenter_gt'] = filtered_data['xCenter'].values[:future_sequence_length]
predictions_df['yCenter_gt'] = filtered_data['yCenter'].values[:future_sequence_length]

# Save predictions to CSV
predictions_df.to_csv('results/acceleration_predictions.csv', index=False)

# Calculate Average Displacement Error
avg_disp_error = average_displacement_error(predictions_df)
print(f"Average Displacement Error (for each point): {avg_disp_error.mean()}")

average_distance_errors = []
for t in range(future_sequence_length):
    subset_distance = avg_disp_error[:t + 1]
    average_error = np.mean(subset_distance)
    average_distance_errors.append(average_error)

# Calculate Final Displacement Error
final_disp_error = final_displacement_error(predictions_df)
print(f"Final Displacement Error (last point): {final_disp_error}")

time_range = np.linspace(1/25, future_sequence_length/25, num=future_sequence_length)

plt.figure(figsize=(10, 6))
plt.plot(time_range, avg_disp_error, marker='o', linestyle='-', label='Average Distance Error')
plt.plot(time_range, average_distance_errors, marker='x', linestyle='-', label='Deviated Distance')
plt.xlabel('Time [seconds]')
plt.ylabel('Deviation in Distance [m]')
plt.title('Distance deviation and ADE vs Time')
plt.grid(True)
plt.legend()
plt.savefig(r'results\Acceleration Deviations.png', dpi=300, bbox_inches='tight', format='png')
plt.show()

elapsed_time = time.time() - start_time
print(f"Elapsed Time: {elapsed_time:.2f} seconds")
