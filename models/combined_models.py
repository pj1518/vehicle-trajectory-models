import numpy as np
import pandas as pd 

  
def acc_model(x, skip_width, future_sequence_length):
    # Extract positions, velocities, and accelerations
    positions = (x.iloc[:, 4:6]).values
    velocities = (x.iloc[:, 9:11]).values
    accelerations = (x.iloc[:, 11:13]).values
    
    predictions = []

    # Time step (assuming 1/25 seconds based on your information)
    dt = (skip_width + 1) / 25
    
    # Initial prediction using the constant acceleration formula
    pred_position1 = positions[0] + velocities[0] * dt + 0.5 * accelerations[0] * dt**2
    predictions.append(pred_position1.tolist())
    
    velocity = velocities[0] + accelerations[0]

    pred_positions_list = [pred_position1]
    
    for t in range(future_sequence_length - 1):
        pred_positions = pred_positions_list[-1] + velocity * dt + 0.5 * accelerations[0] * dt**2
        velocity = velocity + accelerations[0] * dt
        
        pred_positions_list.append(pred_positions)
        predictions.append(pred_positions.tolist())

    return np.array(predictions)

   
    
def bicycle_model(current_x, current_y, current_vx, current_vy, current_ax, current_ay,
              current_heading, heading_change, wheelbase, dt, estimated_steering_angle):
    # Calculate the change in velocity along the x and y directions
    delta_vx = (current_ax * np.cos(current_heading) - current_ay * np.sin(current_heading)) * dt
    delta_vy = (current_ax * np.sin(current_heading) + current_ay * np.cos(current_heading)) * dt

    new_vx = current_vx + delta_vx
    new_vy = current_vy + delta_vy

    # Calculate the change in x and y positions
    delta_x = (new_vx * np.cos(current_heading) - new_vy * np.sin(current_heading)) * dt
    delta_y = (new_vx * np.sin(current_heading) + new_vy * np.cos(current_heading)) * dt

    # Calculate delta_heading (yaw rate)
    if isinstance(wheelbase, (int, float)) and wheelbase != 0:
        delta_heading = (new_vx / wheelbase) * np.tan(estimated_steering_angle) * dt
    else:
        delta_heading = heading_change

    # Predict the new x and y positions
    new_x = current_x + delta_x
    new_y = current_y + delta_y
    new_heading = current_heading + delta_heading

    return new_x, new_y, new_heading