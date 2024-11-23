import numpy as np

def euclidean_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_ade(predicted_x, predicted_y, ground_truth_x, ground_truth_y):
    """Calculate Average Displacement Error (ADE)."""
    displacement_errors = np.sqrt((predicted_x - ground_truth_x) ** 2 + (predicted_y - ground_truth_y) ** 2)
    return np.mean(displacement_errors)

def calculate_fde(predicted_x, predicted_y, ground_truth_x, ground_truth_y):
    """Calculate Final Displacement Error (FDE)."""
    final_x_pred, final_y_pred = predicted_x[-1], predicted_y[-1]
    final_x_gt, final_y_gt = ground_truth_x[-1], ground_truth_y[-1]
    return euclidean_distance(final_x_pred, final_y_pred, final_x_gt, final_y_gt)
