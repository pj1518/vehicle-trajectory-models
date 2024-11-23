import pandas as pd

def downsample(data, skip_width):
    """Downsample the data."""
    return data.iloc[::skip_width + 1].reset_index(drop=True)

def load_and_filter_data(recording_id, skip_width=0):
    """
    Load and filter the dataset for a specific recording ID. 
    Filters car tracks and downsample the data.
    """
    # Load the data
    data_track = pd.read_csv(f'data_processing/dataset/data/{recording_id}_tracks.csv')
    data_meta = pd.read_csv(f'data_processing/dataset/data/{recording_id}_tracksMeta.csv')

    # Filter only car tracks
    car_ids = data_meta[data_meta['class'] == 'car']['trackId']
    data_filtered = data_track[data_track['trackId'].isin(car_ids)]

    # Downsample the data
    data_downsampled = downsample(data_filtered, skip_width)

    # Columns to keep
    columns_to_keep = ["trackId", "xCenter", "yCenter", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]
    data = data_downsampled.loc[:, columns_to_keep]

    return data
