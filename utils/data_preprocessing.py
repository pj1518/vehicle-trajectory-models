import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def downsample(data, skip_width):
    """Downsample the data."""
    return data.iloc[::skip_width + 1].reset_index(drop=True)

def load_and_filter_data(recording_id, skip_width=0):
    """
    Load and filter the dataset for a specific recording ID. 
    Filters car tracks and downsample the data.
    """
    # Load the data
    data_track = pd.read_csv(f'data/{recording_id}_tracks.csv')
    data_meta = pd.read_csv(f'data/{recording_id}_tracksMeta.csv')

    # Filter only car tracks
    car_ids = data_meta[data_meta['class'] == 'car']['trackId']
    data_filtered = data_track[data_track['trackId'].isin(car_ids)]

    # Downsample the data
    data_downsampled = downsample(data_filtered, skip_width)

    # Columns to keep
    #columns_to_keep = ["trackId", "xCenter", "yCenter", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]
    #data = data_downsampled.loc[:, columns_to_keep]

    return data_downsampled

def normalize(df, start_column):
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        df.iloc[:, start_column:] = min_max_scaler.fit_transform(df.iloc[:, start_column:])
        return df

def prepare_sequences(data, input_features, output_features, num_input_frames, num_output_frames):
    input_data = data[input_features].values
    output_data = data[output_features].values

    X, y = [], []
    for i in range(num_input_frames, len(input_data) - num_output_frames + 1):
        X.append(input_data[i - num_input_frames:i])
        y.append(output_data[i:i + num_output_frames])

    X = np.array(X)
    y = np.array(y)

    y = np.reshape(y, (y.shape[0], num_output_frames * len(output_features)))
    return X, y
