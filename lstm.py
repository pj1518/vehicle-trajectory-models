import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.data_preprocessing import downsample, normalize, prepare_sequences
from models.lstm_model import CustomDataset, LSTMModel

# Load and preprocess data
recording_id = "25"
data_track = pd.read_csv(f'data/{recording_id}_tracks.csv')
data_meta = pd.read_csv(f'data/{recording_id}_tracksMeta.csv')
new_data_track = data_meta[data_meta['class'] == 'car']
new_data_track_id = new_data_track['trackId']
data_track_filtered = data_track[data_track['trackId'].isin(new_data_track_id)]

skip_width = 5
num_input_frames = 20
num_output_frames = 10
tracks_data_down_sampled = downsample(data_track_filtered, skip_width)
tracks_data_norm = normalize(tracks_data_down_sampled, 4)

input_features = ['xCenter', 'yCenter', 'heading', 'xVelocity', 'yVelocity', 'lonVelocity', 'latVelocity', 'latAcceleration']
output_features = ['xCenter', 'yCenter', 'heading']

X, y = prepare_sequences(tracks_data_norm, input_features, output_features, num_input_frames, num_output_frames)
X_train, X_test_full, y_train, y_test_full = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test_full, y_test_full, test_size=0.5, shuffle=False, random_state=42)

train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
test_dataset = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)
test_loader = DataLoader(test_dataset, batch_size=128)

# Train the model
model = LSTMModel(num_input_frames, len(input_features), num_output_frames, len(output_features))
trainer = pl.Trainer(max_epochs=50, callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=10)])
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)

