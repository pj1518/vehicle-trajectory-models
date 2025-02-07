import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(pl.LightningModule):
    def __init__(self, num_input_frames, num_input_features, num_output_frames, num_output_features):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_input_features, hidden_size=32, batch_first=True)
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_output_frames * num_output_features)
        self.relu = nn.ReLU()
        self.loss_fn = nn.MSELoss()
        self.train_losses_per_epoch = []
        self.val_losses_per_epoch = []
        self.test_losses = []
        # Metric storage
        self.test_ade = []
        self.test_fde = []

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the last output from LSTM
        x = self.relu(self.fc1(lstm_out))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)  # Log training loss for tracking
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)  # Log validation loss for tracking
        return loss
    
    def test_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self(X)
        loss = self.loss_fn(y_pred, y)
        self.test_losses.append(loss.item())
        
        # Assuming y and y_pred are shaped (batch_size, num_points * 2) where num_points = num_output_features
        y_pred_coords = y_pred.view(-1, 2)  # [x_pred, y_pred]
        y_gt_coords = y.view(-1, 2)  # [x_gt, y_gt]

        # Compute average and final displacement errors
        displacement_errors = torch.sqrt((y_pred_coords[:, 0] - y_gt_coords[:, 0]) ** 2 +
                                         (y_pred_coords[:, 1] - y_gt_coords[:, 1]) ** 2)
        
        ade = displacement_errors.mean().item()
        fde = displacement_errors[-1].item()

        self.test_ade.append(ade)
        self.test_fde.append(fde)

        return loss
    
    def on_test_epoch_end(self):
        avg_test_loss = sum(self.test_losses) / len(self.test_losses)
        avg_ade = sum(self.test_ade) / len(self.test_ade)
        avg_fde = sum(self.test_fde) / len(self.test_fde)

        print(f'Average Test Loss: {avg_test_loss:.4f}')
        print(f'Average Displacement Error (ADE): {avg_ade:.4f}')
        print(f'Final Displacement Error (FDE): {avg_fde:.4f}')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]
    
    def on_train_epoch_end(self, outputs=None):
        # Print the average training and validation loss
        avg_train_loss = self.trainer.logged_metrics.get('train_loss', 0)
        avg_val_loss = self.trainer.logged_metrics.get('val_loss', 0)
        print(f"Epoch {self.current_epoch + 1}: Average Training Loss: {avg_train_loss:.4f}, Average Validation Loss: {avg_val_loss:.4f}")

        if avg_train_loss is not None:
            self.train_losses_per_epoch.append(avg_train_loss.item())
        if avg_val_loss is not None:
            self.val_losses_per_epoch.append(avg_val_loss.item())
    
    def on_train_end(self):
        # Plot the losses after training is complete
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.train_losses_per_epoch) + 1), self.train_losses_per_epoch, label='Training Loss')
        plt.plot(range(1, len(self.val_losses_per_epoch) + 1), self.val_losses_per_epoch, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss per Epoch')
        plt.show()



