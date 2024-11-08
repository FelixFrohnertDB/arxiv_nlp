import logging
import torch
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, auc
import re 
from tqdm import tqdm 


class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)

class MLP(nn.Module):
    def __init__(self, raw_size, drop=.25):
        super().__init__()
        
        self.raw = nn.Sequential(
            Flatten(),
            nn.Linear(raw_size, 128), nn.PReLU(), nn.BatchNorm1d(128),
            nn.Dropout(drop), nn.Linear(128, 64), nn.PReLU(), nn.BatchNorm1d(64),
            nn.Dropout(drop), nn.Linear( 64, 64), nn.PReLU(), nn.BatchNorm1d(64)
            )
        
        self.output = nn.Sequential(
            nn.Linear(64, 32), nn.PReLU(), nn.Linear(32, 1), nn.Sigmoid())
              
        self.init_weights()

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)  # Initialize weights
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)  # Initialize biases to zero
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        out = self.output(raw_out)
        return out

# Define the training and validation functions


def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for data, labels, _, _ in train_loader:
        
        optimizer.zero_grad()
        outputs = model(data.float())
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    train_accuracy = 100 * correct_train / total_train
    train_loss = running_loss / len(train_loader)
    return train_loss, train_accuracy

def validate_one_epoch(model, val_loader, criterion):
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for data, labels, _, _ in val_loader:
            
            outputs = model(data.float())
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    
    val_loss = running_val_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    return val_loss, val_accuracy

# Training loop with early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, patience=7, file_name='saved_files/best_x_model.pth'):
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = validate_one_epoch(model, val_loader, criterion)

        scheduler.step()
        
        logging.info(f'Epoch [{epoch+1}/{num_epochs}], '
                     f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
                     f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), file_name)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                logging.info("Early stopping triggered")
                logging.info("Best model saved with {best_val_loss:.2f} accuracy")
                break

