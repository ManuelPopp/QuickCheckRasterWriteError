#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:34:01 2023

Train and evaluate the model.
"""
__author__ = "Manuel"
__date__ = "Tue Nov 21 09:34:01 2023"
__credits__ = ["Manuel R. Popp"]
__license__ = "Unlicense"
__version__ = "1.0.1"
__maintainer__ = "Manuel R. Popp"
__email__ = "requests@cdpopp.de"
__status__ = "Development"

#-----------------------------------------------------------------------------|
#> Import modules
import os, sys, torch
from torch import device, cuda
from torch.nn import BCELoss
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torchvision import datasets, transforms

# Append current file directory to import from other Python scripts
dir_py = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_py)

from model import SimpleCNN

#-----------------------------------------------------------------------------|
#> Set directories
dir_main = os.path.dirname(dir_py)
dir_dat = os.path.join(dir_main,  "dat")
dir_out = os.path.join(dir_main,  "out")
out_mod = os.path.join(dir_main, "mod", "trained_model_sd.pt")

#-----------------------------------------------------------------------------|
#> Create data loaders
# Data loader
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels = 1),
        transforms.ToTensor()
    ])
    
    dataset = datasets.ImageFolder(
        root = os.path.join(dir_dat, "trn"), transform = transform
        )
    
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
        )
    
    batch_size = 32
    train_loader = DataLoader(
        train_dataset, batch_size = batch_size, shuffle = True
        )
    
    val_loader = DataLoader(
        val_dataset, batch_size = batch_size, shuffle = False
        )
    
    # Independent test set
    test_dataset = datasets.ImageFolder(
        root = os.path.join(dir_dat, "tst"), transform = transform
        )
    
    test_loader = DataLoader(
        test_dataset, batch_size = batch_size, shuffle = False
        )

#-----------------------------------------------------------------------------|
#> Train the CNN
# Initialize the model
if __name__ == "__main__":
    model = SimpleCNN()
    
    # Define loss function and optimizer
    criterion = BCELoss()
    optimizer = Adam(model.parameters(), lr = 0.001)
    
    # Set the device (CPU or GPU)
    dev = device("cuda" if cuda.is_available() else "cpu")
    model.to(dev)
    
    val_acc = []
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(dev), labels.to(dev)
    
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().view(-1, 1))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print training loss
        print(
            f"Epoch {epoch + 1}/{num_epochs}, " +
            f"Training Loss: {running_loss / len(train_loader)}"
            )
    
        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
    
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(dev), labels.to(dev)
                outputs = model(inputs)
                
                predicted = (outputs > 0.5).float()
        
                total_val += labels.size(0)
                correct_val += (predicted == labels.float().view(-1, 1)).sum(
                    ).item()
        
        accuracy_val = correct_val / total_val
        
        print(
            f"Epoch {epoch + 1}/{num_epochs}, " +
            f"Validation Accuracy: {accuracy_val * 100:.2f}%"
            )
        
        val_acc += [accuracy_val]
        
        # Very, very rudimentary stop condition based on test runs
        if len(val_acc) > 8 and accuracy_val > .98:
            torch.save(model.state_dict(), out_mod)
            if val_acc[-1] < val_acc[-2]:
                break
    
    output = [f"Last epoch: {epoch}/{num_epochs}\n"]
    output += [f"Validation accuracy: {val_acc}\n"]

#-----------------------------------------------------------------------------|
#> Test the model
if __name__ == "__main__":
    model.eval()
    correct_test = 0
    total_test = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(dev), labels.to(dev)
            outputs = model(inputs)
            
            # Apply sigmoid activation and convert to binary predictions (0 or 1)
            predicted = (outputs > 0.5).float()
    
            total_test += labels.size(0)
            correct_test += (predicted == labels.float().view(-1, 1)).sum(
                ).item()
    
    tst_acc = correct_test / total_test
    print(f"Accuracy on the test set: {tst_acc * 100:.2f}%")
    
    output += [f"Test accuracy: {tst_acc}\n"]
    
    with open("training.out", "w") as f:
        f.writelines(output)
