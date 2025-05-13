# Imports
from train_central import OurModel, validate, get_transform, encode_segmap, MyCityscapes, get_validation_transform
import segmentation_models_pytorch as smp
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torchmetrics.classification import JaccardIndex
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from Fedavg import prepare_dataloaders, pad_batch, average_weights, CombinedLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, OneCycleLR

train_transform = get_transform()
val_transform = get_validation_transform()
ignore_index = 255

# Function for plotting mIOU vs Communication rounds
def plot_iou(training_ious, validation_ious, filename="iou_vs_rounds (FedProx).png"):

    plt.figure(figsize=(10, 6))
    rounds = range(1, len(training_ious) + 1)
    # Plot Training mIoU
    plt.plot(rounds, training_ious, color='skyblue', label='Training mIoU')
    
    # Plot Validation mIoU
    plt.plot(rounds, validation_ious, color='purple', label='Validation mIoU')
    
    plt.title('Training and Validation mIoU vs Communication Rounds for FedProx model')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Performance  Metric: mIoU')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(filename)  # Save the plot to a file
    plt.close()  # Close the plot to free up memory

# Federated proximal loss
def fedprox_loss(local_model, global_model, criterion, inputs, targets, mu_prox, device):
    local_model.to(device)
    global_model.to(device)
    local_model.train()

    # Compute standard loss
    outputs = local_model(inputs)
    loss = criterion(outputs, targets)

    # Add proximal term
    proximal_term = 0.0
    for param_local, param_global in zip(local_model.parameters(), global_model.parameters()):
        proximal_term += (param_local - param_global).norm(2)

    loss += (mu_prox / 2) * proximal_term
    return loss

# Training loop
def train_fedprox(client_model, train_loader, criterion, optimizer, metrics, device, epochs, global_model, mu_prox, max_norm = 10):
    client_model.train()
    client_model = client_model.to(device)

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_iou = 0
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            image, target = batch

            if image.size(0) == 1:
                continue
            image, target = image.to(device), target.to(device)

            current_batch_size = image.size(0)

            optimizer.zero_grad()
            output = client_model(image)
            target = encode_segmap(target).long()
            loss = fedprox_loss(client_model, global_model, criterion, image, target, mu_prox, device)
            
            
            loss.backward()

            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters=client_model.parameters(), max_norm=max_norm)   # Clip gradients

            optimizer.step()  # Update the model parameters                      
            iou = metrics(output, target)
            epoch_loss += loss.item() * current_batch_size
            epoch_iou += iou.item() * current_batch_size
            total_samples += current_batch_size

    avg_loss = epoch_loss / total_samples
    avg_iou = epoch_iou / total_samples

    return client_model.state_dict(), avg_loss, avg_iou, total_samples


# Main loop
def main_fedprox():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # cuda device
    root_dir = '/data/raml/group1/chandu/mmsegmentation/data/cityscapes'  # root directory of dataset
    num_classes = 20
    batch_size = 4
    num_workers = 4
    local_epochs = 10
    num_clients = 10
    communication_rounds = 100  # Number of communication rounds
    mu_prox = 0.001  # Proximal algorithm parameter

    global_model = OurModel(num_classes).to(device)
    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    criterion = CombinedLoss(dice_loss, ce_loss)
    metrics = JaccardIndex(task='multiclass', num_classes=num_classes).to(device)

    client_dataloaders, val_loader = prepare_dataloaders(root_dir, batch_size, num_workers, num_clients, train_transform, val_transform)
    training_ious = []   # List to store training mIoU for each communication round
    validation_ious = [] # List to store validation mIoU for each communication round

    with open("fedprox.txt", "w") as f:
        for round in range(communication_rounds):
            local_weights = []
            local_losses = []
            local_ious = []

            for client in range(num_clients):
                print(f"Training on client {client + 1}/{num_clients}")
                client_model = OurModel(num_classes).to(device)
                client_model.load_state_dict(global_model.state_dict())
                client_optimizer = optim.AdamW(client_model.parameters(), lr=1e-3)

                client_weights, client_loss, client_iou, num_samples = train_fedprox(client_model, client_dataloaders[client], criterion, client_optimizer, metrics, device, local_epochs, global_model, mu_prox)
                local_weights.append(client_weights)
                local_losses.append(client_loss)
                local_ious.append(client_iou)
                
            global_weights = average_weights(local_weights)
            global_model.load_state_dict(global_weights)

            avg_loss = sum(local_losses) / len(local_losses)
            avg_iou = sum(local_ious) / len(local_ious)
            training_ious.append(avg_iou)

            val_loss, val_iou = validate(global_model, val_loader, criterion, metrics, device)
            validation_ious.append(val_iou)

            f.write(f"Round {round + 1}/{communication_rounds}\n")
            f.write(f"Train Loss: {avg_loss:.4f}, Train IoU: {avg_iou:.4f}\n")
            f.write(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}\n")
    
    plot_iou(training_ious, validation_ious, filename="iou_vs_rounds (FedProx).png")   # Plotting iou vs rounds
    
    model_save_path = "fedprox_model.pth"
    torch.save(global_model.state_dict(), model_save_path)   # MOdel saving
    print(f"Model saved to {model_save_path}")

    return global_model

if __name__ == "__main__":
    start_time = time.time()
    main_fedprox()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")

