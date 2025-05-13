# Imports
import time
import matplotlib.pyplot as plt
from train_central import get_transform, encode_segmap, get_validation_transform
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch
from torchmetrics.classification import JaccardIndex
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from Fedavg_noniid import prepare_dataloaders_noniid
from Fedavg import pad_batch, CombinedLoss
from Fedavg_man import CustomDeepLabV3, validate
from Fedprox_man import fedprox_loss

train_transform = get_transform()
val_transform = get_validation_transform()
max_norm = 10
ignore_index = 255

# Function for plotting mIOU vs Communication rounds
def plot_iou(training_ious, validation_ious, filename="iou_vs_rounds (FedProx MAN non-iid).png"):

    plt.figure(figsize=(10, 6))
    rounds = range(1, len(training_ious) + 1)
    # Plot Training mIoU
    plt.plot(rounds, training_ious, color='skyblue', label='Training mIoU')
    
    # Plot Validation mIoU
    plt.plot(rounds, validation_ious, color='purple', label='Validation mIoU')
    
    plt.title('Training and Validation mIoU vs Communication Rounds for FedProx MAN non-iid model')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Performance  Metric: mIoU')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(filename)  # Save the plot to a file
    plt.close()  # Close the plot to free up memory

def weighted_average_weights(weights, num_samples):
    """
    Compute the weighted average of model weights based on the number of samples.
    
    Args:
        weights (list of dict): A list where each element is a dictionary of model weights.
        num_samples (list of int): A list of the number of samples each model has trained on.
    
    Returns:
        dict: A dictionary containing the weighted average of the model weights.
    """
    avg_weights = {}
    total_samples = sum(num_samples)
    
    # Iterate over each key (layer) in the model weights
    for key in weights[0].keys():
        # Initialize the weighted sum for the current layer
        weighted_sum = torch.zeros_like(weights[0][key], dtype=torch.float32)
        
        # Iterate over each client's weights
        for i in range(len(weights)):
            weighted_sum += weights[i][key].float() * (num_samples[i] / total_samples)
        
        # Store the weighted average in the result dictionary
        avg_weights[key] = weighted_sum
        
    return avg_weights


# Training loop
def fedprox_man_noniid(client_model, train_loader, criterion, optimizer, metrics, device, epochs, global_model, mu_prox, mu, min_batch_size=4):
    client_model.train()
    client_model = client_model.to(device)
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_iou = 0
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            image, target = pad_batch(batch, min_batch_size)
            image, target = image.to(device), target.to(device)

            optimizer.zero_grad()
            output, var_list = client_model(image)  # output now includes variance list
            target = encode_segmap(target).long()
            loss = fedprox_loss(client_model, global_model, criterion, image, target, mu_prox, device)

            
            var_loss = 0
            # Adding variance loss
            for var in var_list:
                var_loss += torch.mean(var)
                
            total_loss = loss + ((mu/2) * var_loss)
            
            total_loss.backward()

            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters=client_model.parameters(), max_norm=max_norm)  # Clip gradients
                
            optimizer.step()
            iou = metrics(output['out'], target)
            batch_size = image.size(0)
            epoch_loss += total_loss.item() * batch_size
            epoch_iou += iou.item() * batch_size
            total_samples += batch_size
    
        avg_loss = epoch_loss / total_samples
        avg_iou = epoch_iou / total_samples
    
    return client_model.state_dict(), avg_loss, avg_iou, total_samples


# Main loop
def main_fedprox_man_noniid():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    root_dir = '/data/raml/group1/chandu/mmsegmentation/data/cityscapes'
    num_classes = 20
    batch_size = 4
    num_workers = 4
    local_epochs = 10
    num_clients = 10
    communication_rounds = 100  # Number of communication rounds
    mu_prox = 0.001  # Proximal algorithm parameter
    mu = 0.1  # MAN regularizer parameter
    alpha = 0.6  # Parameter for non-iid distribution

    global_model = CustomDeepLabV3(num_classes).to(device)
    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    criterion = CombinedLoss(dice_loss, ce_loss)
    metrics = JaccardIndex(task='multiclass', num_classes=num_classes).to(device)

    client_dataloaders, val_loader = prepare_dataloaders_noniid(root_dir, batch_size, num_workers, num_clients, alpha, train_transform, val_transform)
    training_ious = []   # List to store training mIoU for each communication round
    validation_ious = [] # List to store validation mIoU for each communication round

    with open("fedprox_man_noniid.txt", "w") as f:
        for round in range(communication_rounds):
            local_weights = []
            local_losses = []
            local_ious = []
            local_sizes = []

            for client in range(num_clients):
                print(f"Training on client {client + 1}/{num_clients}")
                client_model = CustomDeepLabV3(num_classes).to(device)
                client_model.load_state_dict(global_model.state_dict())
                client_optimizer = optim.AdamW(client_model.parameters(), lr=1e-3)

                client_weights, client_loss, client_iou, num_samples = fedprox_man_noniid(client_model, client_dataloaders[client], criterion, client_optimizer, metrics, device, local_epochs, global_model, mu_prox, mu)
                local_weights.append(client_weights)
                local_losses.append(client_loss)
                local_ious.append(client_iou)
                local_sizes.append(num_samples)
 
            global_weights = weighted_average_weights(local_weights, local_sizes)
            global_model.load_state_dict(global_weights)

            avg_loss = sum(local_losses) / len(local_losses)
            avg_iou = sum(local_ious) / len(local_ious)
            training_ious.append(avg_iou)

            val_loss, val_iou = validate(global_model, val_loader, criterion, metrics, device, mu)
            validation_ious.append(val_iou)

            f.write(f"Round {round + 1}/{communication_rounds}\n")
            f.write(f"Train Loss: {avg_loss:.4f}, Train IoU: {avg_iou:.4f}\n")
            f.write(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}\n")

    plot_iou(training_ious, validation_ious, filename="iou_vs_rounds (FedProx MAN non-iid).png")   # Plotting iou vs rounds
    
    model_save_path = "fedprox_man_noniid_model.pth"
    torch.save(global_model.state_dict(), model_save_path)   # MOdel saving
    print(f"Model saved to {model_save_path}")

    return global_model

if __name__ == "__main__":
    start_time = time.time()
    main_fedprox_man_noniid()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")

