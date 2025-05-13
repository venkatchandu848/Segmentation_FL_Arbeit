# Imports
import time
import matplotlib.pyplot as plt
from train_central import OurModel, validate, get_transform, encode_segmap, MyCityscapes, get_validation_transform
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch
from torchmetrics.classification import JaccardIndex
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from Fedavg_noniid import prepare_dataloaders_noniid, weighted_average_weights
from Fedavg import CombinedLoss
from Fedprox import train_fedprox

train_transform = get_transform()
val_transform = get_validation_transform()
ignore_index = 255

# Function for plotting mIOU vs Communication rounds
def plot_iou(training_ious, validation_ious, filename="iou_vs_rounds (FedProx noniid).png"):

    plt.figure(figsize=(10, 6))
    rounds = range(1, len(training_ious) + 1)
    # Plot Training mIoU
    plt.plot(rounds, training_ious, color='skyblue', label='Training mIoU')
    
    # Plot Validation mIoU
    plt.plot(rounds, validation_ious, color='purple', label='Validation mIoU')
    
    plt.title('Training and Validation mIoU vs Communication Rounds for FedProx non-iid model')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Performance  Metric: mIoU')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(filename)  # Save the plot to a file
    plt.close()  # Close the plot to free up memory

def main_fedprox_noniid():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")   # cuda device
    root_dir = '/data/raml/group1/chandu/mmsegmentation/data/cityscapes'   # root directory of dataset
    num_classes = 20
    batch_size = 4
    num_workers = 4
    local_epochs = 10
    num_clients = 10
    communication_rounds = 100  # Number of communication rounds
    alpha = 0.6   # Parameter for non-iid distribution
    mu_prox = 0.001  # Proximal algorithm parameter

    model = OurModel(num_classes).to(device)
    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    criterion = CombinedLoss(dice_loss, ce_loss)
    metrics = JaccardIndex(task='multiclass', num_classes=num_classes).to(device)

    client_dataloaders, val_loader = prepare_dataloaders_noniid(root_dir, batch_size, num_workers, num_clients, alpha, train_transform, val_transform)
    training_ious = []   # List to store training mIoU for each communication round
    validation_ious = [] # List to store validation mIoU for each communication round

    with open("fedprox_noniid.txt", "w") as f:
        for round in range(communication_rounds):
            local_weights = []
            local_losses = []
            local_ious = []
            local_sizes = []

            for client in range(num_clients):
                print(f"Training on client {client + 1}/{num_clients}")
                client_model = OurModel(num_classes).to(device)
                client_model.load_state_dict(model.state_dict())
                client_optimizer = optim.AdamW(client_model.parameters(), lr=1e-3)

                client_weights, client_loss, client_iou, num_samples = train_fedprox(client_model, client_dataloaders[client], criterion, client_optimizer, metrics, device, local_epochs, model, mu_prox)

                local_weights.append(client_weights)
                local_losses.append(client_loss)
                local_ious.append(client_iou)
                local_sizes.append(num_samples)

                
            global_weights = weighted_average_weights(local_weights, local_sizes)
            model.load_state_dict(global_weights)

            avg_loss = sum(local_losses) / len(local_losses)
            avg_iou = sum(local_ious) / len(local_ious)
            training_ious.append(avg_iou)

            val_loss, val_iou = validate(model, val_loader, criterion, metrics, device)
            validation_ious.append(val_iou)

            f.write(f"Round {round + 1}/{communication_rounds}\n")
            f.write(f"Train Loss: {avg_loss:.4f}, Train IoU: {avg_iou:.4f}\n")
            f.write(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}\n")


    plot_iou(training_ious, validation_ious, filename="iou_vs_rounds (FedProx noniid).png")   # Plotting iou vs rounds
    
    model_save_path = "fedprox_noniid_model.pth"
    torch.save(model.state_dict(), model_save_path)   # Model saving
    print(f"Model saved to {model_save_path}")

    return model

if __name__ == "__main__":
    start_time = time.time()
    main_fedprox_noniid()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")

