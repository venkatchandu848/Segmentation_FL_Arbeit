# Imports
from train_central import OurModel, validate, get_transform, encode_segmap, MyCityscapes, get_validation_transform
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch
from torchmetrics.classification import JaccardIndex
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from Fedavg import prepare_dataloaders, pad_batch, average_weights
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, OneCycleLR

train_transform = get_transform()
val_transform = get_validation_transform()
ignore_index = 255

# Loss function
class CombinedLoss(nn.Module):
    def __init__(self, dice_loss, ce_loss):
        super(CombinedLoss, self).__init__()
        self.dice_loss = dice_loss
        self.ce_loss = ce_loss

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)
        return dice + ce
    
# Function for plotting mIOU vs Communication rounds
def plot_iou(training_ious, validation_ious, filename="iou_vs_rounds (FedDyn).png"):

    plt.figure(figsize=(10, 6))
    rounds = range(1, len(training_ious) + 1)
    # Plot Training mIoU
    plt.plot(rounds, training_ious, color='skyblue', label='Training mIoU')
    
    # Plot Validation mIoU
    plt.plot(rounds, validation_ious, color='purple', label='Validation mIoU')
    
    plt.title('Training and Validation mIoU vs Communication Rounds for FedDyn model')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Performance  Metric: mIoU')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(filename)  # Save the plot to a file
    plt.close()  # Close the plot to free up memory

# Training loop
def train_feddyn_model(model, train_loader, criterion, optimizer, metrics, device, epochs, max_norm = 10, lambda_reg=0.01):
    model.train()
    model = model.to(device)
    global_model_params = {name: param.clone() for name, param in model.named_parameters()}

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_iou = 0
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            image, target = batch

            if image.size(0) == 1:
                continue
            image, target = image.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(image)
            target = encode_segmap(target).long()
            loss = criterion(output, target)
            
            reg_loss = 0
            for name, param in model.named_parameters():
                reg_loss += torch.norm(param - global_model_params[name])
            loss += lambda_reg * reg_loss

            loss.backward()


            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)
            optimizer.step()  # Update the model parameters
            
            iou = metrics(output, target)
            batch_size = image.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_iou += iou.item() * batch_size
            total_samples += batch_size

    avg_loss = epoch_loss / total_samples
    avg_iou = epoch_iou / total_samples

    return model.state_dict(), avg_loss, avg_iou, total_samples


# Main loop
def main_feddyn():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # cuda device

    root_dir = '/data/raml/group1/chandu/mmsegmentation/data/cityscapes' # Root directory of dataset
    num_classes = 20
    batch_size = 4
    num_workers = 4
    local_epochs = 10
    num_clients = 10
    communication_rounds = 100  # Number of communication rounds

    model = OurModel(num_classes).to(device)
    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    criterion = CombinedLoss(dice_loss, ce_loss)  # loss function
    metrics = JaccardIndex(task='multiclass', num_classes=num_classes).to(device)

    client_dataloaders, val_loader = prepare_dataloaders(root_dir, batch_size, num_workers, num_clients, train_transform, val_transform)
    training_ious = []   # List to store training mIoU for each communication round
    validation_ious = [] # List to store validation mIoU for each communication round
    with open("feddyn.txt", "w") as f:
        for round in range(communication_rounds):
            local_weights = []
            local_losses = []
            local_ious = []

            for client in range(num_clients):
                print(f"Training on client {client + 1}/{num_clients}")
                client_model = OurModel(num_classes).to(device)
                client_model.load_state_dict(model.state_dict())
                client_optimizer = optim.AdamW(client_model.parameters(), lr=1e-3)

                client_weights, client_loss, client_iou, num_samples = train_feddyn_model(client_model, client_dataloaders[client], criterion, client_optimizer, metrics, device, local_epochs)

                local_weights.append(client_weights)
                local_losses.append(client_loss)
                local_ious.append(client_iou)
                
            global_weights = average_weights(local_weights)
            model.load_state_dict(global_weights)

            avg_loss = sum(local_losses) / len(local_losses)
            avg_iou = sum(local_ious) / len(local_ious)
            training_ious.append(avg_iou)

            val_loss, val_iou = validate(model, val_loader, criterion, metrics, device)
            validation_ious.append(val_iou)

            f.write(f"Round {round + 1}/{communication_rounds}\n")
            f.write(f"Train Loss: {avg_loss:.4f}, Train IoU: {avg_iou:.4f}\n")
            f.write(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}\n")


    plot_iou(training_ious, validation_ious, filename="iou_vs_rounds (FedDyn).png")
    
    model_save_path = "feddyn_model.pth"
    torch.save(model.state_dict(), model_save_path)  # Model saving
    print(f"Model saved to {model_save_path}")

    return model

if __name__ == "__main__":
    start_time = time.time()
    main_feddyn()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")

