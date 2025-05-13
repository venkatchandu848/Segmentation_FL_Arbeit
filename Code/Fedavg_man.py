# Imports
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchmetrics.classification import JaccardIndex
import torch.optim as optim
from train_central import get_transform, encode_segmap, get_validation_transform
from Fedavg import average_weights, prepare_dataloaders, pad_batch, CombinedLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, OneCycleLR

ignore_index = 255
max_norm = 10
train_transform = get_transform()
val_transform = get_validation_transform()

# Model modification to include MAN regularizer
class CustomDeepLabV3(nn.Module):
    def __init__(self, num_classes=21):
        super(CustomDeepLabV3, self).__init__()
        self.deeplabv3 = smp.DeepLabV3(encoder_name="mobilenet_v2", 
                                       encoder_weights="imagenet", 
                                       in_channels=3, 
                                       classes=num_classes)
    
    def forward(self, x):
        features = self.deeplabv3.encoder(x)
        x = features[-1]
        
        var1 = torch.mean(x**2, dim=[1, 2, 3])  # Variance after encoder
        
        x = self.deeplabv3.decoder(x)
        var2 = torch.mean(x**2, dim=[1, 2, 3])  # Variance after decoder
        
        x = self.deeplabv3.segmentation_head(x)
        var3 = torch.mean(x**2, dim=[1, 2, 3])  # Variance after segmentation head
        
        var_list = [var1, var2, var3]
        
        return {'out': x}, var_list

# Function to plot mIOU vs Communication rounds
def plot_iou(training_ious, validation_ious, filename="iou_vs_rounds (Fedavg_MAN).png"):

    plt.figure(figsize=(10, 6))
    rounds = range(1, len(training_ious) + 1)
    
    # Plot Training mIoU
    plt.plot(rounds, training_ious, color='skyblue', label='Training mIoU')
    
    # Plot Validation mIoU
    plt.plot(rounds, validation_ious, color='purple', label='Validation mIoU')
    
    plt.title('Training and Validation mIoU vs Communication Rounds for Fedavg MAN model')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Performance  Metric: mIoU')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(filename)  # Save the plot to a file
    plt.close()  # Close the plot to free up memory

# Training loop
def train_fedavg_man(model, train_loader, criterion, optimizer, metrics, device, epochs, mu, min_batch_size=4):
    model.train()
    model = model.to(device)
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_iou = 0
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            image, target = pad_batch(batch, min_batch_size)
            image, target = image.to(device), target.to(device)

            optimizer.zero_grad()
            output, var_list = model(image)  # output now includes variance list
            target = encode_segmap(target).long()
            loss = criterion(output['out'], target)
            
            var_loss = 0
            # Adding variance loss
            for var in var_list:
                var_loss += torch.mean(var)

            total_loss = loss + (mu/2) * var_loss # MAN regularizer
            
            total_loss.backward()

            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
                
            optimizer.step()
            iou = metrics(output['out'], target)
            batch_size = image.size(0)
            epoch_loss += total_loss.item() * batch_size
            epoch_iou += iou.item() * batch_size
            total_samples += batch_size
        
    avg_loss = epoch_loss / total_samples
    avg_iou = epoch_iou / total_samples
    
    return model.state_dict(), avg_loss, avg_iou

# Validation loop
def validate(model, dataloader, criterion, metrics, device, mu):
    model.eval()
    epoch_loss = 0
    epoch_iou = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            image, segment = batch
            image, segment = image.to(device), segment.to(device)

            output, var_list  = model(image)
            segment = encode_segmap(segment).long()

            loss = criterion(output['out'], segment)
            iou = metrics(output['out'], segment)

            var_loss = 0
            # Adding variance loss
            for var in var_list:
                var_loss += torch.mean(var)

            total_loss = loss + (mu/2) * var_loss

            batch_size = image.size(0)
            epoch_loss += total_loss.item() * batch_size
            epoch_iou += iou.item() * batch_size
            total_samples += batch_size

    avg_loss = epoch_loss / total_samples
    avg_iou = epoch_iou / total_samples
    
    return avg_loss, avg_iou

# Main loop
def main_fedavg_man():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # cuda device

    root_dir = '/data/raml/group1/chandu/mmsegmentation/data/cityscapes' # Root directory of dataset
    num_classes = 20
    batch_size = 4
    num_workers =  4
    local_epochs = 10
    num_clients = 10
    communication_rounds = 100
    mu = 0.1  # Regularization parameter for FedMAN

    model = CustomDeepLabV3(num_classes).to(device)
    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    criterion = CombinedLoss(dice_loss, ce_loss)
    metrics = JaccardIndex(task='multiclass', num_classes=num_classes).to(device)

    client_dataloaders, val_loader = prepare_dataloaders(root_dir, batch_size, num_workers, num_clients, train_transform=train_transform, val_transform=val_transform)
    training_ious = []   # List to store training mIoU for each communication round
    validation_ious = [] # List to store validation mIoU for each communication round

    with open("fedavg_man.txt", "w") as f:
        for round in range(communication_rounds):
            local_weights = []
            local_losses = []
            local_ious = []
            
            for client in range(num_clients):
                client_model = CustomDeepLabV3(num_classes).to(device)
                client_model.load_state_dict(model.state_dict())
                client_optimizer = optim.AdamW(client_model.parameters(), lr=1e-3)

                client_weights, client_loss, client_iou = train_fedavg_man(client_model, client_dataloaders[client], criterion, client_optimizer, metrics, device, local_epochs, mu)
                
                local_weights.append(client_weights)
                local_losses.append(client_loss)
                local_ious.append(client_iou)
            
            global_weights = average_weights(local_weights)
            model.load_state_dict(global_weights)
            
            avg_loss = sum(local_losses) / len(local_losses)
            avg_iou = sum(local_ious) / len(local_ious)
            training_ious.append(avg_iou)

            val_loss, val_iou = validate(model, val_loader, criterion, metrics, device, mu=mu)
            validation_ious.append(val_iou)
            
            f.write(f"Round {round + 1}/{communication_rounds}\n")
            f.write(f"Train Loss: {avg_loss:.4f}, Train IoU: {avg_iou:.4f}\n")
            f.write(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}\n")

    plot_iou(training_ious, validation_ious, filename="iou_vs_rounds (Fedavg MAN).png")  # Plotting mIOU vs Communication rounds
    
    model_save_path = "fedavg_man_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    return model

if __name__ == "__main__":
    start_time = time.time()
    main_fedavg_man()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")