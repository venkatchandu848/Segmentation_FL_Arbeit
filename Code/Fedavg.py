# Imports
from train_central import OurModel, validate, get_transform, get_validation_transform,  encode_segmap, MyCityscapes, decode_segmap
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch
from torchmetrics.classification import JaccardIndex
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, OneCycleLR
import time
import matplotlib.pyplot as plt

ignore_index=255
train_transform = get_transform()
val_transform = get_validation_transform()

# Preparing Data loader for clients 
def prepare_dataloaders(root_dir, batch_size, num_workers, num_clients, train_transform, val_transform, seed=42):  
# Load the entire dataset
    full_dataset = MyCityscapes(root=root_dir, split='train', mode='fine', transform=train_transform)

# Split dataset indices for each client
    indices = list(range(len(full_dataset)))
    np.random.seed(seed)
    np.random.shuffle(indices)

    client_splits = np.array_split(indices, num_clients)

    client_dataloaders = []
    for i, split in enumerate(client_splits):
        client_subset = Subset(full_dataset, split)
        client_loader = DataLoader(client_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        client_dataloaders.append(client_loader)

    # Validation dataset
    val_dataset = MyCityscapes(root=root_dir, split='val', mode='fine', transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return client_dataloaders, val_loader

# padding batch
def pad_batch(batch, min_batch_size):
    image, target = batch
    if len(image) < min_batch_size:
        padding_size = min_batch_size - len(image)
        image = torch.cat([image, image[:padding_size]], dim=0)
        target = torch.cat([target, target[:padding_size]], dim=0)
    return image, target

# Training loop
def train_fedavg_model(model, train_loader, criterion, optimizer, metrics, device, epochs, min_batch_size=4, max_norm = 10):
    model.train()
    model = model.to(device)

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_iou = 0
        total_samples = 0
        metrics.reset()
        for batch_idx, batch in enumerate(train_loader):
            image, target = batch
            image, target = pad_batch(batch, min_batch_size)
            image, target = image.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(image)
            target = encode_segmap(target).long()
            loss = criterion(output, target)
            loss.backward()

            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm)  # Clip gradients
                
            optimizer.step()
            
            iou = metrics(output, target)
            batch_size = image.size(0)
            epoch_loss += loss.item() * batch_size
            epoch_iou += iou.item() * batch_size
            total_samples += batch_size

    avg_loss = epoch_loss / total_samples
    avg_iou = epoch_iou / total_samples

    return model.state_dict(), avg_loss, avg_iou, total_samples


# Averaging weights (done on server side)
def average_weights(weights):
    avg_weights = {}
    for key in weights[0].keys():
        param_tensors = [w[key] for w in weights]
        if torch.is_floating_point(param_tensors[0]) or param_tensors[0].is_complex():
            avg_weights[key] = torch.stack(param_tensors).mean(dim=0)
        else:
            avg_weights[key] = torch.stack(param_tensors).float().mean(dim=0).to(param_tensors[0].dtype)
    return avg_weights


# Combined loss: Cross entropy loss + Dice Loss
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
def plot_iou(training_ious, validation_ious, filename="iou_vs_rounds (Fedavg).png"):

    plt.figure(figsize=(10, 6))
    rounds = range(1, len(training_ious) + 1)
    # Plot Training mIoU
    plt.plot(rounds, training_ious, color='skyblue', label='Training mIoU')
    
    # Plot Validation mIoU
    plt.plot(rounds, validation_ious, color='purple', label='Validation mIoU')
    
    plt.title('Training and Validation mIoU vs Communication Rounds for Fedavg model')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Performance  Metric: mIoU')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(filename)  # Save the plot to a file
    plt.close()  # Close the plot to free up memory

def visualize_segmentation(model, dataloader, device, filename="segmentation_visualization (Fedavg).png"):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images, targets = batch
            images, targets = images.to(device), targets.to(device)
            
            # Get the model predictions
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Move images, targets, and preds to CPU for visualization
            images = images.cpu()
            targets = targets.cpu()
            preds = preds.cpu()

            # Select the first image from the batch for visualization
            image = images[0].permute(1, 2, 0)  # Convert from CxHxW to HxWxC
            target = targets[0].cpu().numpy()
            pred = preds[0].cpu().numpy()

            ground_truth_rgb = decode_segmap(encode_segmap(target))
            pred_rgb = decode_segmap(pred)


            # Plot original image, ground truth, and predicted mask
            plt.figure(figsize=(18, 6))
            
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title('Original Image')
            
            plt.subplot(1, 3, 2)
            plt.imshow(ground_truth_rgb)
            plt.title('Ground Truth')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(pred_rgb)
            plt.title('Predicted Mask')
            plt.axis('off')

            plt.savefig(filename)  # Save the plot to a file
            plt.close()  # Close the plot to free up memory
            break  # Only visualize the first batch

# Main loop
def main_fedavg():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Cuda device
    root_dir = '/data/raml/group1/chandu/mmsegmentation/data/cityscapes'  # root directory of images
    num_classes = 20   # Compressing it to 20 classes. 
    batch_size = 4
    num_workers = 4
    local_epochs = 10
    num_clients = 10
    communication_rounds = 100  # Number of communication rounds

    model = OurModel(num_classes).to(device)
    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    criterion = CombinedLoss(dice_loss, ce_loss)
    metrics = JaccardIndex(task='multiclass', num_classes=num_classes).to(device)  # mIOU metric

    client_dataloaders, val_loader = prepare_dataloaders(root_dir, batch_size, num_workers, num_clients, train_transform=train_transform, val_transform=val_transform)
    training_ious = []   # List to store training mIoU for each communication round
    validation_ious = [] # List to store validation mIoU for each communication round

    with open("fedavg.txt", "w") as f:
        for round in range(communication_rounds):
            local_weights = []
            local_losses = []
            local_ious = []

            for client in range(num_clients):
                f.write(f"Training on client {client + 1}/{num_clients}\n")
                client_model = OurModel(num_classes).to(device)
                client_model.load_state_dict(model.state_dict())
                client_optimizer = optim.AdamW(client_model.parameters(), lr=1e-3)

                client_weights, client_loss, client_iou, num_samples = train_fedavg_model(client_model, client_dataloaders[client], criterion, client_optimizer, metrics, device, local_epochs)

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

    # Plot and save training and validation mIoU vs communication rounds
    plot_iou(training_ious, validation_ious, filename="iou_vs_rounds (Fedavg).png")

    # Visualize and save the segmentation results
    visualize_segmentation(model, val_loader, device, filename="segmentation_results.png")

    model_save_path = "fedavg_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    return model

if __name__ == "__main__":
    start_time = time.time()
    main_fedavg()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")