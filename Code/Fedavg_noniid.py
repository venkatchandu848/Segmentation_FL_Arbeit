# Imports
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchmetrics.classification import JaccardIndex
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from train_central import MyCityscapes, get_transform, OurModel, validate, get_validation_transform
from Fedavg import CombinedLoss, pad_batch, encode_segmap
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, OneCycleLR

train_transform = get_transform()
val_transform = get_validation_transform()
ignore_index = 255

# Dirichlet function to create non-iid distribution
def partition_data_dirichlet(full_dataset, num_clients, alpha):
    # Step 1: Calculate total number of images
    total_images = len(full_dataset)
    print(f"Total number of images in the dataset: {total_images}")

    # Step 2: Sample proportions for each client using Dirichlet distribution
    client_proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
    print(f"Dirichlet client proportions: {client_proportions}")

    # Step 3: Scale proportions to total number of images
    num_images_per_client = (client_proportions * total_images).astype(int)
    print(f"Initial number of images per client: {num_images_per_client}")

    # Step 4: Ensure each client gets at least some minimum number of images
    min_images_per_client = 30  # Adjust as needed
    num_images_per_client = np.maximum(num_images_per_client, min_images_per_client)
    print(f"Adjusted minimum number of images per client: {num_images_per_client}")

    # Step 5: Calculate the actual number of images to assign
    total_assigned_images = np.sum(num_images_per_client)
    if total_assigned_images > total_images:
        # If total assigned images exceed total dataset images, reduce excess
        diff = total_assigned_images - total_images
        while diff > 0:
            # Choose a client randomly and reduce their assigned images
            reduce_client = np.random.randint(num_clients)
            if num_images_per_client[reduce_client] > min_images_per_client:
                num_images_per_client[reduce_client] -= 1
                diff -= 1

    elif total_assigned_images < total_images:
        # If total assigned images are less than total dataset images, increase deficit
        diff = total_images - total_assigned_images
        while diff > 0:
            # Choose a client randomly and increase their assigned images
            increase_client = np.random.randint(num_clients)
            num_images_per_client[increase_client] += 1
            diff -= 1
    print(f"Final number of images per client: {num_images_per_client}")

    assert np.sum(num_images_per_client) == total_images, "Total images assigned to clients do not match total images in dataset"

    # Step 6: Initialize data structures to hold partitioned data
    client_indices = [[] for _ in range(num_clients)]

    # Step 7: Shuffle all image indices
    all_indices = np.arange(total_images)
    np.random.shuffle(all_indices)

    # Step 8: Distribute shuffled indices to clients
    start_idx = 0
    for client_id in range(num_clients):
        end_idx = start_idx + num_images_per_client[client_id]
        client_indices[client_id] = all_indices[start_idx:end_idx]
        start_idx = end_idx

    return client_indices

# Dataloaders for non-iid
def prepare_dataloaders_noniid(root_dir, batch_size, num_workers, num_clients, alpha, train_transform, val_transform):
    # Load the entire dataset
    full_dataset = MyCityscapes(root=root_dir, split='train', mode='fine', transform=train_transform)

    # Partition data using Dirichlet distribution
    client_indices = partition_data_dirichlet(full_dataset, num_clients, alpha)

    assert sum(len(indices) for indices in client_indices) == len(full_dataset), \
        "Error: Total sum of client images does not equal total dataset size"
    
    client_dataloaders = []
    for indices in client_indices:
        client_subset = Subset(full_dataset, indices)
        client_loader = DataLoader(client_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        client_dataloaders.append(client_loader)

    # Validation dataset
    val_dataset = MyCityscapes(root=root_dir, split='val', mode='fine', transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return client_dataloaders, val_loader

# Weighted average for weights aggregation on server side
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

# Function for plotting mIOU vs Communication rounds
def plot_iou(training_ious, validation_ious, filename="iou_vs_rounds (Fedavg non-iid).png"):
    
    plt.figure(figsize=(10, 6))
    rounds = range(1, len(training_ious) + 1)
    # Plot Training mIoU
    plt.plot(rounds, training_ious, color='skyblue', label='Training mIoU')
    
    # Plot Validation mIoU
    plt.plot(rounds, validation_ious, color='purple', label='Validation mIoU')
    
    plt.title('Training and Validation mIoU vs Communication Rounds for Fedavg non-iid model')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Performance  Metric: mIoU')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(filename)  # Save the plot to a file
    plt.close()  # Close the plot to free up memory

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

# Main loop
def main_fedavg_noniid():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")  #cuda device

    root_dir = '/data/raml/group1/chandu/mmsegmentation/data/cityscapes'  # Root directory of dataset
    num_classes = 20
    batch_size = 8
    num_workers = 4 
    local_epochs = 10
    num_clients = 10
    communication_rounds = 100  # Number of communication rounds
    alpha = 0.6   # Parameter for non-iid distribution

    model = OurModel(num_classes).to(device)
    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=0.1)
    criterion = CombinedLoss(dice_loss, ce_loss)
    metrics = JaccardIndex(task='multiclass', num_classes=num_classes).to(device)

    client_dataloaders, val_loader = prepare_dataloaders_noniid(root_dir, batch_size, num_workers, num_clients, alpha, train_transform=train_transform, val_transform=val_transform)
    training_ious = []   # List to store training mIoU for each communication round
    validation_ious = [] # List to store validation mIoU for each communication round

    with open("fedani.txt", "w") as f:
        for round in range(communication_rounds):
            local_weights = []
            local_losses = []
            local_ious = []
            local_sizes = []

            for client in range(num_clients):
                client_model = OurModel(num_classes).to(device)
                client_model.load_state_dict(model.state_dict())
                client_optimizer = optim.AdamW(client_model.parameters(), lr=1e-3, weight_decay=0.01)
                
                client_weights, client_loss, client_iou, num_samples = train_fedavg_model(client_model, client_dataloaders[client], criterion, client_optimizer, metrics, device, local_epochs)

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

        
    plot_iou(training_ious, validation_ious, filename="iou_vs_rounds (Fedavg non-iid).png")

    model_save_path = "fedavg_noniid_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    return model

if __name__ == "__main__":
    start_time = time.time()
    main_fedavg_noniid()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")
