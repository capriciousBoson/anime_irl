import argparse
import time
import os
import torch
import torchvision.utils as vutils
from CustomDataset import CustomDataset  # Importing the CustomDataset class
from torchvision import transforms
import sys

# Define a function to save model checkpoints
def save_checkpoint(model, optimizer, epoch, loss, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filename)

# Define a function to load model checkpoints
def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    return start_epoch, checkpoint['loss']

# Define a function to generate and save images
def generate_and_save_images(model, dataloader, epoch, save_dir):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for anime_images, _ in dataloader:
            # Generate images
            generated_images = model(anime_images)
            # Create a grid of images
            grid = vutils.make_grid(generated_images, nrow=4, normalize=True)
            # Save the generated image grid
            vutils.save_image(grid, os.path.join(save_dir, f'generated_epoch_{epoch}.png'))
            break  # Just generate one batch for visualization

# Define the training method
def train(ittr_model, optimizer, criterion, dataloader, resume, num_epochs=10, checkpoint_interval=5, max_training_time=3600):
    # Directory to save generated images and model checkpoints
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('generated_images', exist_ok=True)

    # Load from a checkpoint if requested
    checkpoint_path = 'checkpoints/model_epoch_last.pth'
    start_epoch = 0

    if resume and os.path.exists(checkpoint_path):
        start_epoch, _ = load_checkpoint(ittr_model, optimizer, checkpoint_path)
        print(f'Resuming training from epoch {start_epoch + 1}')
    else:
        print('Starting training from scratch.')

    # Start training timer
    start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        ittr_model.train()  # Set the model to training mode
        for anime_images, real_images in dataloader:
            optimizer.zero_grad()

            # Forward pass
            outputs = ittr_model(anime_images)

            # Compute loss
            loss = criterion(outputs, real_images)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Save the model checkpoint every checkpoint_interval epochs
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_filename = f'checkpoints/model_epoch_{epoch + 1}.pth'
            save_checkpoint(ittr_model, optimizer, epoch + 1, loss.item(), checkpoint_filename)
            print(f'Model checkpoint saved: {checkpoint_filename}')

            # Generate and save images after checkpointing
            generate_and_save_images(ittr_model, dataloader, epoch + 1, 'generated_images')
            print(f'Generated images saved for epoch {epoch + 1}')

        # Check if the maximum training time has been reached
        if time.time() - start_time > max_training_time:
            print("Maximum training time reached. Stopping training.")
            break

    # Save the last checkpoint at the end of training
    save_checkpoint(ittr_model, optimizer, epoch + 1, loss.item(), 'checkpoints/model_epoch_last.pth')
    print("Last checkpoint saved for resuming future training.")

# Define the main function
def main():
    # Argument parser setup
    # parser = argparse.ArgumentParser(description='Train ITTR model for image translation.')
    # parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')
    # parser.add_argument('--anime_dir', type=str, required=True, help='Directory of anime images')
    # parser.add_argument('--real_dir', type=str, required=True, help='Directory of real images')
    # args = parser.parse_args()

    Resume = sys.argv[1]

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    anime_dir = 'Dataset\testA'
    real_dir = 'Dataset\testB'

    dataset = CustomDataset(anime_dir, real_dir, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize model, optimizer, and loss criterion
    ittr_model = ...  # Initialize your ITTR model here
    optimizer = torch.optim.Adam(ittr_model.parameters(), lr=1e-4)
    criterion = ...  # Define your loss function here

    # Start training
    train(ittr_model, optimizer, criterion, dataloader, Resume)

if __name__ == '__main__':
    main()
