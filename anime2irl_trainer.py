import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16
from torchvision import transforms
from ITTR_model import ITTRGenerator  # Import the generator class
from CustomDataset import CustomDataset  # Assuming this is the custom dataset you already have
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.utils as vutils

# Training parameters
lr = 0.0002
betas = (0.5, 0.999)
batch_size = 8
num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_dir = './checkpoints'  # Directory to save checkpoints

# Loss functions
l1_loss = nn.L1Loss()

# VGG Perceptual loss function
vgg = vgg16(pretrained=True).features[:16].eval().to(device)
for param in vgg.parameters():
    param.requires_grad = False

# Function for perceptual loss
def perceptual_loss(vgg, gen_image, real_image):
    gen_features = vgg(gen_image)
    real_features = vgg(real_image)
    return l1_loss(gen_features, real_features)

# Function for cycle consistency loss
def cycle_consistency_loss(generator, fake_photo, real_anime):
    reconstructed_anime = generator(fake_photo)
    return l1_loss(reconstructed_anime, real_anime)

# Function to save model checkpoints
def save_checkpoint(epoch, model, optimizer, loss, file_name="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(checkpoint, os.path.join(checkpoint_dir, file_name))
    print(f"Checkpoint saved at epoch {epoch}")

# Function to load model checkpoints if available
def load_checkpoint(model, optimizer, file_name="checkpoint.pth"):
    checkpoint_path = os.path.join(checkpoint_dir, file_name)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded: starting from epoch {start_epoch}, loss: {loss}")
        return start_epoch, loss
    else:
        print("No checkpoint found, starting training from scratch")
        return 0, float('inf')  # Start from scratch if no checkpoint exists

# Function to generate outputs for test images and save them
def generate_and_save_test_outputs(generator, epoch,test_dir,checkpoint_output_dir):
    if not os.path.exists(checkpoint_output_dir):
        os.makedirs(checkpoint_output_dir)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize for Tanh
    ])
    
    for file in os.listdir(test_dir):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(test_dir, file)
            image = Image.open(img_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
            
            # Generate output
            generator.eval()  # Set generator to evaluation mode
            with torch.no_grad():
                generated_image = generator(input_tensor)
            
            # Save output image
            output_image_path = os.path.join(checkpoint_output_dir, f"{file.split('.')[0]}_epoch_{epoch}.png")
            vutils.save_image(generated_image, output_image_path, normalize=True)
            print(f"Generated output saved for {file} at epoch {epoch}")

# Training loop
def train(generator, dataloader, optimizer, num_epochs):
    generator.train()

    start_epoch, prev_loss = load_checkpoint(generator, optimizer)

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        
        for batch in dataloader:
            real_anime, real_photo = batch['anime'].to(device), batch['photo'].to(device)
            
            # Generate fake photorealistic images from anime images
            fake_photo = generator(real_anime)
            
            # Calculate L1 pixel loss
            pixel_loss = l1_loss(fake_photo, real_photo)
            
            # Perceptual loss
            vgg_loss = perceptual_loss(vgg, fake_photo, real_photo)
            
            # Cycle consistency loss
            cycle_loss = cycle_consistency_loss(generator, fake_photo, real_anime)
            
            # Total loss
            total_loss = pixel_loss + 0.1 * vgg_loss + 10 * cycle_loss  # Adjust weights as necessary
            
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item()}")

        # Save checkpoint every 10 epochs
        test_dir = 'Dataset\testA'
        checkpoint_output_dir = 'output'
        if (epoch + 1) % 10 == 0:
            save_checkpoint(epoch + 1, generator, optimizer, total_loss.item())
            generate_and_save_test_outputs(generator, epoch + 1,test_dir,checkpoint_output_dir)

# Main function
if __name__ == "__main__":
    # Initialize the generator
    generator = ITTRGenerator().to(device)
    
    # Optimizer
    optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    
    # Dataset and DataLoader
    custom_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize for Tanh
    ])
    dataset = CustomDataset('Dataset\trainA', 'Dataset\trainB', custom_transforms)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Start training
    train(generator, dataloader, optimizer, num_epochs)
