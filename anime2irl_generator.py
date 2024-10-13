import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from ITTR_model import ITTRGenerator

# Function to load the model architecture
def load_model():
    # Initialize your model architecture
    model = ITTRGenerator()  # Replace with your actual model class
    return model

# Function to load the model weights and generate an image
def generate_image(input_image_path, output_image_path, model_weights_path, device='cpu'):
    # Load your model
    model = load_model()

    checkpoint_path = model_weights_path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load the model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set the model to evaluation mode
    model.eval()
    
    # Define image transformation (same as during training)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Adjust based on your model's input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Adjust if you used different normalization
    ])
    

    # Load and preprocess the input image
    input_image = Image.open(input_image_path)
    input_tensor = transform(input_image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Perform inference (disable gradient calculation for efficiency)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Post-process the output tensor
    output_tensor = output_tensor.squeeze(0).cpu()  # Remove batch dimension
    output_tensor = (output_tensor * 0.5) + 0.5  # De-normalize if you used Normalize
    output_image = transforms.ToPILImage()(output_tensor)

    # Stack the test and generated images horizontally
    combined_image = Image.new('RGB', (input_image.width + output_image.width, input_image.height))
    combined_image.paste(input_image, (0, 0))
    combined_image.paste(output_image, (input_image.width, 0))
    
    # Save the output image
    combined_image.save(output_image_path)
    
    # Optionally display the generated image
    plt.imshow(combined_image)
    plt.axis('off')
    plt.show()

# Example usage
input_image_path = 'Dataset/testA/frame_6740_0.png'
output_image_path = 'output'
saved_model = 'ITTR_checkpoint_epoch_20.pth'
generate_image(input_image_path, output_image_path, saved_model)
