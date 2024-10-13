import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from ITTR_model import ITTRGenerator  # Import the generator class
from CustomDataset import CustomDataset  # Assuming this is the custom dataset you already have
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.utils as vutils
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def save_input_output_images(input_tensor, output_tensor, folder_path, file_name="gen_img"):
    """
    Save input and output images as a horizontally stacked image.

    Args:
    - input_tensor (torch.Tensor): Tensor containing input images
    - output_tensor (torch.Tensor): Tensor containing output images
    - folder_path (str): Path to the folder where images will be saved
    - file_name (str): Name of the saved image file (without extension)

    Returns:
    - None
    """
    os.makedirs(folder_path, exist_ok=True)
    
    # Convert tensors to numpy arrays and then to PIL images
    input_tensor = (input_tensor + 1) / 2
    output_tensor = (output_tensor + 1) / 2

    input_images = [(input_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8') for input_image in input_tensor]
    output_images = [(output_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8') for output_image in output_tensor]
    
    # Stack input and output images horizontally
    stacked_images = [Image.new('RGB', (input_image.shape[1] + output_image.shape[1], input_image.shape[0])) for input_image, output_image in zip(input_images, output_images)]
    for stacked_image, input_image, output_image in zip(stacked_images, input_images, output_images):
        stacked_image.paste(Image.fromarray(input_image), (0, 0))
        stacked_image.paste(Image.fromarray(output_image), (input_image.shape[1], 0))
    
    # Save the stacked images
    for i, stacked_image in enumerate(stacked_images):
        file_path = os.path.join(folder_path, f"{file_name}_{i}.png")
        stacked_image.save(file_path)




def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show(block=False)
    plt.pause(7)
    plt.close()


