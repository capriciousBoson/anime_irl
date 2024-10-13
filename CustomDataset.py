import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, anime_dir, real_dir, transform=None):
        self.anime_images = os.listdir(anime_dir)
        self.real_images = os.listdir(real_dir)
        self.anime_dir = anime_dir
        self.real_dir = real_dir
        self.transform = transform

    def __len__(self):
        return min(len(self.anime_images), len(self.real_images))

    def __getitem__(self, idx):
        anime_image = Image.open(os.path.join(self.anime_dir, self.anime_images[idx])).convert("RGB")
        real_image = Image.open(os.path.join(self.real_dir, self.real_images[idx])).convert("RGB")

        if self.transform:
            anime_image = self.transform(anime_image)
            real_image = self.transform(real_image)

        return anime_image, real_image

# Usage example
# from torchvision import transforms

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# dataset = CustomDataset('/path/to/dataset/anime', '/path/to/dataset/real', transform)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
