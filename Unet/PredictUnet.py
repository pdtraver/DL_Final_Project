# %%
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from PIL import Image

# %%
if torch.cuda.is_available():
    # Set the default device to CUDA
    device = torch.device('cuda')
    torch.set_default_device(device)
    print('Using CUDA for tensor operations')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU for tensor operations')

# %%
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.down1 = self.contract_block(3, 64, 7, 3)
        self.down2 = self.contract_block(64, 128, 3, 1)
        self.down3 = self.contract_block(128, 256, 3, 1)
        self.down4 = self.contract_block(256, 512, 3, 1)

        # Decoder
        self.up3 = self.expand_block(512, 256, 3, 1)
        self.up2 = self.expand_block(256, 128, 3, 1)
        self.up1 = self.expand_block(128, 64, 3, 1)
        self.final_up = nn.ConvTranspose2d(64, 49, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final = nn.Conv2d(49, 49, kernel_size=1)  # Change from 1 to 48

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        # Decoder
        x = self.up3(x4)
        x = self.up2(x + x3)
        x = self.up1(x + x2)
        x = self.final_up(x + x1)
        x = self.final(x)

        return x

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        return expand

# %%
# Initialize the model
model = UNet().to(device)

# Path to your saved checkpoint
checkpoint_path = "TrainUnet/Unet_checkpoint_epoch_89.pt"  # Replace xx with the actual epoch number

# Load the checkpoint
checkpoint = torch.load(checkpoint_path)

# Update model's state dictionary
model.load_state_dict(checkpoint)

# If you're using the model for inference
model.eval()

# %%
class CLEVRDataset(Dataset):
    def __init__(self, path, transform=None):
        self.video_paths = [os.path.join(path, dir_path) for dir_path in os.listdir(path) if dir_path.startswith('video')]
        self.transform = transform
        self._get_num_samples()
        
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),  # Converts PIL Image to Tensor.
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard normalization for pre-trained models.
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return self.num_samples

    def _get_num_samples(self):
        self.num_samples = 22 * len(self.video_paths)

    def __getitem__(self, idx):
        image_index = idx % 22
        folder_index = int(idx/22)
        
        img_name = os.path.join(self.video_paths[folder_index], f'image_{image_index}.png')
        
        image = Image.open(img_name).convert("RGB")

        image = self.transform(image).to(device)

        return image, self.video_paths[folder_index]


# %%
batch_size = 22
# Dataset and DataLoader
dataset = CLEVRDataset(path='DL_Final_Project/dataset/unlabeled')

# Assuming 'dataset' is already defined
generator = torch.Generator(device='cuda')
sampler = SequentialSampler(dataset)  # Use SequentialSampler to maintain order
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)

# %%
for images, folder in dataloader:
    images = images.to(device)  # Assuming 'device' is defined, e.g., device = torch.device("cuda")
    save_path = os.path.join("UnetData_89_epoch", folder[0], "mask.npy")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    outputs = model(images)  # Assuming 'model' is defined and properly configured
    argmax_result = torch.argmax(outputs, dim=1)
    np.save(save_path, argmax_result.cpu().detach().numpy())  # Save the outputs as a NumPy array

# %%



