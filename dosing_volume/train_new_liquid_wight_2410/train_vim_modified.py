import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F


class FillEdgesTransform:
    def __call__(self, image):
        img_array = np.array(image)
        # Create a mask for black pixels
        mask = (img_array == 0).all(axis=-1)  # Identify black pixels

        # Get the color to fill (e.g., using the edge pixels)
        if np.any(mask):
            edge_color = img_array[~mask].mean(axis=0)  # Average color of non-black pixels
            # Mix the edge color with white to make it lighter
            lighter_color = edge_color * 0.93 + np.array([255, 255, 255]) * 0.07  # Adjust the ratio as needed
            img_array[mask] = lighter_color  # Fill black pixels with the lighter color
        
        return Image.fromarray(img_array.astype(np.uint8))

# Custom dataset class
class DigitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_labels = self._load_labels()

    def _load_labels(self):
        labels = []
        # Iterate over each subfolder (digit folder) in the root directory
        for label in range(10):  # Assuming folders are named num_0, num_1, num_2, ..., num_9
            label_dir = os.path.join(self.root_dir, 'num_' + str(label))
            if os.path.isdir(label_dir):
                # Iterate over each image file in the subfolder
                for img_file in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_file)
                    # Only add file paths and corresponding labels (digits)
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):  # Filter for common image formats
                        labels.append((img_path, label))  # Use folder name as label
        return labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path).convert("RGB")  # Read image and convert to RGB
        if self.transform:
            image = self.transform(image)
        return image, label

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15):
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Randomly split the dataset into train, validation, and test sets
    train_dataset, val_test_dataset = random_split(dataset, [train_size, total_size - train_size])
    val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_loader, val_loader, test_loader


transform = transforms.Compose([
    transforms.Resize((200, 100)),  # Resize the image to a fixed size
    transforms.RandomRotation(10),  # Rotate the image with expansion
    FillEdgesTransform(),  # Fill edges to avoid black borders
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ColorJitter(contrast=1.0),  # Enhance contrast
    transforms.ToTensor(),  # Convert to tensor
])

'''test the imagae augmentation'''
# image_path = 'dosing_volume/train_new_liquid_wight_2410/balance/separate/num_0/1.jpg'
# image = Image.open(image_path)

# transformed_image = transform(image)
# transformed_image_np = transformed_image.numpy().squeeze()

# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].imshow(image)
# ax[0].set_title('原始图像')
# ax[0].axis('off')

# ax[1].imshow(transformed_image_np, cmap='gray')
# ax[1].set_title('变换后的图像')
# ax[1].axis('off')

# plt.show()



# Define training, validation, and test datasets
full_dataset = DigitDataset(root_dir='dosing_volume/train_new_liquid_wight_2410/balance/separate', transform=transform)
train_loader, val_loader, test_loader = split_dataset(full_dataset)

# Check dataset sizes
print(f"Train dataset size: {len(train_loader.dataset)}")
print(f"Validation dataset size: {len(val_loader.dataset)}")
print(f"Test dataset size: {len(test_loader.dataset)}")


class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes,n_heads=4,num_encoder_layers=3):
        super(TransformerModel, self).__init__()
        self.positional_encoding=nn.Parameter(torch.zeros(1, 200, 100))
        encoder_layers=self.transformer=nn.TransformerEncoderLayer(d_model=100,nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.fc = nn.Linear(100, num_classes)
        
    def forward(self,x):
        #batch_size为16，图片是28×28，这里 x  torch.Size([16, 1, 28, 28])
        x=x.view(x.size(0),200,100)   #torch.Size([16,  28, 28])
        pos_enc = self.positional_encoding
        x+= pos_enc
        x = x.permute(1, 0, 2)  # 转换为trans需要 (sequence_length, batch_size, embedding_dim)
        x=self.transformer_encoder(x)        #torch.Size([28 ,16, 28])
        x=x.mean(dim=0)                      #[sequence_length, batch_size, embedding_dim]
        x=self.fc(x)
        return x                         #torch.Size([16，10])    [batch_size，num_classes]
model = TransformerModel(input_dim=100, num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)   


# Training and validation function
def train(model,train_loader,criterion,optimizer,epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            
# Testing function
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({100. * correct / len(test_loader.dataset):.0f}%)\n')

# 训练和评估
for epoch in range(1, 9):
    print(f"Epoch {epoch}")
    train(model, train_loader, criterion, optimizer, epoch)
    torch.save(model.state_dict(), f'./vit_model_epoch{epoch}.pth')
    # Test the model
    test(model, test_loader, criterion)


# file_path = 'dosing_volume/train_new_liquid_wight_2410/trained_model/'
# if not os.path.exists(file_path):
#     os.makedirs(file_path)

# path = file_path+f'vim_1023.pth'
# torch.save(model.state_dict(), path)