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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

# Define Vision Transformer model
class VisionTransformer(nn.Module):
    def __init__(self, num_classes, embed_dim=128, num_heads=4, num_layers=6):
        super(VisionTransformer, self).__init__()
        self.patch_size = 10
        self.img_size = (200, 100)
        self.num_patches = (self.img_size[0] // self.patch_size) * (self.img_size[1] // self.patch_size)

        # Patch embedding
        self.patch_embeddings = nn.Conv2d(1, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=num_layers
        )

        # Classification head
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embeddings(x)  # [B, embed_dim, num_patches]
        x = x.flatten(2)  # Flatten to [B, embed_dim, num_patches]
        x = x.permute(0, 2, 1)  # Change to [B, num_patches, embed_dim]

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Classifier
        x = x.mean(dim=1)  # Pooling
        x = self.fc(x)  # [B, num_classes]
        return x

# Instantiate model, loss function, and optimizer
model = VisionTransformer(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and validation function
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        # for images, labels in enumerate(train_loader):
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            # print(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Validation Accuracy: {accuracy:.2f}%")

# Testing function
def test_model(model, test_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# Train and validate the model
train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

# Test the model
test_model(model, test_loader)
