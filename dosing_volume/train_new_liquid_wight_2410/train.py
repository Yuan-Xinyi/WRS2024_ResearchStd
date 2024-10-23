import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


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


# Image augmentation and preprocessing
transform = transforms.Compose([
    transforms.RandomRotation((10,10)),  # Randomly rotate images
    transforms.Resize((200, 100)),  # Resize images
    # transforms.Pad(padding=50, fill=0, padding_mode='constant'),  # Fill empty space with black
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ColorJitter(contrast=0.5),  # Enhance contrast
    transforms.ToTensor(),  # Convert to tensor
])

'''test the imagae augmentation'''
image_path = 'dosing_volume/train_new_liquid_wight_2410/balance/separate/num_0/1.jpg'
image = Image.open(image_path)

transformed_image = transform(image)
transformed_image_np = transformed_image.numpy().squeeze()

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image)
ax[0].set_title('原始图像')
ax[0].axis('off')

ax[1].imshow(transformed_image_np, cmap='gray')
ax[1].set_title('变换后的图像')
ax[1].axis('off')

plt.show()



# # Define training, validation, and test datasets
# orig_dataset = DigitDataset(root_dir='dosing_volume/train_new_liquid_wight_2410/balance/separate')
# dataset = DigitDataset(root_dir='dosing_volume/train_new_liquid_wight_2410/balance/separate', transform=transform)
# # train_dataset = DigitDataset(root_dir='./dataset/train', transform=transform)
# # val_dataset = DigitDataset(root_dir='./dataset/val', transform=transform)
# # test_dataset = DigitDataset(root_dir='./dataset/test', transform=transform)

# # Create data loaders
# data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
# # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # Define Vision Transformer model
# class VisionTransformer(nn.Module):
#     def __init__(self, num_classes, embed_dim=128, num_heads=4, num_layers=6):
#         super(VisionTransformer, self).__init__()
#         self.patch_size = 10
#         self.img_size = (200, 100)
#         self.num_patches = (self.img_size[0] // self.patch_size) * (self.img_size[1] // self.patch_size)

#         # Patch embedding
#         self.patch_embeddings = nn.Conv2d(1, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

#         # Transformer Encoder
#         self.transformer_encoder = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
#             num_layers=num_layers
#         )

#         # Classification head
#         self.fc = nn.Linear(embed_dim, num_classes)

#     def forward(self, x):
#         # Patch embedding
#         x = self.patch_embeddings(x)  # [B, embed_dim, num_patches]
#         x = x.flatten(2)  # Flatten to [B, embed_dim, num_patches]
#         x = x.permute(0, 2, 1)  # Change to [B, num_patches, embed_dim]

#         # Transformer encoding
#         x = self.transformer_encoder(x)

#         # Classifier
#         x = x.mean(dim=1)  # Pooling
#         x = self.fc(x)  # [B, num_classes]
#         return x

# # Instantiate model, loss function, and optimizer
# model = VisionTransformer(num_classes=10)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training function
# def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
#     model.train()
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for images, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# # Validation function
# def validate_model(model, val_loader, criterion):
#     model.eval()
#     total = 0
#     correct = 0
#     with torch.no_grad():
#         for images, labels in val_loader:
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     accuracy = 100 * correct / total
#     print(f"Validation Accuracy: {accuracy:.2f}%")

# # Testing function
# def test_model(model, test_loader):
#     model.eval()
#     total = 0
#     correct = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     accuracy = 100 * correct / total
#     print(f"Test Accuracy: {accuracy:.2f}%")

# # # Train the model
# # train_model(model, train_loader, criterion, optimizer, num_epochs=10)

# # # Validate the model
# # validate_model(model, val_loader, criterion)

# # # Test the model
# # test_model(model, test_loader)
