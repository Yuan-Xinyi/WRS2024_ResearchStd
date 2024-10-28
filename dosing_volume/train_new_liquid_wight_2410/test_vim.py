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

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes,n_heads=4,num_encoder_layers=3):
        super(TransformerModel, self).__init__()
        self.positional_encoding=nn.Parameter(torch.zeros(1, 200, 100))
        encoder_layers=self.transformer=nn.TransformerEncoderLayer(d_model=100,nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.fc = nn.Linear(100, num_classes)
        
    def forward(self, x):
        # batch_size is 16, image size is 28x28, here x is torch.Size([16, 1, 28, 28])
        x = x.view(x.size(0), 200, 100)  # Reshape to torch.Size([16, 200, 100])
        # Add positional encoding
        pos_enc = self.positional_encoding
        x += pos_enc
        # Permute to (sequence_length, batch_size, embedding_dim) for transformer
        x = x.permute(1, 0, 2)  # torch.Size([200, 16, 100])
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # torch.Size([200, 16, 100])
        # Average over the sequence length dimension
        x = x.mean(dim=0)  # torch.Size([16, 100])
        # Pass through fully connected layer
        x = self.fc(x)  # torch.Size([16, 10]) [batch_size, num_classes]
        
        return x

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


transform = transforms.Compose([
    transforms.Resize((200, 100)),  # Resize the image to a fixed size
    # transforms.RandomRotation(10),  # Rotate the image with expansion
    # FillEdgesTransform(),  # Fill edges to avoid black borders
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ColorJitter(contrast=1.0),  # Enhance contrast
    transforms.ToTensor(),  # Convert to tensor
])

def test_image_augmentation(image_path):
    image = Image.open(image_path)

    transformed_image = transform(image)
    transformed_image_np = transformed_image.numpy().squeeze()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image)
    ax[0].set_title('original image')
    ax[0].axis('off')

    ax[1].imshow(transformed_image_np, cmap='gray')
    ax[1].set_title('image after transformation')
    ax[1].axis('off')

    plt.show()

def test_single_image(model_path, test_image_path):

    model = TransformerModel(input_dim=100, num_classes=10)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    image = Image.open(test_image_path)

    image_tensor = transform(image).unsqueeze(0)  # add batch_size

    with torch.no_grad():
        output = model(image_tensor)
        output_logits = output.squeeze().numpy()
        output_probs = torch.nn.functional.softmax(output, dim=1).squeeze().numpy()

        # Print model's raw output logits and probabilities
        print("Model raw output logits:", output_logits)
        print("Prediction probabilities:", output_probs)

    # 4. Visualize input image and prediction results
    plt.figure(figsize=(14, 6))

    # Input image
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(image, cmap='gray')

    # Model raw output (logits)
    plt.subplot(1, 3, 2)
    plt.title("Model Logits")
    plt.bar(range(len(output_logits)), output_logits)
    plt.xlabel('Class')
    plt.ylabel('Logit Value')

    # Prediction probabilities
    plt.subplot(1, 3, 3)
    plt.title("Prediction Probabilities")
    plt.bar(range(len(output_probs)), output_probs)
    plt.xlabel('Class')
    plt.ylabel('Probability')

    plt.tight_layout()
    plt.show()

def evaluate_entire_dataset(model_path, dataloader):
    model = TransformerModel(input_dim=100, num_classes=10)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Accuracy of the model on the dataset: {accuracy * 100:.2f}%')

if __name__ == "__main__":
    '''test the imagae augmentation'''
    # test_image_augmentation('dosing_volume/train_new_liquid_wight_2410/balance/separate/num_0/1.jpg')
    
    '''test the model with single image'''
    # model_path = '/home/lqin/wrs_2024/dosing_volume/train_new_liquid_wight_2410/results/vim_1028_1440/epoch8.pth'  # accuracy = 0.90
    # test_image_path = 'dosing_volume/train_new_liquid_wight_2410/balance/separate/num_7/108_rc.jpg'
    # # model_path = '/home/lqin/wrs_2024/dosing_volume/train_new_liquid_wight_2410/trained_model/vit_model_epoch1.pth'
    # test_single_image(model_path=model_path, test_image_path=test_image_path)

    '''test all the images in seperate dataset without transformation'''
    full_dataset = DigitDataset(root_dir='dosing_volume/train_new_liquid_wight_2410/balance/separate', transform = transform)
    dataloader = DataLoader(full_dataset, batch_size=1, shuffle=False)
    model_path = '/home/lqin/wrs_2024/dosing_volume/train_new_liquid_wight_2410/results/vim_1028_1440/epoch8.pth'  # accuracy = 0.90
    evaluate_entire_dataset(model_path, dataloader) # accuracy = 0.908
