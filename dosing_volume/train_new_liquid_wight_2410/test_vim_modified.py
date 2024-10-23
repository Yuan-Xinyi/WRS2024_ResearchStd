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


model = TransformerModel(input_dim=100, num_classes=10)
model.load_state_dict(torch.load('/home/lqin/wrs_2024/dosing_volume/train_new_liquid_wight_2410/trained_model/vit_model.pth'))
model.eval()
image_path='dosing_volume/train_new_liquid_wight_2410/balance/separate/num_0/2_sl_sl.jpg'
image = Image.open(image_path)

image_tensor = transform(image).unsqueeze(0)  # 添加 batch_size

with torch.no_grad():
    output = model(image_tensor)
    output_logits = output.squeeze().numpy()
    output_probs = torch.nn.functional.softmax(output, dim=1).squeeze().numpy()

    # 打印模型预测的原始值和概率
    print("模型原始输出 logits:", output_logits)
    print("预测概率:", output_probs)

# 4. 可视化输入图像和预测结果
plt.figure(figsize=(14, 6))

# 输入图像
plt.subplot(1, 3, 1)
plt.title("Input Image")
plt.imshow(image, cmap='gray')

# 模型原始输出（logits）
plt.subplot(1, 3, 2)
plt.title("Model Logits")
plt.bar(range(len(output_logits)), output_logits)
plt.xlabel('Class')
plt.ylabel('Logit Value')

# 预测概率
plt.subplot(1, 3, 3)
plt.title("Prediction Probabilities")
plt.bar(range(len(output_probs)), output_probs)
plt.xlabel('Class')
plt.ylabel('Probability')

plt.tight_layout()
plt.show()

