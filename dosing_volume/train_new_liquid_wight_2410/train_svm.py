import os
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Custom dataset class to load the images
class DigitDataset:
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
                for img_file in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_file)
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):
                        labels.append((img_path, label))
        return labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path).convert("RGB")  # Read image and convert to RGB
        if self.transform:
            image = self.transform(image)
        return np.array(image), label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((200, 100)),  # Resize the image to a fixed size
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor()  # Convert to tensor
])

# Load the dataset
dataset = DigitDataset(root_dir='dosing_volume/train_new_liquid_wight_2410/balance/separate', transform=transform)

# Prepare data and labels
data = []
labels = []

for i in tqdm(range(len(dataset)), desc="Loading dataset", unit="image"):
    img, label = dataset[i]
    img = img.flatten()  # Flatten the image to a 1D vector for SVM
    data.append(img)
    labels.append(label)

data = np.array(data)
labels = np.array(labels)

# Split data into training and test sets (70% train, 15% validation, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train an SVM classifier with progress bar
svm_clf = svm.SVC(kernel='rbf', gamma=0.1)
print("Training SVM model...")
for _ in range(1):  # We only have one step in SVM, but adding a progress bar for consistency
    svm_clf.fit(X_train, y_train)

# Validate the model with progress bar
print("Validating SVM model...")
val_predictions = []
for i in tqdm(range(len(X_val)), desc="Validating", unit="sample"):
    val_predictions.append(svm_clf.predict([X_val[i]])[0])

val_accuracy = accuracy_score(y_val, val_predictions)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Test the model with progress bar
print("Testing SVM model...")
test_predictions = []
for i in tqdm(range(len(X_test)), desc="Testing", unit="sample"):
    test_predictions.append(svm_clf.predict([X_test[i]])[0])

test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Example of predicting on a single image
def predict_single_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.flatten().numpy().reshape(1, -1)  # Flatten and reshape for SVM
    prediction = svm_clf.predict(image)
    return prediction

# Test on a new image
test_image_path = 'dosing_volume/train_new_liquid_wight_2410/balance/separate/num_0/1.jpg'
predicted_label = predict_single_image(test_image_path)
print(f"Predicted label for test image: {predicted_label}")
