import torch
import sys
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import torch.optim as optim
import torch.nn.functional as F
import os
from collections import Counter
from PIL import Image
from torchvision.transforms import InterpolationMode
torch.manual_seed(0)

transform = transforms.Compose([
    transforms.Resize((180, 150)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization
])

# Define transformations with data augmentation for training dataset
train_transform = transforms.Compose([
    transforms.Resize((180, 150)),
     transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization
])

class BirdClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(BirdClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 16 filters
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 32 filters
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 64 filters
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 128 filters
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # New layer with 256 filters
        self.bn5 = nn.BatchNorm2d(256)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # Reduces the feature map size by half

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 5 * 4, 512)  # Adjust input size as per the new conv layers
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)  # Output layer

        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Convolutional layers with ReLU, batch normalization, and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # New convolutional layer
        x = self.pool(F.relu(self.bn5(self.conv5(x))))  # New convolutional layer

        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Additional dropout
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Output layer

        return x
def preprocess_images(input_dir, output_dir, size=(180, 150), interpolation=InterpolationMode.BILINEAR):
    """
    Preprocess images with a specified interpolation method, resize them,
    and save them to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # Load the image
                img_path = os.path.join(root, file)
                img = Image.open(img_path)
                
                # Resize with the specified interpolation method
                resized_img = img.resize(size, interpolation)
                if resized_img.mode == 'RGBA':
                    resized_img = resized_img.convert('RGB')
                # Save to the output directory
                rel_path = os.path.relpath(root, input_dir)
                save_dir = os.path.join(output_dir, rel_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                resized_img.save(os.path.join(save_dir, file))
def train_model(model, train_loader, val_loader, criterion, optimizer, device, model_pth):
    num_epochs = 50
    model.to(device)
    best_val_accuracy=0
    for epoch in range(num_epochs):
        print(f"Starting with epoch {epoch+1}")
        # Training Phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation Phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Save the best model weights
            torch.save(model.state_dict(), model_pth)
            print("Validation accuracy improved, model saved.")
    print("Training complete.")

      
if __name__ == "__main__": 
    dataPath = sys.argv[1]
    trainStatus = sys.argv[2]
    modelPath = sys.argv[3] if len(sys.argv) > 3 else "default/model/path"
if trainStatus == "train":
    print("training")
else:
    print("infer")
print(f"Training: {trainStatus}")
print(f"path to dataset: {dataPath}")
print(f"path to model: {modelPath}")

# Check if CUDA (GPU support) is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading ...")
# Load the full dataset
input_data_path =dataPath
output_data_path = "preprocessed_dataset"

# Preprocess with different interpolation methods, e.g., BICUBIC for smoother resizing
preprocess_images(input_data_path, output_data_path, interpolation=Image.Resampling.LANCZOS)

# Load the preprocessed images
dataset = datasets.ImageFolder(root=output_data_path, transform=transform)
# dataset = datasets.ImageFolder(root=dataPath, transform=transform)


print("Extracting...")

# List to store the labels
labels = []

# Iterate through each subdirectory (class)
for class_index, class_name in enumerate(sorted(os.listdir(dataPath))):  # Sorting to maintain consistent order
    class_path = os.path.join(dataPath, class_name)
    
    if os.path.isdir(class_path):  # Check if it's a directory
        # Count the number of image files in the class directory
        num_images = len([f for f in os.listdir(class_path) 
                          if os.path.isfile(os.path.join(class_path, f)) 
                          and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])  # Filter by image extensions
        
        # Append the class index `num_images` times to the labels list
        labels.extend([class_index] * num_images)


print("Splitting ...")

# Initialize StratifiedShuffleSplit
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(range(len(labels)), labels))

# Create training and validation subsets
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

# Apply the transformations to the training dataset
train_dataset.dataset.transform = train_transform  # Update only the training dataset with augmentations


# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
# Model, Optimizer, and Loss Function
model = BirdClassifier(num_classes=10)

criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


train_model(model,train_loader,val_loader,criterion,optimizer,device,modelPath)
