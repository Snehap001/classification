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

torch.manual_seed(0)

transform = transforms.Compose([
    transforms.Resize((120, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization
])

# Define transformations with data augmentation for training dataset
train_transform = transforms.Compose([
    transforms.Resize((120, 100)),
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

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # Pooling reduces size by half

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 30 * 25, 128)  # Adjusted for output size after conv and pooling
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Forward pass through convolutional layers with ReLU, batch normalization, and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Automatically handle batch size

        # Fully connected layers with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Output layer
        x = self.fc2(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, model_pth):
    num_epochs = 25
    model.to(device)
    
    for epoch in range(num_epochs):
        print(f"Starting with epoch {epoch}")
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
        
        scheduler.step()  # Update learning rate
    
    # Save the trained model
    torch.save(model.state_dict(), model_pth)
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
dataset = datasets.ImageFolder(root=dataPath, transform=transform)


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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# Model, Optimizer, and Loss Function
model = BirdClassifier(num_classes=10)

criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler (optional)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


train_model(model,train_loader,val_loader,criterion,optimizer,scheduler,device,modelPath)
