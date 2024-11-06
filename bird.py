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
import csv
torch.manual_seed(0)

# Define transformations with data augmentation for the training dataset
train_transform = transforms.Compose([
    transforms.Resize((180, 150),interpolation=Image.Resampling.LANCZOS),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization
])

# Define transformations for the validation/test dataset
transform = transforms.Compose([
    transforms.Resize((180, 150),interpolation=Image.Resampling.LANCZOS),
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
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 256 filters
        self.bn5 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)  # Pooling layer
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 5 * 4, 512)  # Adjust input size as per the conv layers
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)  # Output layer
        self.dropout = nn.Dropout(0.3)  # Dropout for regularization

    def forward(self, x):
        # Convolutional layers with ReLU, batch normalization, and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Output layer
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, device, model_pth):
    num_epochs = 60
    model.to(device)
    best_val_accuracy = 0
    for epoch in range(num_epochs):
        print(f"Starting with epoch {epoch+1}")
        model.train()
        running_loss = 0.0
        validation_loss=0.0
        train_total=0.0
        train_correct=0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_pth)
            print("Validation accuracy improved, model saved.")
    print("Training complete.")


def test_model(model, test_loader, device, output_csv='bird.csv'):
    model.eval()  
    results = []
    with torch.no_grad():  
        for images, _ in test_loader:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            # Get predictions by taking the class with the highest score
            _, predicted = torch.max(outputs, 1)
            results.extend(predicted.cpu().numpy())

    # Write only predicted labels to a CSV file
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Predicted_Label'])
        for label in results:
            writer.writerow([label])

      
if __name__ == "__main__": 
    dataPath = sys.argv[1]
    trainStatus = sys.argv[2]
    modelPath = sys.argv[3] if len(sys.argv) > 3 else "default/model/path"
    if trainStatus == "train":
        print("training")
        # Check if CUDA (GPU support) is available, otherwise use CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading ...")
        # Load the full dataset
        # Preprocess with different interpolation methods, e.g., BICUBIC for smoother resizing

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
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False,num_workers=4)
        # Model, Optimizer, and Loss Function
        model = BirdClassifier(num_classes=10)

        class_counts = Counter([label for _, label in train_dataset])  # `train_dataset` from your DataLoader

        # Calculate weights as the inverse of the class frequency
        total_samples = sum(class_counts.values())
        class_weights = {class_idx: total_samples / count for class_idx, count in class_counts.items()}

        # Create a tensor of class weights
        weight_tensor = torch.tensor([class_weights[i] for i in range(len(class_counts))], dtype=torch.float).to(device)

        # Pass the weights to CrossEntropyLoss
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)

        train_model(model,train_loader,val_loader,criterion,optimizer,device,modelPath)


    else:
        print("infer")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BirdClassifier(num_classes=10) 
        model.to(device)
        model.load_state_dict(torch.load(modelPath, weights_only=True))
        model.eval()    
        test_dataset=datasets.ImageFolder(root=dataPath, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        test_model(model,test_loader,device)
