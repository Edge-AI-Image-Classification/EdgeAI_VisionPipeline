import torch
from torch.utils.data import DataLoader, random_split, Subset
#Importing caltech101 dataset
from torchvision.datasets import Caltech101
from torchvision import transforms

# Transformations 
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Loading new dataset
dataset = Caltech101(root='data', download=True, transform=transform)

# #making a smaller randomized subset for quick testing. Comment out if not needed
# testset_size = 100
# indices = torch.randperm(len(dataset))[:testset_size]
# testset = Subset(dataset, indices)
# print(f"Using only {testset_size} images for a quick test run.")

# #training and validation on smaller subset. Comment out if not needed. 
# train_size = int(0.8 * len(testset))
# val_size   = len(testset) - train_size
# train_dataset, val_dataset = random_split(testset, [train_size, val_size])

# Splitting dataset in to an 80/20 split. 80 for training and 20 for validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print("Number of training images:", len(train_dataset))
print("Number of validation images:", len(val_dataset))


# Dataloaders for training and validation datasets
trainDataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valDataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
