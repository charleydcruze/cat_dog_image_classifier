#Cat or Dog Image Classifier: ResNet-18 Transfer Learning

#Fine-tunes a pre-trained ResNet-18 (ImageNet weights) to classify images as cat or dog. 
#Only layer4 and the final fully connected layer are trained-
# earlier layers are frozen.

import os #<---Library that allows for interaction with the operating system.
import torch #<---Core PyTorch machine learning library. Provides tensors and automatic differentiation (autograd), which-
# calculates gradients.

import torch.nn as nn #<---Imports PyTorch's neural network module which provides building blocks-
# such as nn.Linear for fully connected layers or nn.CrossEntropyLoss for loss functions.

import torch.optim as optim #<---Contains optimisers, these are the programs that use gradients to decide how to nudge the weights.

from torch.utils.data import DataLoader #<---Feeds data to the model in small batches as neural networks train on-
# batches rather than 1 image at a time (too slow) or the whole dataset at a time (too memory intensive).
from torchvision import datasets, models, transforms #<---Contains pre-trained AI models and image preprocessing tools.

#Image Preprocessing
transform = transforms.Compose([ 
    transforms.Resize((224, 224)), #<---Resizes any image uploaded to 224x224 (ResNet Dimensions)-
    # as this means it can understand and process the image.
    
    transforms.ToTensor(), #<---Converts pixel colour values into a mathematical tensor. 
    #A tensor is a mathematical object that generalises scalars, vectors and matrices into higher-
    # dimensions to describe multilinear relationships. For example a greyscale image is a 2D tensor-
    # (height, width), a coloured image is a 3D tensor(height, width, colour channels).
    
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #<---Normalises RGB-
    # channels using ImageNet dataset mean and standard deviation (Z = (X - mean) / std).
    #Expects a 3-channel tensor with values scaled [0.0, 1.0] from ToTensor(). 
    #Required so input image statistics match the pre-trained ResNet weights distribution.
    #The mean numbers are the average colour of red(.485), green(.456), blue(.406) from the 1.4 million-
    # images that were used to train the ResNet18 CNN. The std (standard deviation) represent the spread-
    # of colour intensity from the ImageNet training data. Mean and Std are unitless(dimensionless).
])

#Load dataset from project folder.
dataset_path = "dataset"
dataset = datasets.ImageFolder(root=dataset_path, transform = transform)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

print(
    f"Dataset loaded successfully Found {len(dataset)} images across"
    f"classes: {dataset.classes}"
)

#Loads ResNet-18 model and configures transfer learning.
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)


#Iterates through every weight parameter in the network and turns off gradient calculations.
#Convolutional neural networks learn hierarchical feature representations (edges, textures,-
# spots, and object shapes in the lower/middle layers)
#By setting gradient calculations to false, you freeze the pre-trained feature extractor layers so-
# PyTorch won't alter or degrade those learned visual weights.
#With "layer4" change only the early layers are frozen but layer4 and fc are kept active for Grad-CAM computation.
for name, param in model.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True #<---Enable gradient computation for Grad-CAM.

#Reads the number of input nodes feeding into ResNet-18s original final linear layer.
#in_features is a PyTorch attribute that stores the number of input nodes/features entering that layer.
num_features = model.fc.in_features

#This replaces the 1000-class output layer of the CNN into a 3-class output layer for Cats, Dogs and Neither/Other.
#.fc stands for Fully Connected layer. In PyTorch fc is the variable name asssigned to the final classification/output layer.
model.fc = nn.Linear(num_features, 3) #<---Index 0 = Cat, Index 1 = Dog, Index 2 = Other.

#Sets loss function and optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
)

#Fine-tuning loop (5 Epochs) Epoch = A complete pass through a dataset.
print("\n Starting fine-tuning on 1:1:1 balanced dataset.")
model.train()

for epoch in range(5):
    running_loss = 0.0
    correct = 0
    total = 0

for inputs, labels in train_loader:
    optimizer.zero_grad() #<---Clears previous gradients.
    outputs = model(inputs) #<---Forwards pass
    loss = criterion(outputs, labels)#<---Compute loss.
    loss.backward() #<---Backward pass
    optimizer.step() #<---Update weights

    running_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

epoch_loss = running_loss / len(train_loader)
epoch_acc = (correct / total) * 100
print(
    f"Epoch {epoch + 1}/5 - Loss: {epoch_loss:.4f} - Accuracy:"
    f" {epoch_acc:.2f}%"
)

#Save learned weights.
weights_save_path = "cat_dog_resnet18.pth"
torch.save(model.state_dict(),weights_save_path)
print(
    f"\n Fine-tuning complete. Saved learned weights to '{weights_save_path}'"
)

