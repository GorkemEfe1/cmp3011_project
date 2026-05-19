# Using Resnet because it offers better quality than mobile net and my gpu can handle it
# We have to Resize and normalize because ImageNet, the dataset that ResNet has been trained on
# had spesific mean RGB values and image size, if we do these two steps our dataset looks somewhat 
# like what resnet has come to expect
# To Get Resize or Normalize Values for models under tprchvision use:
# from torchvision.models import ResNet50_Weights
# print(ResNet50_Weights.IMAGENET1K_V2.transforms())

# I am using the smallest Resnet(resnet18) model for simplicity, changing it is really easy

import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

# 1 Constansts for readability
DATA_DIR = "./images" # One directory up
EPOCHS = 20 # between 10-30 
BATCH_SIZE = 32 # between 16-64 actually the higher this is the faster but less accurate the training will be
LR = 1e-4 # means 0.0001 / between 1e-3(faster) - 1e-5(slower) 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
# 2 Setting up Data
# Create transforms
tf_train = transforms.Compose([  # the transforms to apply to training images
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
tf_val = transforms.Compose([  # the transforms to apply to training images
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
# Apply transforms
train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=tf_train) # apply transformation on train data set, automatically creates labels from the folder names, pretty much builds a dataset to use
val_ds   = datasets.ImageFolder(f"{DATA_DIR}/validation",   transform=tf_val) #apply transform on validation data set
# Data loading so that I don't load the entire dataset into my ram and gpu memory at once.
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True) #shuffle is added so the model can see different expression right after another
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE) 
print(f"Classes: {train_ds.classes}") # f string to include variables with the {} sign inside of the string
print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

# 3 Setting up the model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) # 18 is layer number, can also be 34-50-101-152, additionally the expression inside means the weights gathered from training on this dataset, which in our case is imagenetv1 but for 50 and above we cna use imagenetv2
model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes)) # Changing the fully connected layer output to 7 from resnet18's 512, while the input layer stays as default nn.Linear(input layers, output layers)
model = model.to(DEVICE) 

# 4 Training
criterion = nn.CrossEntropyLoss() # Loss function, apperantly conventionally called the criterion, this can be other loss calculation functions
optimizer = torch.optim.Adam(model.parameters(), lr=LR) # Where actual learning happens, Adam optimizer can be changed for another optimizer

for epoch in range(1, EPOCHS+1): # can be range(Epochs) but since normal people start counting from 1 we add 1 to both
    model.train() # Set model to training mode, all nn.module based modules have train() and eval() modes
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE) # Putting the images and labels in to the gpu memory
        optimizer.zero_grad() # Reset Gradients. Gradients are what tells you where to shift the weigths to, weigths are what you learn and get improved
        loss = criterion(model(images), labels) # Comparing models predictions to the actual labels model(images) is where the model runs through the batch
        loss.backward() # Gradient is calculated
        optimizer.step() # Gradient is applied to the weights
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(dim=1) # we are running the model on the image batch here. One predicted class(label) per image, this is then compared to the true data(images). ARGMAX picks the most likely class because it has the highest prediction number and gives that to us among the 7 classes, if it was dim 0 then it would give us the biggest out of the 32 values.
            correct += preds.eq(labels).sum().item() # preds.eq(labels) gives true or false, then sum counts the true predictions, item makes the tensor into a number
            total   += labels.size(0) # how many labels there are... this is equal to image size
    print(f"Epoch {epoch}/{EPOCHS}  val_acc={100*correct/total:.1f}%") # .1f limits  it to something like 90.5 rather than 90.5000001 or smt

torch.save({"state_dict": model.state_dict(), "classes": train_ds.classes}, "fer_model.pth") # Save weights and class names, class names can be stuff like "happy", "sad"...
print("Saved → fer_model.pth")

# Also remember to compare it to non fine tuned version
# Clean the Pip Environment
# Publish to github