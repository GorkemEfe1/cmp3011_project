# I am using the smallest Resnet(resnet18) model for simplicity, changing it is really easy

import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader


DATA_DIR = "./images"
EPOCHS = 20 
BATCH_SIZE = 64
LR = 1e-4 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

tf_train = transforms.Compose([  
    transforms.Resize((224,224)),
    # transforms.RandomHorizontalFlip(p=0.5), # New(Haven't tested) TEST This first
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
tf_val = transforms.Compose([  
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=tf_train)
val_ds   = datasets.ImageFolder(f"{DATA_DIR}/validation",   transform=tf_val) 

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True) 
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE) 
print(f"Classes: {train_ds.classes}")
print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")


model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
model = model.to(DEVICE) 


criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=LR) 
# Freeze all layers ATTEMPT FAILED
# train_params = []
# for name, layer in model.named_children():
#     if name in ['conv1', 'bn1', 'layer1', 'layer2']:
#         for param in layer.parameters():
#             param.requires_grad = False
#     if name not in ['conv1', 'bn1', 'layer1', 'layer2']:
#         for param in layer.parameters():
#             train_params.append(param)
# optimizer = torch.optim.Adam(train_params, lr=LR) 

for epoch in range(1, EPOCHS+1):
    model.train() 
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE) 
        optimizer.zero_grad() 
        loss = criterion(model(images), labels) 
        loss.backward() 
        optimizer.step() 
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(dim=1) 
            correct += preds.eq(labels).sum().item() 
            total   += labels.size(0) 
    print(f"Epoch {epoch}/{EPOCHS}  val_acc={100*correct/total:.1f}%") 

torch.save({"state_dict": model.state_dict(), "classes": train_ds.classes}, "fer_model.pth")
print("Saved → fer_model.pth")

# Also remember to compare it to non fine tuned version
# Clean the Pip Environment
# Publish to github