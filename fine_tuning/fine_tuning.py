# I am using the smallest Resnet(resnet18) model for simplicity, changing it is really easy

import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


DATA_DIR = "./images"
EPOCHS = 20 
BATCH_SIZE = 64
LR = 1e-4 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

tf_train = transforms.Compose([  
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
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


for epoch in range(1, EPOCHS+1):
    model.train() 
    train_correct = 0
    train_total = 0
    prediction = []
    truth = []

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE) 
        optimizer.zero_grad() 
        outputs = model(images)
        loss = criterion(outputs, labels) 
        loss.backward() 
        optimizer.step() 
        train_preds = outputs.argmax(dim=1) 
        train_correct += train_preds.eq(labels).sum().item() 
        train_total   += labels.size(0) 
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(dim=1) 
            correct += preds.eq(labels).sum().item() 
            total   += labels.size(0) 
            prediction.extend(preds.tolist())
            truth.extend(labels.tolist())
    print(f"Epoch {epoch}/{EPOCHS}  val_acc={100*correct/total:.1f}% train_acc={100*train_correct/train_total:.1f}") 

a = confusion_matrix(truth, prediction)
disp = ConfusionMatrixDisplay(a)
disp.plot()
plt.show()

torch.save({"state_dict": model.state_dict(), "classes": train_ds.classes}, "fer_model.pth")
print("Saved → fer_model.pth")

# Also remember to compare it to non fine tuned version
# Clean the Pip Environment
# Publish to github