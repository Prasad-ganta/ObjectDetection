import torch
import torchvision
from torchvision.datasets import VOCDetection
from torchvision.transforms import functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
def collate_fn(batch):
    return tuple(zip(*batch))

transform = T.Compose([
    T.Resize((300, 300)),
    T.ToTensor()
])
train_dataset = VOCDetection(root='.', year='2007', image_set='train', download=True, transform=transform)
val_dataset = VOCDetection(root='.', year='2007', image_set='val', download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 
    'car', 'cat', 'chair', 'cow', 'dog', 'horse', 'motorbike', 'person', 
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
weights = SSD300_VGG16_Weights.DEFAULT
model = ssd300_vgg16(weights=weights)
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=0.0005)
num_epochs = 10
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        formatted_targets = []
        for t in targets:
            boxes = []
            labels = []
            anns = t['annotation']['object']
            if not isinstance(anns, list): 
                anns = [anns]
            for obj in anns:
                bbox = obj['bndbox']
                xmin = float(bbox['xmin'])
                ymin = float(bbox['ymin'])
                xmax = float(bbox['xmax'])
                ymax = float(bbox['ymax'])
                label = obj['name']
                if label in VOC_CLASSES:
                    labels.append(VOC_CLASSES.index(label))
                else:
                    labels.append(0)  
                boxes.append([xmin, ymin, xmax, ymax])
            formatted_targets.append({
                "boxes": torch.tensor(boxes).float().to(device),
                "labels": torch.tensor(labels).long().to(device)
            })

        try:
            loss_dict = model(images, formatted_targets)
            losses = sum(loss for loss in loss_dict.values())
        except:
            continue

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "ssd_pascalvoc_mobilenet.pth")
print("Model saved!")

model.eval()
with torch.no_grad():
    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for img_tensor, output in zip(images, outputs):
            img = img_tensor.permute(1, 2, 0).cpu().numpy()
            plt.imshow(img)
            for box in output['boxes']:
                x1, y1, x2, y2 = box
                plt.gca().add_patch(plt.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    fill=False, edgecolor='red', linewidth=2
                ))
            plt.axis("off")
            plt.show()
        break  
