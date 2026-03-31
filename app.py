import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

st.title("CIFAR-10 Image Classifier (CNN)")

# Classes (CIFAR-10)
classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

# Model class (same as notebook)
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(4*4*128,256),
            nn.ReLU(),
            nn.Linear(256,10),
        )

    def forward(self,x):
        x=self.conv_layers(x)
        x=x.view(x.size(0),-1)
        x=self.fc_layers(x)
        return x

# Load model
model = CNN()
model.load_state_dict(torch.load("cnn_model.pt", map_location=torch.device('cpu')))
model.eval()

# Image transform (same as training)
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# Upload image
file = st.file_uploader("Upload Image", type=["jpg","png"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    output = model(img)
    _, pred = torch.max(output,1)

    st.success(f"Prediction: {classes[pred.item()]}")