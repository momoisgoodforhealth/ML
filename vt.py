import torch 
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np 
from PIL import Image
import pandas as pd
import os 
import wandb
from tqdm import tqdm
from functools import partial


train_dir = "final_dataset/train"
test_dir = "final_dataset/train"

#dataset = load_dataset("huggingface/cats-image")

from PIL import Image
image = Image.open('image/cat.jpg')
convert_tensor = transforms.ToTensor()

transform = transforms.Compose([transforms.PILToTensor()])
tensor = transform(image)
#convert_tensor(img)
#model_name = 'google/vit-base-patch16-224'
#model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

model.eval()    

encodings = feature_extractor(images=tensor, return_tensors="pt")

inputs = feature_extractor(tensor, return_tensors="pt")

pixel_values = encodings["pixel_values"]
outputs = model(pixel_values) 


logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
answer = model.config.id2label[predicted_class_idx]

print(answer)


#with torch.no_grad():
    #logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
#predicted_label = logits.argmax(-1).item()
#print(model.config.id2label[predicted_label])