from torchvision.datasets.vision import data
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
from PIL import Image
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import os
import shutil
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard.writer import SummaryWriter

root_dir = "/home/nick/School/ECE50024/CNN-Facial-Classifier"
data_dir = f"{root_dir}/train"

if __name__ == "__main__":
    resnet = InceptionResnetV1(
            classify=True,
            pretrained='vggface2',
            num_classes=100
    )
    resnet.load_state_dict(torch.load("state_dict.pt", map_location=torch.device('cpu')))
    resnet.eval()

    img = Image.open("train_cropped/Adam Sandler/1.jpg")
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = resnet(tensor)

    trans = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(data_dir + '_cropped', transform=trans)
    classes = dataset.class_to_idx

    _, predicted = torch.max(output, 1)
    temp = list(classes.items())
    c = [key for idx, key in enumerate(temp) if idx == predicted]
    print(predicted)
    print(c[0][0])
