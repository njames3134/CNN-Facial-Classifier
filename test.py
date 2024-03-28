import csv
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from PIL import Image
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import datasets, transforms
from torchvision.datasets.vision import data
from torchvision.transforms import v2
from tqdm import tqdm

from facenet_pytorch import (MTCNN, InceptionResnetV1,
                             fixed_image_standardization, training)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

root_dir = "/home/nick/School/ECE50024/CNN-Facial-Classifier"
data_dir = f"{root_dir}/train_cropped"
test_dir = f"{root_dir}/test_cropped"

img_size = 160

if __name__ == "__main__":
    resnet = InceptionResnetV1(
            classify=True,
            pretrained='vggface2',
            num_classes=100
    )
    resnet.load_state_dict(torch.load("state_dict.pt", map_location=torch.device('cpu')))
    resnet.eval()

    # mtcnn = MTCNN(
    #     image_size=img_size,
    #     select_largest=False,
    #     post_process=True,
    #     margin=32,
    #     min_face_size=20,
    #     device=device
    # )
    #
    # for root, dirs, files in os.walk(data_dir):
    #     rel_path = os.path.relpath(root, data_dir)
    #     for filename in tqdm(files, desc=f"Processing: {rel_path}"):
    #         if filename.endswith('.jpg') or filename.endswith('.jpeg'):
    #             rel_path = os.path.relpath(root, data_dir)
    #             output_crop = os.path.join(root_dir, "test_cropped")
    #             output_dir = os.path.join(output_crop, rel_path)
    #             output_path = os.path.join(output_dir, filename)
    #             if(os.path.exists(output_path)):
    #                 continue
    #             os.makedirs(output_dir, exist_ok=True)
    #
    #             # load image
    #             img_path = os.path.join(root, filename)
    #             img = Image.open(img_path).convert('RGB')
    #
    #             # getting face
    #             mtcnn(img, save_path=output_path)
    #
    #             # resizing
    #             if (os.path.exists(output_path)):
    #                 img = Image.open(output_path).resize((img_size, img_size))
    #                 # print(f"Resized: {output_path}")
    #                 img.save(output_path)
    #             else:
    #                 img.resize((img_size, img_size))
    #                 img.save(output_path)
    #                 print(f"Failed to crop: {output_path}")
    image_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg') or f.endswith('.jpeg')], key=lambda x: int(os.path.splitext(x)[0]))
    trans = transforms.Compose([
        v2.ToImage(),
        fixed_image_standardization,
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    name = 0
    with open('pred.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Id', 'Category'])

        for file in tqdm(image_files):
            if file.endswith('.jpg') or file.endswith('.jpeg'):
                img_path = os.path.join(test_dir, file)
                img = Image.open(img_path)
                tensor = trans(img).unsqueeze(0)
                with torch.no_grad():
                    output = resnet(tensor)

                dataset = datasets.ImageFolder(data_dir, transform=trans)
                classes = dataset.class_to_idx

                predicted = torch.argmax(output, 1)
                temp = list(classes.items())

                c = [key for idx, key in enumerate(temp) if idx == predicted][0]
                csvwriter.writerow([name, c[0]])
                name+=1
