import argparse
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


class ImageProcessing():
    def __init__(self, dataset, dataset_dir) -> None:
        super().__init__()

        self.dataset = dataset
        self.dataset_dir = dataset_dir

    def buildFolders(self):
        with open(f"{self.dataset_dir}.csv", 'r') as file:
            next(file)
            lines = file.readlines()

        classifiers = []

        for line in lines:
            data = line.split(',')
            number = data[0].rstrip()
            name = data[2].rstrip()
            classifiers.append((number, name))

        output_dir = f"{self.dataset_dir}_processed"
        os.makedirs(output_dir, exist_ok=True)

        for number, name in tqdm(classifiers):
            class_dir = os.path.join(output_dir, name)
            os.makedirs(class_dir, exist_ok=True)

            for filename in os.listdir(self.dataset_dir):
                if filename == f"{number}.jpg":
                    src_path = os.path.join(self.dataset_dir, filename)
                    dst_path = os.path.join(class_dir, filename)
                    shutil.copy(src_path, dst_path)

    def cropImages(self, model):
        for root, dirs, files in os.walk(self.dataset_dir):
            rel_path = os.path.relpath(root, self.dataset_dir)
            for filename in tqdm(files, desc=f"Processing: {rel_path}"):
                if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                    rel_path = os.path.relpath(root, self.dataset_dir)
                    output_crop = os.path.join(root_dir, f"{self.dataset}_cropped")
                    output_dir = os.path.join(output_crop, rel_path)
                    output_path = os.path.join(output_dir, filename)
                    if(os.path.exists(output_path)):
                        continue
                    os.makedirs(output_dir, exist_ok=True)

                    # load image
                    img_path = os.path.join(root, filename)
                    img = Image.open(img_path).convert('RGB')

                    # getting face
                    model(img, save_path=output_path)

                    # resizing
                    if (os.path.exists(output_path)):
                        img = Image.open(output_path).resize((img_size, img_size))
                        img.save(output_path)

class Model():
    def __init__(self, model, optimizer, scheduler, trans, device) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trans = trans
        self.device = device
    
    def train(self, crop_images):
        if (crop_images):
            ip = ImageProcessing("train", data_dir)
            ip.buildFolders()

            mtcnn = MTCNN(
                image_size=img_size,
                select_largest=False,
                post_process=True,
                margin=32,
                min_face_size=20,
                device=device
            )

            ip.cropImages(mtcnn)
            del mtcnn

        dataset = datasets.ImageFolder(data_dir + '_cropped', transform=self.trans)
        img_inds = np.arange(len(dataset))
        np.random.shuffle(img_inds)
        train_inds = img_inds[:int(0.8 * len(img_inds))]
        val_inds = img_inds[int(0.8 * len(img_inds)):]

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(train_inds)
        )
        val_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(val_inds)
        )

        loss_fn = torch.nn.CrossEntropyLoss()
        metrics = {
            'acc': training.accuracy
        }

        writer = SummaryWriter()
        writer.iteration, writer.interval = 0, 10

        print('\n\nInitial')
        print('-' * 56)
        self.model.eval()
        training.pass_epoch(
            self.model, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=self.device,
            writer=writer
        )

        epochs = 8

        for epoch in range(epochs):
            print('\nEpoch {}/{}'.format(epoch + 1, epochs))
            print('-' * 56)

            self.model.train()
            training.pass_epoch(
                self.model, loss_fn, train_loader, self.optimizer, self.scheduler,
                batch_metrics=metrics, show_running=True, device=self.device,
                writer=writer
            )

            self.model.eval()
            training.pass_epoch(
                self.model, loss_fn, val_loader,
                batch_metrics=metrics, show_running=True, device=self.device,
                writer=writer
            )

        writer.close()
        # torch.save(resnet.state_dict(), os.path.join(root_dir, "state_dict.pt"))
        return
    def test(self, crop_images, model_load="state_dict.pt"):
        self.model.load_state_dict(torch.load(model_load, map_location=torch.device('cpu')))

        for root, dirs, files in os.walk(data_dir):
            rel_path = os.path.relpath(root, data_dir)
            for filename in tqdm(files, desc=f"Processing: {rel_path}"):
                if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                    rel_path = os.path.relpath(root, data_dir)
                    output_crop = os.path.join(root_dir, "test_cropped")
                    output_dir = os.path.join(output_crop, rel_path)
                    output_path = os.path.join(output_dir, filename)
                    if(os.path.exists(output_path)):
                        continue
                    os.makedirs(output_dir, exist_ok=True)

                    # load image
                    img_path = os.path.join(root, filename)
                    img = Image.open(img_path).convert('RGB')

                    # getting face
                    mtcnn(img, save_path=output_path)

                    # resizing
                    if (os.path.exists(output_path)):
                        img = Image.open(output_path).resize((img_size, img_size))
                        # print(f"Resized: {output_path}")
                        img.save(output_path)
                    else:
                        img.resize((img_size, img_size))
                        img.save(output_path)
                        print(f"Failed to crop: {output_path}")

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

        return

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

root_dir = "/home/nick/School/ECE50024/CNN-Facial-Classifier"
data_dir = f"{root_dir}/train"
data_small_dir = f"{root_dir}/train_small"

batch_size = 32

img_size = 160

def main(mode, crop_images=False):
    dataset_dir = data_dir + '_processed'

    resnet = InceptionResnetV1(
            classify=True,
            pretrained='vggface2',
            num_classes=100,
            dropout_prob=0.75,
    ).to(device)

    optimizer = optim.Adam(resnet.parameters(), lr=0.0008)
    scheduler = MultiStepLR(optimizer)

    trans = transforms.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(5),
        fixed_image_standardization,
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    model = Model(resnet, optimizer, scheduler, trans, device)
    if mode == 'train':
        model.train(crop_images)
    elif mode == 'test':
        model.test(crop_images, "state_dict.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument('--mode', type=str,
                        help='either \'train\' or \'test\'')
    parser.add_argument('--crop_images', type=bool,
                        help='whether to crop images or not', default=False)
    args = parser.parse_args()

    if args.mode != 'train' and args.mode != 'test':
        parser.error("Invalid mode. Please use either 'train' or 'test'")
    else:
        print(f"Running in {args.mode} mode")
        main(args.mode, args.crop_images)

