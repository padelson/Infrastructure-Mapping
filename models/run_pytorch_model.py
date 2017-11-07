# from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from sklearn.metrics import f1_score

filetail = ".0.npy"
continuous = True
lr = 1e-4 # was 0.01 for binary
momentum = 0.9 # was 0.4 for binary

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
                dataloders = dataloaders_train
                current_dataset = dataset_train
            else:
                model.train(False)  # Set model to evaluate mode
                dataloders = dataloaders_test
                current_dataset = dataset_test

            running_loss = 0.0
            running_corrects = 0.0
            running_tp = 0.0

            # Iterate over data.
            for data in dataloders:
                # get the inputs
                inputs = data['image']
                labels = data['labels'].type(torch.FloatTensor)

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                if not continuous:
                    running_corrects += torch.sum(preds == labels.data)
                # running_tp += torch.sum(torch.eq((preds == labels.data), labels.data))

                # print (preds == labels.data)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects / dataset_size

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

class AddisDataset(Dataset):
    """Addis dataset."""
    def __init__(self, from_index, to_index, csv_file, root_dir, column, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory with all the numpy files.
            column (string): Variable to predict
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert (to_index < 3591 and from_index >= 0)
        self.data = pd.read_csv(csv_file)[column][from_index:to_index] # TODO: lol indexing is jank rn will change
        self.root_dir = root_dir
        self.transform = transform
        self.from_index = from_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 's1_median_addis_multiband_224x224_%d.npy' % (self.from_index+idx))
        image = np.load(img_name)[:, :, :3]
        labels = self.data[self.from_index+idx]
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'labels': labels}

        return sample

####### Initialize Data

data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

num_examples = 100
train_test_split = 0.9
split_point = int(num_examples*train_test_split)

data_dir = '../addis_s1_center_cropped'
dataset_train = AddisDataset(0, split_point, csv_file='../Addis_data_processed.csv',
                                    root_dir=data_dir,
                                    column='distance_piazza',
                                    transform=data_transforms)
dataset_test = AddisDataset(split_point, num_examples, csv_file='../Addis_data_processed.csv',
                                    root_dir=data_dir,
                                    column='distance_piazza',
                                    transform=data_transforms)

dataloaders_train = DataLoader(dataset_train, batch_size=10, shuffle=True, num_workers=4)
dataloaders_test = DataLoader(dataset_test, batch_size=10, shuffle=False, num_workers=4)
dataset_size = len(dataset_train)

use_gpu = torch.cuda.is_available()

######## Train Model

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 1)

if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()
if continuous:
    criterion = nn.MSELoss(size_average=True)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=momentum)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=20)

# for data in dataloders:
#     # get the inputs
#     inputs = data['image']
#     labels = data['labels'].type(torch.LongTensor)

#     # wrap them in Variable
#     if use_gpu:
#         inputs = Variable(inputs.cuda())
#         labels = Variable(labels.cuda())
#     else:
#         inputs, labels = Variable(inputs), Variable(labels)

#     # forward
#     outputs = model_ft(inputs)
#     _, preds = torch.max(outputs.data, 1)
