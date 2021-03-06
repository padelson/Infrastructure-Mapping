# from __future__ import print_function, division
import sys
sys.path.append("..")
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
from utils import addis as util
from sklearn.metrics import f1_score

satellite = 'l8'
filetail = ".0.npy"
continuous = False 
lr = 1e-4
momentum = 0.9
len_dataset = 3591
num_examples = 1000
train_test_split = 0.9
last_many_f1 = 3
batch_size = 64
num_workers = 4
num_epochs = 10

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    # all_results = open(satellite + '_results.csv', 'w')
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_train_acc = 0.0
    best_f1 = 0.0
    best_train_f1 = 0.0
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
	    dataset_size = len(current_dataset)

            running_loss = 0.0
            running_corrects = 0.0
            running_preds = np.array([])

            # Iterate over data.
            for data in dataloders:
                # get the inputs
                inputs = data['image']
                if continuous: 
                    labels = data['labels'].type(torch.FloatTensor)
                else:
                    labels = data['labels'].type(torch.LongTensor)

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
                    if not continuous and epoch >= num_epochs - last_many_f1: 
                        running_preds = np.hstack((running_preds, preds.cpu().numpy()))
                # running_tp += torch.sum(torch.eq((preds == labels.data), labels.data))

                # print (preds == labels.data)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects / dataset_size
            epoch_f1 = 0.0
            if not continuous and epoch >= num_epochs - last_many_f1:
                epoch_f1 = f1_score(current_dataset.data, running_preds)
		print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_f1))
	    	# all_results.write(','.join([str(epoch), phase, str(epoch_loss), str(epoch_acc), str(epoch_f1)]) + '\n')
	    # else:
		# print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    # phase, epoch_loss, epoch_acc)
                # all_results.write(','.join([str(epoch), phase, str(epoch_loss), str(epoch_acc)]) + '\n')

            if phase == 'train' and epoch_f1 > best_train_f1:
                best_train_acc = epoch_acc
		best_train_f1 = epoch_f1
            # deep copy the model
            if phase == 'val' and epoch_f1 > best_f1:
		# print data['id'], preds, labels
                best_acc = epoch_acc
		best_f1 = epoch_f1
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_train_f1, best_f1, best_train_acc, best_acc

class AddisDataset(Dataset):
    """Addis dataset."""
    def __init__(self, indices, csv_file, root_dir, column, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory with all the numpy files.
            column (string): Variable to predict
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file)[column][indices].values
        self.root_dir = root_dir
        self.transform = transform
        self.indices = indices

        if not continuous:
            self.balance = float(np.sum(self.data)) / float(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, satellite + '_median_addis_multiband_224x224_%d.npy' % (self.indices[idx]))
        image = np.load(img_name)[:, :, :3][:, :, ::-1].copy()
        labels = self.data[idx]
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'labels': labels, 'id': indices[idx]}

        return sample

####### Initialize Data

data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

indices = np.arange(len_dataset)
np.random.shuffle(indices)
split_point = int(num_examples*train_test_split)
train_indices = indices[:split_point]
test_indices = indices[split_point:num_examples]

data_dir = '../addis_' + satellite + '_center_cropped'
all_results = open(satellite + '_results_binary.csv', 'w')
for col in util.binary_features:
	dataset_train = AddisDataset(train_indices, csv_file='../Addis_data_processed.csv',
					    root_dir=data_dir,
					    column=col,
					    transform=data_transforms)
	dataset_test = AddisDataset(test_indices, csv_file='../Addis_data_processed.csv',
					    root_dir=data_dir,
					    column=col,
					    transform=data_transforms)
	print "Balances: train: %f, test: %f" % (dataset_train.balance, dataset_test.balance)

	dataloaders_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
	dataloaders_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	dataset_size = len(dataset_train)

	use_gpu = torch.cuda.is_available()

	######## Train Model
	# torch.set_default_tensor_type('torch.cuda.FloatTensor')
	model_ft = models.resnet18(pretrained=True)
	# uncomment for fixed model
	# for param in model_ft.parameters():
    	#	param.requires_grad = False
	num_ftrs = model_ft.fc.in_features
	if continuous:
	    model_ft.fc = nn.Linear(num_ftrs, 1)
	else:
	    model_ft.fc = nn.Linear(num_ftrs, 2)

	if use_gpu:
	    model_ft = model_ft.cuda()

	if not continuous:
	    criterion = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor([dataset_train.balance, 1-dataset_train.balance]))
	if continuous:
	    criterion = nn.MSELoss(size_average=True)

	# Observe that all parameters are being optimized
	optimizer_ft = optim.Adam(model_ft.parameters(), lr=lr)
	# Decay LR by a factor of 0.1 every 7 epochs
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=4, gamma=0.1)

	model_ft, train_f1, f1, train_acc, acc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
			       num_epochs=num_epochs)
	all_results.write(','.join([col, str(dataset_train.balance), str(dataset_test.balance), str(train_f1), str(f1), str(train_acc), str(acc)]) + '\n')
