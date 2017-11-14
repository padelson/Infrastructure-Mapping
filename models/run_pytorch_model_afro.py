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
# from .utils import addis as util
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

#################
import matplotlib.pyplot as plt



# satellite = 'l8'
# filetail = ".npy"
# continuous = False
# lr = 1e-4
# momentum = 0.9
# len_dataset = 3591

# data_dir = '../../data'
# data_dir = '../../data/afrobarometer/afro_224x224'
data_dir = '../../data/afrobarometer/afro_des_s1_center_cropped'
# data_dir = '/mnt/mounted_bucket/afro_l8_center_cropped'
# column = 'pit_latrine_depth_val2_when_bl_dw39_val1'


prefix_sat = 's1'
# num_examples = 100
# train_test_split = 0.9
continuous = False
lr = 1e-3 # was 0.01 for binary
momentum = 0.4 # was 0.4 for binary
last_many_f1 = 5
batch_size = 10
num_workers = 8
num_epochs = 2

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
        for phase in ['train', 'vali']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
                dataloders = dataloaders_train
                current_dataset = afDataset_train
            else:
                model.train(False)  # Set model to evaluate mode
                dataloders = dataloaders_test
                current_dataset = afDataset_test
            dataset_size = len(current_dataset)

            running_loss = 0.0
            running_corrects = 0.0
            running_preds = np.array([])

            # Iterate over data.
            for data in dataloders:
                # get the inputs
                inputs = data['image']
                if continuous:
                    labels = data['label'].type(torch.FloatTensor)
                else:
                    labels = data['label'].type(torch.LongTensor)

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
            else:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
            all_results.write(','.join([str(epoch), phase, str(epoch_loss), str(epoch_acc)]) + '\n')
            # deep copy the model
            if phase == 'train' and epoch_f1 > best_train_f1:
                best_train_acc = epoch_acc
                best_train_f1 = epoch_f1            

            if phase == 'vali' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_train_f1, best_f1, best_train_acc, best_acc

##############################
#
class AfroDatasetManager(object):
    def __init__(self, indices, csv_file, img_root_dir,
                 column, col_id="id", transform=None,
                 binary=True, satellite='l8'):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory with all the numpy files.
            column_namelist (list of strings): Variable to predict
            transform (callable, optional): Optional transform to be applied
                on a sample.

        """
        ## self column keyword
        self.column_name = column
        self.col_id_name = col_id
        ## processing the indices
        self.response_data = pd.read_csv(csv_file, index_col=False)[[column, col_id]] #.values  # TODO: lol indexing is jank rn will change
        self.root_dir = img_root_dir
        self.transform = transform
        self.prefix_sate = satellite
        self.img_ids = self.collect_img_ids()
        self.indices = np.array(list(set(self.response_data[col_id].values) & set(self.img_ids))) # intersect ids

    def collect_img_ids(self):
        img_ids_collection = []
        fnames = os.listdir(self.root_dir)
        for f in fnames:
            img_loc_id = f.strip(".npy").split("_")[-1]
            if img_loc_id is not None:
                img_ids_collection.append(int(img_loc_id))

        return img_ids_collection

    def split_train_test_by_Y(self, test_size=0.2):
        mat_y_wID = self.response_data[self.response_data[self.col_id_name].isin(self.indices) ]
        ids_ = mat_y_wID[self.col_id_name].values
        ylabels_ = mat_y_wID[self.column_name].astype(float).values
        id_train, id_test, ys_train, ys_test = train_test_split(ids_, ylabels_, stratify=ylabels_,
                                                                test_size=test_size, random_state=3)
        return id_train, id_test, ys_train, ys_test


##########################
class AfrobDataset(Dataset):
    """Addis dataset."""
    def __init__(self, indices, csv_file, root_dir, column,
                 prefix_sat='l8', continuous=False, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory with all the numpy files.
            column (string): Variable to predict
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file)[column][indices].values # TODO: lol indexing is jank rn will change
        self.root_dir = root_dir
        self.transform = transform
        self.indices = indices
        self.satellite = prefix_sat

        if not continuous:
            # it works in binary case
            self.balance = float(np.sum(self.data)) / float(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.satellite + '_median_afro_multiband_224x224_%d.npy' % (self.indices[idx]))
        try :
            if self.satellite == 'l8':
                image = np.load(img_name)[:, :, :3][:,:,::-1].copy()
            elif self.satellite == 's1':
                image = (np.uint8(np.load(img_name))).copy() #.astype(np.uint8).copy()

            labels = self.data[idx]
            if self.transform:
                image = self.transform(image)
                # print(image)
                # raise NotImplementedError("break!!!!")
            sample = {'image': image, 'label': labels, 'id': self.indices[idx]}

            return sample
        except Exception:
            pass 
        
        


####### Initialize Data

data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# data_dir = 'hymenoptera_data'
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x])
#                   for x in ['train', 'val']}
# dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
#                                              shuffle=True, num_workers=4)
#               for x in ['train', 'val']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# class_names = image_datasets['train'].classes
#
# use_gpu = torch.cuda.is_available()

# all_results = open(prefix_sat + '_results_binary.csv', 'w')



def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()
    # plt.pause(3.001)  # pause a bit so that plots are updated


# Get a batch of training data
#inputs, classes = next(iter(dataloders['train']))

# Make a grid from batch
#out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])



# for col in util.binary_features:
categories=["eaelectricity", "eapipedwater", "easewage", "earoad",
            'eacellphone',
	    'eapostoffice',
	    'easchool',
	    'eapolicestation',
 	    'eahealthclinic',
	    'eamarketstalls',
	    'eabank']

for j in range(11):  # FIXME will change it into feature name
    col = categories[j] #"eaelectricity"
    all_results = open(prefix_sat +'_'+ col + '_results_binary.csv', 'w')
    Af_dataManager = AfroDatasetManager(indices=None,
                                        csv_file="../Afrobarometer/process-data/Af_normed_response_mat_wID.csv",
                                        img_root_dir=data_dir, column=col,
                                        col_id="id",
                                        binary=True, satellite=prefix_sat)

    ids_train, ids_test, ys_train, ys_test = Af_dataManager.split_train_test_by_Y()

    afDataset_train = AfrobDataset((np.array(ids_train) ),
                                   csv_file="../Afrobarometer/process-data/Af_normed_response_mat_wID.csv",
                                   root_dir=data_dir, column=col,
                                   prefix_sat=prefix_sat, continuous=False, transform=data_transforms)

    afDataset_test = AfrobDataset((np.array(ids_test) ),
                                  csv_file="../Afrobarometer/process-data/Af_normed_response_mat_wID.csv",
                                  root_dir=data_dir, column=col,
                                  prefix_sat=prefix_sat, continuous=False, transform=data_transforms)


    dataloaders_train = DataLoader(afDataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloaders_test = DataLoader(afDataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataset_size = len(afDataset_train)

    # for i in range(2):
    #     sample = afDataset_train[i]
    #     imshow(sample['image'], title=sample['id'])
    #     print(sample['image'])


    # raise NotImplementedError("======================!")

    use_gpu = torch.cuda.is_available()

    ######## Train Model
    if use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

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
        if use_gpu:
            criterion = nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor([afDataset_train.balance, 1-afDataset_train.balance]))
        else:
            criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([afDataset_train.balance, 1-afDataset_train.balance]))
    if continuous:
        criterion = nn.MSELoss(size_average=True)

    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=momentum)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=lr)

    # Decay LR by a factor of 0.1 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

    model_ft, train_f1, f1, train_acc, acc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
			       num_epochs=num_epochs)
    #all_results.write(col + ',' + str(train) + ',' + str(val) + '\n') 
    all_results.write(','.join([col, str(afDataset_train.balance), str(afDataset_test.balance), str(train_f1), str(f1), str(train_acc), str(acc)]) + '\n')


