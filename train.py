import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
import copy
import time
import argparse
from sys import argv
import os

def get_data_dir(_data_dir):
    data_dir = _data_dir #'assets/flower_data'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    return train_dir, valid_dir, test_dir

def get_data(_data_dir, batch_size_ = 128):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    b_size = batch_size_
    train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                           #transforms.Resize(255),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ColorJitter(
                                                                    brightness=0.4,
                                                                    contrast=0.4,
                                                                    saturation=0.4,
                                                                 ),
                                           transforms.ToTensor(),
                                           normalize
                                          ])

    valid_transform = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          normalize])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          normalize])

    data_transforms = {
        'train': train_transforms,
        'valid': valid_transform,
        'test' : test_transforms
    }

    train_dir, valid_dir, test_dir = get_data_dir(_data_dir)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    image_datasets = {
        'train': train_dataset,
        'valid': valid_dataset,
        'test' : test_dataset
    }

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=b_size)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=b_size)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=b_size)

    dataloaders_dict = {
        'train': trainloader,
        'valid': validloader,
        'test' : testloader
    }

    return dataloaders_dict, image_datasets

def save_model_checkpoint(model_state_dict, class_to_idx, architecure, checkpoint_file_path, isSave2File = False):
    checkpoint = {
        'state_dict': model_state_dict, #model.state_dict(),
        'class_to_idx': class_to_idx, #image_datasets['train'].class_to_idx,
        'arch': architecure # e.g. 'vgg19'
    }
    if (isSave2File):
        torch.save(checkpoint, checkpoint_file_path)
    return checkpoint

def get_chkpoint_file_path(save_dir):
    file_path = save_dir + '/checkpoint.pth'
    return file_path

def train_model(model, device, dataloaders, criterion, optimizer, save_dir, architecture, num_epochs=10):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                print("best_acc = " + str(best_acc))
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'valid':
                val_acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    checkpoint_file_path = get_chkpoint_file_path(save_dir)
    checkpoint = save_model_checkpoint(best_model_wts, image_datasets['train'].class_to_idx,
                          architecture, checkpoint_file_path, True)

    return model, val_acc_history, checkpoint

def getClassifier(hidden_units = 25088):
    classifier = nn.Sequential( nn.Linear(hidden_units, 4096, bias=True),
                                nn.ReLU(True),
                                nn.Dropout(0.5),
                                nn.Linear(4096, 4096, bias=True),
                                nn.ReLU(True),
                                nn.Dropout(0.5),
                                nn.Linear(4096, 102, bias=True),
                                nn.LogSoftmax(dim=1))

    return classifier

def get_model(architecture):
    model = None
    if (architecture == "vgg19"):
        model = models.vgg19(pretrained=True)
    elif (architecture == "vgg19_bn"):
        model = models.vgg19_bn(pretrained=True)
    else:
        print("Only vgg19 or vgg19_bn are supported!")
    return model

def configure_model_training(isUseGPU, architecture, epochs = 50, hidden_units = 25088, learning_rate = 0.00001):
    device = torch.device("cuda" if torch.cuda.is_available() and isUseGPU else "cpu")

    model = get_model(architecture)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = getClassifier(hidden_units)

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate, betas=(0.95, 0.999), weight_decay=1e-5)

    model.to(device)
    num_epochs = epochs

    return model, device, criterion, optimizer, num_epochs

if __name__ == '__main__':
    epochs = 50
    hidden_units = 25088
    useGPU = True
    checkpoint_dir = "."
    data_dir = "flowers"
    save_dir = "."
    learning_rate = 0.00001
    architecture = "vgg19"

    parser = argparse.ArgumentParser(description="Flowers classifier training module")

    parser.add_argument('--gpu', action="store_true", default=False)
    parser.add_argument('--epochs', action="store", dest="epochs", type=int)
    parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int)
    parser.add_argument('--learning_rate', action="store", dest="learning_rate", type=float)
    parser.add_argument('--save_dir', action="store", dest="save_dir")

    if (argv[1] != None):
        data_dir = argv[1]
    args = parser.parse_args(argv[2:])

    if (args.hidden_units != None):
        hidden_units = args.hidden_units
    if (args.epochs != None):
        epochs = args.epochs
    if (args.gpu != None):
        useGPU = args.gpu
    if (args.save_dir != None):
        save_dir = args.save_dir
    if (args.learning_rate != None):
        learning_rate = args.learning_rate

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataloaders_dict, image_datasets = get_data(data_dir)
    model, device, criterion, optimizer, num_epochs = configure_model_training(useGPU, architecture, epochs, hidden_units, learning_rate)
    model_ft, hist, checkpoint = train_model(model, device, dataloaders_dict, criterion, optimizer, save_dir, architecture, num_epochs)
