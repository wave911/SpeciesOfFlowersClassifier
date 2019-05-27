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
import json
import numpy as np

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    from PIL import Image
    img = Image.open(image)

    if img.size[0] > img.size[1]:
        img.thumbnail((99999, 255))
    else:
        img.thumbnail((255, 99999))

    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,
                      top_margin))

    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean)/std

    img = img.transpose((2, 0, 1))
    return img

def get_cat_to_json(file_path):
    with open(file_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image_tensor = torch.from_numpy(image).float().to(device)
    model_input = image_tensor.unsqueeze(0)

    probs = torch.exp(model.forward(model_input))
    top_probs, top_labs = probs.topk(topk)
    top_probs = top_probs.cpu().detach().numpy().tolist()[0]
    top_labs = top_labs.cpu().detach().numpy().tolist()[0]
    idx_to_class = {val: key for key, val in model.class_to_idx.items() }
    top_labels = [idx_to_class[lab] for lab in top_labs]

    return top_probs, top_labels

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

def load_model_checkpoint(architecure, checkpoint_file_path, hidden_units = 25088):
    checkpoint = torch.load(checkpoint_file_path)
    model = None
    if (checkpoint['arch'] == architecure):
        if (architecure == 'vgg19'):
            model = models.vgg19(pretrained=True)
        elif (architecure == 'vgg19_bn'):
            model = models.vgg19_bn(pretrained=True)
        else:
            print("Currently vgg19 or vgg19_bn are supported!")
            return
        for param in model.parameters():
            param.requires_grad = False

        model.classifier = getClassifier(hidden_units)
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])

    return model

if __name__ == '__main__':
    useGPU = True
    top_k = 5
    hidden_units = 25088
    image_path = "flowers/test/2/image_05100.jpg"
    checkpoint_file_path = 'classifier.pth'
    category_names_file_path = "cat_to_name.json"
    parser = argparse.ArgumentParser(description="Flowers classifier predict module")

    parser.add_argument('--gpu', action="store_true", default=False)
    parser.add_argument('--top_k', action="store", dest="top_k", type=int)
    parser.add_argument('--category_names', action="store", dest="category_names")
    parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int)

    if (len(argv) <= 1):
        print("image_path & checkpoint_file_path are not provided! Exit")
        exit(-1)

    if (argv[1] != None):
        image_path = argv[1]
    if (argv[2] != None):
        checkpoint_file_path = argv[2]

    args = parser.parse_args(argv[3:])
    if (args.top_k != None):
        top_k = args.top_k
    if (args.gpu != None):
        useGPU = args.gpu
    if (args.category_names != None):
        category_names_file_path = args.category_names
    if (args.hidden_units != None):
        hidden_units = args.hidden_units

    device = torch.device("cuda" if torch.cuda.is_available() and useGPU else "cpu")
    model = load_model_checkpoint('vgg19', checkpoint_file_path, hidden_units)
    model.to(device)

    cat_to_name = get_cat_to_json(category_names_file_path)
    probs, classes = predict(image_path, model, top_k)

    flowers_by_name = [cat_to_name[x] for x in classes]
    print(probs)
    print(classes)
    print(flowers_by_name)
