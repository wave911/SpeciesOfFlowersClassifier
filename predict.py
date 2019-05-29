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

def getClassifier(input_units = 25088, hidden_units = 4096):
    classifier = nn.Sequential( nn.Linear(input_units, hidden_units, bias=True),
                                nn.ReLU(True),
                                nn.Dropout(0.5),
                                nn.Linear(hidden_units, hidden_units, bias=True),
                                nn.ReLU(True),
                                nn.Dropout(0.5),
                                nn.Linear(hidden_units, 102, bias=True),
                                nn.LogSoftmax(dim=1))

    return classifier

def load_model_checkpoint(checkpoint_file_path):
    checkpoint = torch.load(checkpoint_file_path)
    model = None
    architecture = checkpoint['arch']

    if (architecture == "vgg19"):
        model = models.vgg19(pretrained=True)
    elif (architecture == "densenet121"):
        model = models.densenet121(pretrained=True)
    elif (architecture == "alexnet"):
        model = models.alexnet(pretrained=True)
    elif (architecture == "googlenet"):
        model = models.alexnet(pretrained=True)
    else:
        print("Only vgg19, densenet121, alexnet or googlenet are supported!")
        return
    for param in model.parameters():
        param.requires_grad = False

    hidden_units = checkpoint["hidden_units"]
    input_units = checkpoint["input_units"]
    model.classifier = getClassifier(input_units, hidden_units)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model

if __name__ == '__main__':
    useGPU = True
    top_k = 5
    hidden_units = 25088
    image_path = "flowers/test/2/image_05100.jpg"
    checkpoint_file_path = 'classifier.pth'
    category_names_file_path = None
    parser = argparse.ArgumentParser(description="Flowers classifier predict module")

    parser.add_argument('--gpu', action="store_true", default=False)
    parser.add_argument('--top_k', action="store", dest="top_k", type=int)
    parser.add_argument('--category_names', action="store", dest="category_names")

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

    device = torch.device("cuda" if torch.cuda.is_available() and useGPU else "cpu")
    model = load_model_checkpoint(checkpoint_file_path)
    model.to(device)

    flowers_by_name = None

    probs, classes = predict(image_path, model, top_k)
    if (category_names_file_path != None):
        cat_to_name = get_cat_to_json(category_names_file_path)
        flowers_by_name = [cat_to_name[x] for x in classes]

    print(probs)
    print(classes)
    if (flowers_by_name != None):
        print(flowers_by_name)
