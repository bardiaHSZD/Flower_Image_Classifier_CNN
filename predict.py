import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
from PIL import Image
import sys
import json

def cat_to_name_loader(cat_to_name_filename = 'cat_to_name.json'):
    with open(cat_to_name_filename, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name    

def load_network(checkpoint = 'checkpoint.pth'):
    checkpoint = torch.load('checkpoint.pth')

    model_rebuilt = getattr(models, checkpoint['arch'])(pretrained=True)

    # Freezing the pre-trained model parameters so we don't backprop gradients through them
    for param in model_rebuilt.parameters():
        param.requires_grad = False

    model_rebuilt.classifier = checkpoint['classifier']
    model_rebuilt.load_state_dict(checkpoint['state_dict'])
    model_rebuilt.class_to_idx = checkpoint['class_to_idx']
    
    return model_rebuilt

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    size_criteria = 256
    width, height = image.size
    shortest_side = min(width, height)
    img = image.resize([int((width/shortest_side)*size_criteria), int((height/shortest_side)*size_criteria)], Image.ANTIALIAS)  
    size = int((width/shortest_side)*size_criteria), int((height/shortest_side)*size_criteria)
    crop_size = 224, 224
    area = ((size[0]-crop_size[0])/2, (size[0]-crop_size[1])/2, (size[0]-crop_size[0])/2 + crop_size[0], (size[0]-crop_size[1])/2 + crop_size[1])
    img = img.crop(area)
    width, height = img.size
    np_img = (np.array(img))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = np_img.astype(float)
    np_img[:][:] = (np.true_divide(np_img[:][:],255.0)-mean)/std
    np_img_transpose = np_img.transpose((2, 0, 1))
    return np_img_transpose

def imshow(np_img_transpose, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    np_img_transpose = np_img_transpose.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img_transpose = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    np_img_transpose = np.clip(np_img_transpose, 0, 1)
    
    ax.imshow(np_img_transpose)
    
    return ax

def predict_image_class(image_path, model, topk=3, device='cuda'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''  
    img = Image.open(image_path)
    img_to_array = process_image(img)
    prediction_model = model
    top_k = topk
    prediction_model.to(device)
    img_to_array = np.expand_dims(img_to_array, axis=0)
    img_to_torch = torch.from_numpy(img_to_array)
    img_to_device = img_to_torch.to(device)
    img_to_device = img_to_device.float()
    outputs = prediction_model(img_to_device)
    probs, idxs = outputs.data.topk(top_k)
    
    index_to_class = {val: key for key, val in prediction_model.class_to_idx.items()}
    top_classes = [index_to_class[each] for each in idxs.data.squeeze().data.cpu().numpy()]

    probs = np.exp(probs.data.squeeze().data.cpu().numpy())

    return probs, top_classes

def display_results(input_image, cat_to_name_path, model_rebuilt, topk, device):
    probs, classes = predict_image_class(input_image, model_rebuilt, topk, device)
    img = plt.imread(input_image)
    topk_class_names = []
    cat_to_name = cat_to_name_loader(cat_to_name_path)
    for i in range(len(classes)):
        topk_class_names.append(cat_to_name[classes[i]])  

    print("The most probable flower kind is '{}' with the associated probability of %{:5.2f}; and, the next {} probable flower kinds are:".format(topk_class_names[0], 100*probs[0], topk-1)) 
    for i in range(1,topk):
        print("- '{}' with the probability of %{:5.4f}".format(topk_class_names[i], 100.0*probs[i])) 
    
    return topk_class_names, probs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("input", type = str, default = 'flowers/test/1/image_06743.jpg', 
                        help = 'path to the input flower image')

    parser.add_argument("checkpoint", type = str, default = 'checkpoint.pth', 
                        help = 'path to saved chackpoint')

    parser.add_argument('--top_k', type = int, default = 3, 
                        help = 'number of top probable flower kinds') 
    
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
                        help = 'the category names file .json') 
      
    parser.add_argument('--gpu', action = 'store_true', help = 'learning rate of the network') 
    
    in_args = parser.parse_args()
    
    if in_args.gpu:
        if in_args.gpu and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'    
    else:
        device = 'cpu'
        
    cat_to_name = cat_to_name_loader(in_args.category_names)
    
    model_rebuilt = load_network(in_args.checkpoint)
    
    model_rebuilt.eval()
    topk_class_names, probabilities = display_results(in_args.input, in_args.category_names, model_rebuilt, in_args.top_k, device)
    
    
    
    
    
    
    
    
    
    
    
    
    
